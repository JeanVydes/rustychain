// agent.rs

use std::sync::Arc;
use tokio::sync::Mutex;

use crate::{
    Error, FunctionCall, FunctionResult, GenerationConfig, Inference, LLM,
    execution::{
        context::Context,
        environment::ExecutionEnvironment,
        events::{Event, InterceptionResponse},
        listeners::{EventListener, ListenerRegistry},
        node::{ExecutionNode, NodeId, NodeStatus, NodeType},
    },
};

#[derive(Debug, Clone)]
pub enum AgentStatus {
    Idle,
    Executing,
    Finished(Box<Inference>),
    Faulted(String),
}

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub max_iterations: usize,
    pub generation_config: GenerationConfig,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            generation_config: GenerationConfig::default(),
        }
    }
}

pub struct Agent<S, EV, T>
where
    S: Into<String> + Clone + Send + Sync + 'static,
    EV: ExecutionEnvironment + 'static,
    T: Send + Sync + 'static,
{
    pub name: String,
    pub llm: Arc<LLM>,
    pub context: Context<S, EV, T>,
    listeners: Arc<ListenerRegistry<S, EV, T>>,
    config: AgentConfig,
    status: AgentStatus,
    iterations: usize,
    root: Option<NodeId>,
}

impl<S, EV, T> Agent<S, EV, T>
where
    S: Into<String> + Clone + Send + Sync + 'static,
    EV: ExecutionEnvironment + 'static + Clone,
    T: Send + Sync + 'static + Clone,
{
    pub fn new(
        name: impl Into<String>,
        execution_id: S,
        llm: Arc<LLM>,
        runtime: EV,
        state: Arc<Mutex<T>>,
    ) -> Self {
        Self {
            name: name.into(),
            llm,
            context: Context::new(execution_id, runtime, state),
            listeners: Arc::new(ListenerRegistry::new()),
            config: AgentConfig::default(),
            status: AgentStatus::Idle,
            iterations: 0,
            root: None,
        }
    }

    /// Register a listener for observing events (non-blocking)
    pub async fn on_event(&self, listener: Arc<dyn EventListener<S, EV, T>>) {
        self.listeners.add_observer(listener).await;
    }

    /// Register a listener for intercepting and controlling execution
    pub async fn intercept(&self, listener: Arc<dyn EventListener<S, EV, T>>, priority: i32) {
        self.listeners.add_interceptor(listener, priority).await;
    }

    /// Advances the agent to the next step
    pub async fn next_step(&mut self, input: Option<Inference>) -> crate::Result<AgentStatus> {
        match (&self.status, input) {
            (AgentStatus::Idle, Some(inf)) => self.start_turn(inf).await,
            (AgentStatus::Executing, None) => self.execute_step().await,
            (AgentStatus::Finished(_) | AgentStatus::Faulted(_), Some(inf)) => {
                self.soft_reset();
                self.start_turn(inf).await
            }
            _ => Err(Error::Internal("Invalid state transition".into())),
        }
    }

    async fn start_turn(&mut self, input: Inference) -> crate::Result<AgentStatus> {
        self.emit_observable(Event::Started {
            message: "Processing user input".to_string(),
        })
        .await;

        let parent = self.get_last_node().await;
        let node_id = self.create_node(NodeType::Inference(input), parent).await?;
        self.set_current_node(node_id).await;
        self.status = AgentStatus::Executing;
        self.execute_step().await
    }

    async fn execute_step(&mut self) -> crate::Result<AgentStatus> {
        if self.iterations >= self.config.max_iterations {
            return Ok(self.transition_to(AgentStatus::Faulted("Max iterations reached".into())));
        }
        self.iterations += 1;

        let current_id = self.get_current_node_id().await?;
        let node = self.get_node(&current_id).await?;

        self.emit_observable(Event::Thinking {
            node_id: current_id.clone(),
        })
        .await;

        match node.node_type {
            NodeType::Inference(inf) => self.process_inference_node(&current_id, inf).await,
            NodeType::ToolResults(_) => self.synthesize_results(&current_id).await,
            NodeType::InferenceResult(ref res) if res.has_function_calls() => {
                self.execute_tool_batch(&current_id, res.function_calls.clone())
                    .await
            }
            _ => Ok(self.status.clone()),
        }
    }

    #[async_recursion::async_recursion]
    async fn process_inference_node(
        &mut self,
        node_id: &NodeId,
        inference: Inference,
    ) -> crate::Result<AgentStatus> {
        self.update_node_status(node_id, NodeStatus::Running)
            .await?;
        let node = self.get_node(node_id).await?;

        let history = if let Some(parent_id) = &node.parent {
            self.context
                .runtime
                .read()
                .await
                .to_history(parent_id)
                .await?
        } else {
            Vec::new()
        };

        let result = self
            .llm
            .inference(inference)
            .with_history(&history)
            .with_config(self.config.generation_config.clone())
            .generate()
            .await?;

        let result_id = self
            .create_node(
                NodeType::InferenceResult(result.clone()),
                Some(node_id.clone()),
            )
            .await?;
        self.set_current_node(result_id.clone()).await;
        self.update_node_status(&result_id, NodeStatus::Completed)
            .await?;
        self.update_node_status(node_id, NodeStatus::Completed)
            .await?;

        self.emit_observable(Event::StepCompleted {
            step: self.iterations,
            node_id: result_id.clone(),
        })
        .await;

        if result.has_function_calls() {
            self.execute_tool_batch(&result_id, result.function_calls)
                .await
        } else {
            self.update_conversation_root().await;

            self.emit_observable(Event::Finished {
                inference: result.clone(),
            })
            .await;

            Ok(self.transition_to(AgentStatus::Finished(Box::new(result))))
        }
    }

    #[async_recursion::async_recursion]
    async fn execute_tool_batch(
        &mut self,
        parent_id: &NodeId,
        calls: Vec<FunctionCall>,
    ) -> crate::Result<AgentStatus> {
        let mut executable_calls = Vec::new();

        for call in calls {
            let event = Event::ToolCallRequested {
                node_id: parent_id.clone(),
                call: call.clone(),
            };

            match self.emit_interceptable(event).await {
                None | Some(InterceptionResponse::Continue) => {
                    executable_calls.push(call);
                }
                Some(InterceptionResponse::Modify { call: modified }) => {
                    executable_calls.push(modified);
                }
                Some(InterceptionResponse::Block { reason }) => {
                    log::debug!("Tool call blocked: {}", reason);
                }
                Some(InterceptionResponse::Replace { .. }) => {
                    log::warn!("Replace not supported for ToolCallRequested");
                }
            }
        }

        if executable_calls.is_empty() {
            return self.synthesize_results(parent_id).await;
        }

        let results = self.execute_tools(executable_calls, parent_id).await?;

        let result_node_id = self
            .create_node(NodeType::ToolResults(results), Some(parent_id.clone()))
            .await?;
        self.set_current_node(result_node_id.clone()).await;
        self.update_node_status(&result_node_id, NodeStatus::Completed)
            .await?;

        self.execute_step().await
    }

    async fn execute_tools(
        &self,
        calls: Vec<FunctionCall>,
        node_id: &NodeId,
    ) -> crate::Result<Vec<FunctionResult>> {
        let futures: Vec<_> = calls
            .into_iter()
            .map(|call| {
                let llm = self.llm.clone();
                let node_id = node_id.clone();
                let listeners = self.listeners.clone();
                let context = self.context.clone();

                async move {
                    let res = if let Some(tool) = llm.get_tool(&call.name) {
                        match tool.execute(&call.arguments).await {
                            Ok(r) => r,
                            Err(e) => serde_json::json!({ "error": e.to_string() }),
                        }
                    } else {
                        serde_json::json!({ "error": "Tool not found" })
                    };

                    let result = FunctionResult {
                        name: call.name.clone(),
                        results: res,
                        context: None,
                    };

                    // Emit completed event (can be intercepted to modify result)
                    let event = Event::ToolCallCompleted {
                        node_id,
                        call,
                        result: result.clone(),
                    };

                    match listeners.emit_interceptable(&event, &context).await {
                        Some(InterceptionResponse::Replace { result: new_result }) => new_result,
                        _ => result,
                    }
                }
            })
            .collect();

        Ok(futures_util::future::join_all(futures).await)
    }

    async fn synthesize_results(&mut self, parent_id: &NodeId) -> crate::Result<AgentStatus> {
        let node = self.get_node(parent_id).await?;
        if let NodeType::ToolResults(results) = node.node_type {
            let tool_inference = Inference::with_function_results(results);
            let node_id = self
                .create_node(
                    NodeType::Inference(tool_inference.clone()),
                    Some(parent_id.clone()),
                )
                .await?;
            self.set_current_node(node_id.clone()).await;
            self.process_inference_node(&node_id, tool_inference).await
        } else {
            Err(Error::Internal("No tool results to synthesize".into()))
        }
    }

    /// Emit an observable event to all listeners
    async fn emit_observable(&self, event: Event) {
        self.listeners.emit_observable(&event, &self.context).await;
    }

    /// Emit an interceptable event and get response
    async fn emit_interceptable(&self, event: Event) -> Option<InterceptionResponse> {
        self.listeners
            .emit_interceptable(&event, &self.context)
            .await
    }

    fn transition_to(&mut self, status: AgentStatus) -> AgentStatus {
        self.status = status;
        self.status.clone()
    }

    async fn create_node(
        &self,
        node_type: NodeType,
        parent: Option<NodeId>,
    ) -> crate::Result<NodeId> {
        let node = ExecutionNode::new(node_type, parent);
        self.context.runtime.read().await.add_node(node).await
    }

    async fn get_node(&self, id: &NodeId) -> crate::Result<ExecutionNode> {
        let runtime = self.context.runtime.read().await;
        let nodes = runtime.get_nodes().await;
        nodes
            .read()
            .await
            .get(id)
            .cloned()
            .ok_or_else(|| Error::NotFound("Node not found".into()))
    }

    async fn get_current_node_id(&self) -> crate::Result<NodeId> {
        self.context
            .current_node
            .read()
            .await
            .clone()
            .ok_or_else(|| Error::Internal("No current node".into()))
    }

    async fn set_current_node(&self, id: NodeId) {
        *self.context.current_node.write().await = Some(id);
    }

    async fn update_node_status(&self, id: &NodeId, status: NodeStatus) -> crate::Result<()> {
        self.context.update_node(id, |n| n.status = status).await
    }

    async fn get_last_node(&self) -> Option<NodeId> {
        if self.root.is_some() {
            self.context.current_node.read().await.clone()
        } else {
            None
        }
    }

    async fn update_conversation_root(&mut self) {
        if let Ok(id) = self.get_current_node_id().await {
            self.root = Some(id);
        }
    }

    pub fn with_config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    pub fn soft_reset(&mut self) {
        self.status = AgentStatus::Idle;
        self.iterations = 0;
    }

    pub async fn hard_reset(&mut self) -> crate::Result<()> {
        self.soft_reset();
        self.root = None;
        *self.context.current_node.write().await = None;
        self.context.runtime.read().await.clear().await
    }
}
