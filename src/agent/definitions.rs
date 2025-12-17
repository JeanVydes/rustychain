use crate::{
    Error, FinishReason, FunctionResult, GenerationConfig, Inference, LLM, Role,
    agent::builder::AgentBuilder,
};
use futures_util::future::join_all;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug)]
pub enum AgentStep {
    Finished(Inference),
    ToolReturn(Inference),
}

#[derive(Debug, Clone)]
pub struct Agent {
    pub name: Option<String>,
    pub description: Option<String>,
    pub llm: Arc<LLM>,
    pub history: Arc<Mutex<Vec<Inference>>>,
    pub max_depth: i32,
    pub generation_config: Option<GenerationConfig>,
    pub current: Option<Inference>,
}

impl Agent {
    pub fn builder(llm: Arc<LLM>) -> AgentBuilder {
        AgentBuilder::new(llm)
    }

    pub async fn run(&self, initial: Inference) -> crate::Result<Inference> {
        let mut current = initial;

        for _ in 0..self.max_depth {
            match self.step(current.clone()).await? {
                AgentStep::Finished(response) => return Ok(response),
                AgentStep::ToolReturn(result_inference) => {
                    current = result_inference;
                }
            }
        }

        Err(crate::Error::MaxDepthReached)
    }

    pub async fn next(&mut self) -> Option<crate::Result<AgentStep>> {
        let inference = self.current.take()?;
        let step = match self.step(inference).await {
            Ok(s) => s,
            Err(e) => return Some(Err(e)),
        };

        self.current = match &step {
            AgentStep::Finished(_) => None,
            AgentStep::ToolReturn(inf) => Some(inf.clone()),
        };

        Some(Ok(step))
    }

    pub async fn step(&self, inference: Inference) -> crate::Result<AgentStep> {
        let history_context = {
            let mut history = self.history.lock().await;
            if inference.content.role != Role::Tool {
                history.push(inference.clone());
            }
            history.clone()
        };

        let mut req = self.llm.inference(inference).with_history(&history_context);

        if let Some(config) = &self.generation_config {
            req = req.with_config(config.clone());
        }

        let response = req.generate().await?;
        {
            let mut history = self.history.lock().await;
            history.push(response.clone());
        }

        if let Some(reason) = &response.finish_reason {
            match reason {
                FinishReason::UnexpectedToolCall => {
                    return Ok(AgentStep::ToolReturn(Inference::with_content(
                        Role::User,
                        "Agent attempted to call a tool unexpectedly.".to_owned(),
                    )));
                }
                FinishReason::TooManyToolCalls => {
                    log::debug!("Agent made too many tool calls in a single response.");
                }
                FinishReason::MaxTokens => {
                    log::debug!("Agent response exceeded maximum token limit.");
                }
                _ => { /* No action needed for other finish reasons */ }
            }
        }

        if !response.has_function_calls() {
            return Ok(AgentStep::Finished(response));
        }

        let mut workers = vec![];
        for call in response.function_calls {
            let llm = self.llm.clone();

            workers.push((
                call.name.clone(),
                tokio::spawn(async move {
                    let tool = match llm.get_tool(&call.name) {
                        Some(t) => t,
                        None => return Err(Error::NotFound(call.name.clone())),
                    };

                    let result = tool.execute(&call.arguments).await;

                    match result {
                        Ok(val) => Ok(FunctionResult {
                            name: call.name.clone(),
                            results: val,
                        }),
                        Err(e) => Err(crate::Error::Internal(e.into())),
                    }
                }),
            ));
        }

        let results = join_all(
            workers
                .into_iter()
                .map(|(name, handle)| async move { (name, handle.await) }),
        )
        .await;

        let mut tool_results: Vec<FunctionResult> = vec![];
        for (name, res) in results {
            match res {
                Ok(inner_result) => match inner_result {
                    Ok(func_res) => tool_results.push(func_res),
                    Err(e) => tool_results.push(FunctionResult {
                        name: name.clone(),
                        results: serde_json::json!({"error": format!("{:?}", e)}),
                    }),
                },
                Err(join_err) => {
                    tool_results.push(FunctionResult {
                        name: name.clone(),
                        results: serde_json::json!({"error": format!("Internal Server Errror: Task Join Error: {:?}", join_err)}),
                    });
                }
            }
        }

        let tool_msg = Inference::with_function_results(tool_results);
        {
            let mut history = self.history.lock().await;
            history.push(tool_msg.clone());
        }

        Ok(AgentStep::ToolReturn(tool_msg))
    }
}
