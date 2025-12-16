use crate::{CoreError, FunctionResult, LLM, Message, Role, agent::builder::AgentBuilder};
use futures_util::future::join_all;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug)]
pub enum AgentStep {
    Finished(Message),
    ToolExecution(Message),
    ToolReturn(Message),
}

#[derive(Debug, Clone)]
pub struct Agent {
    pub name: Option<String>,
    pub description: Option<String>,
    pub llm: Arc<LLM>,
    pub history: Arc<Mutex<Vec<Message>>>,
    pub max_depth: i32,
}

impl Agent {
    pub fn builder(llm: Arc<LLM>) -> AgentBuilder {
        AgentBuilder::new(llm)
    }

    pub async fn run(&self, initial_message: Message) -> crate::Result<Message> {
        let mut current_msg = initial_message;

        for _ in 0..self.max_depth {
            match self.step(current_msg.clone()).await? {
                AgentStep::Finished(response) => return Ok(response),
                AgentStep::ToolExecution(tool_msg) => {
                    current_msg = tool_msg;
                }
                AgentStep::ToolReturn(result_msg) => {
                    current_msg = result_msg;
                }
            }
        }

        Err(Box::from(CoreError::MaxDepthReached))
    }

    pub async fn step(&self, message: Message) -> crate::Result<AgentStep> {
        let history_context = {
            let mut history = self.history.lock().await;
            if message.role != Role::Tool {
                history.push(message.clone());
            }
            history.clone()
        };

        let response = self
            .llm
            .inference(message)
            .with_history(history_context)
            .generate()
            .await?;

        {
            let mut history = self.history.lock().await;
            history.push(response.clone());
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
                        None => return Err(CoreError::NotFound(call.name.clone())),
                    };

                    let result = tool.execute(&call.arguments).await;

                    match result {
                        Ok(val) => Ok(FunctionResult {
                            name: call.name.clone(),
                            results: val,
                        }),
                        Err(e) => Err(crate::CoreError::Internal(e)),
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

        let tool_msg = Message::function_results(tool_results);
        {
            let mut history = self.history.lock().await;
            history.push(tool_msg.clone());
        }

        Ok(AgentStep::ToolReturn(tool_msg))
    }
}
