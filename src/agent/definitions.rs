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

    // Runs the agent until it finishes or reaches max depth
    // The history is updated only when tools are called
    // So the user have to manage the past history outside and also the last inference
    // The initial inference is added to the history
    pub async fn run(&self, initial: Inference) -> crate::Result<Inference> {
        let mut current = initial;

        {
            let mut history = self.history.lock().await;
            history.push(current.clone());
        }

        for _ in 0..self.max_depth {
            match self.step(current.clone()).await? {
                AgentStep::Finished(response) => return Ok(response),
                AgentStep::ToolReturn(result_inference) => {
                    {
                        let mut history = self.history.lock().await;
                        history.push(result_inference.clone());
                    }

                    current = result_inference;
                }
            }
        }

        Err(crate::Error::AgentError {
            index: 0,
            name: "".to_string(),
            source: Box::new(crate::Error::Internal(
                "Agent reached maximum depth without finishing".into(),
            )),
        })
    }

    pub async fn next(&mut self) -> Option<crate::Result<AgentStep>> {
        let input_inference = self.current.take()?;

        // Execute the LLM logic
        let step_result = self.step(input_inference.clone()).await;

        match step_result {
            Ok(AgentStep::Finished(response)) => {
                // Task done. We don't automatically push to history here
                // to allow the developer to inspect the final response first.
                Some(Ok(AgentStep::Finished(response)))
            }
            Ok(AgentStep::ToolReturn(tool_inference)) => {
                // To keep the LLM state valid, we must archive the turn:
                // 1. The Assistant's call (input_inference)
                // 2. The Tool's results (tool_inference)
                {
                    let mut history = self.history.lock().await;
                    history.push(input_inference);
                    history.push(tool_inference.clone());
                }

                // Set current for the next iteration
                self.current = Some(tool_inference.clone());
                Some(Ok(AgentStep::ToolReturn(tool_inference)))
            }
            Err(e) => Some(Err(e)),
        }
    }

    pub async fn step(&self, inference: Inference) -> crate::Result<AgentStep> {
        let history_snapshot = self.history.lock().await;

        let mut req = self
            .llm
            .inference(inference)
            .with_history(&history_snapshot);

        if let Some(config) = &self.generation_config {
            req = req.with_config(config.clone());
        }

        let response = req.generate().await?;

        if let Some(reason) = &response.finish_reason {
            match reason {
                FinishReason::UnexpectedToolCall => {
                    return Ok(AgentStep::ToolReturn(Inference::with_content(
                        Role::User,
                        "Agent attempted to call a tool unexpectedly.".to_owned(),
                    )));
                }
                FinishReason::MalformedFunctionCall => {
                    return Ok(AgentStep::ToolReturn(Inference::with_content(
                        Role::User,
                        "Agent made a malformed function call.".to_owned(),
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
                call.clone(),
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
                            context: call.context.clone(),
                        }),
                        Err(e) => Err(crate::Error::Internal(e.into())),
                    }
                }),
            ));
        }

        let results = join_all(
            workers
                .into_iter()
                .map(|(function_call, handle)| async move { (function_call, handle.await) }),
        )
        .await;

        let mut tool_results: Vec<FunctionResult> = vec![];
        for (function_call, res) in results {
            match res {
                Ok(inner_result) => match inner_result {
                    Ok(func_res) => tool_results.push(func_res),
                    Err(e) => tool_results.push(FunctionResult {
                        name: function_call.name.clone(),
                        results: serde_json::json!({"error": format!("{:?}", e)}),
                        context: function_call.context.clone(),
                    }),
                },
                Err(join_err) => {
                    tool_results.push(FunctionResult {
                        name: function_call.name.clone(),
                        results: serde_json::json!({"error": format!("Internal Server Errror: Task Join Error: {:?}", join_err)}),
                        context: function_call.context.clone(),
                    });
                }
            }
        }

        Ok(AgentStep::ToolReturn(Inference::with_function_results(
            tool_results,
        )))
    }
}
