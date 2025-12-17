use crate::llm::{AnyFunction, LLM, LLMProvider};
use std::sync::Arc;

/// Builder for constructing LLM instances with customizable parameters.
#[derive(Default, Debug, Clone)]
pub struct LLMBuilder {
    /// Model name or identifier.
    pub model: Option<String>,
    /// System prompt to guide the LLM's behavior.
    pub system_prompt: Option<String>,
    /// LLM provider (e.g., OpenAI, Google).
    pub provider: Option<LLMProvider>,
    /// Authorization token or credentials.
    pub authorization: Option<String>,
    /// Collection of tools (functions) to enhance LLM capabilities.
    pub tools: Vec<Arc<dyn AnyFunction>>,
    /// Optional custom endpoint for the LLM API.
    pub endpoint: Option<String>,
}

impl LLMBuilder {
    /// Builds the LLM instance from the configured parameters.
    pub fn build(self) -> crate::Result<LLM> {
        let model = match self.model {
            Some(n) => n,
            None => {
                return Err(crate::Error::Input("LLM name not set".to_owned()));
            }
        };

        let provider = match self.provider {
            Some(p) => p,
            None => {
                return Err(crate::Error::Input("LLM provider not set".to_owned()));
            }
        };

        Ok(LLM {
            model,
            system_prompt: self.system_prompt.unwrap_or_default(),
            provider,
            authorization: self.authorization,
            tools: self.tools,
            endpoint: self.endpoint,
        })
    }

    /// Adds a single tool to the LLM builder.
    pub fn add_tool(mut self, tool: Arc<dyn AnyFunction>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Adds multiple tools to the LLM builder.
    pub fn add_tools(mut self, tools: Vec<Arc<dyn AnyFunction>>) -> Self {
        for tool in tools {
            self.tools.push(tool);
        }
        self
    }

    /// Sets the authorization token.
    pub fn set_authorization(mut self, authorization: String) -> Self {
        self.authorization = Some(authorization);
        self
    }

    /// Sets the model name.
    pub fn set_model(mut self, name: String) -> Self {
        self.model = Some(name);
        self
    }

    /// Sets the system prompt.
    pub fn set_system_prompt(mut self, prompt: String) -> Self {
        self.system_prompt = Some(prompt);
        self
    }

    /// Sets the LLM provider.
    pub fn set_provider(mut self, provider: LLMProvider) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Sets a custom endpoint for the LLM API.
    pub fn set_endpoint(mut self, endpoint: String) -> Self {
        self.endpoint = Some(endpoint);
        self
    }
}
