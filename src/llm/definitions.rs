use futures_core::stream::Stream;
use gemini_rust::{
    ContentBuilder, FunctionCallingMode, Gemini, GeminiBuilder,
    GenerationConfig as GeminiGenerationConfig, ThinkingConfig,
};
use ollama_rs::generation::chat::{ChatMessage, request::ChatMessageRequest};
#[cfg(feature = "openai")]
use openai_api_rs::v1::chat_completion::chat_completion::ChatCompletionRequest;
use openai_api_rs::v1::{
    api::{OpenAIClient, OpenAIClientBuilder},
    chat_completion::{ChatCompletionMessage, chat_completion_stream::ChatCompletionStreamRequest},
};
use schemars::Schema;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use url::Url;

use crate::llm::{
    builder::LLMBuilder, conversation::Inference, function::AnyFunction,
    inference_task::InferenceTask,
};

/// Enum representing supported LLM providers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum LLMProvider {
    Google,
    OpenAI,
    Anthropic,
    Ollama,
}

/// Different modes for LLM "thinking" or planning.
///
/// - `None`: No additional thinking time.
/// - `Dynamic`: Let the LLM decide how much time to spend thinking.
/// - `Sized(i32)`: Allocate a specific amount of time or budget for thinking.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash, PartialOrd, Ord)]
pub enum ThinkingMode {
    None,
    Dynamic,
    Effort(ThinkingEffort),
    Sized(i32),
}

impl ThinkingMode {
    #[cfg(feature = "google")]
    pub fn to_google(&self) -> i32 {
        match self {
            ThinkingMode::None => 0,
            ThinkingMode::Sized(size) => *size,
            // Google doesnt support effort levels directly, set them dynamically
            _ => -1,
        }
    }

    #[cfg(feature = "openai")]
    pub fn to_openai(&self) -> Option<openai_api_rs::v1::chat_completion::Reasoning> {
        match self {
            ThinkingMode::None => Some(openai_api_rs::v1::chat_completion::Reasoning {
                mode: None,
                exclude: None,
                enabled: Some(false),
            }),
            ThinkingMode::Dynamic => None,
            ThinkingMode::Effort(effort) => {
                let reasoning_effort = match effort {
                    ThinkingEffort::Low => openai_api_rs::v1::chat_completion::ReasoningEffort::Low,
                    ThinkingEffort::Medium => {
                        openai_api_rs::v1::chat_completion::ReasoningEffort::Medium
                    }
                    ThinkingEffort::High => {
                        openai_api_rs::v1::chat_completion::ReasoningEffort::High
                    }
                };

                Some(openai_api_rs::v1::chat_completion::Reasoning {
                    mode: Some(openai_api_rs::v1::chat_completion::ReasoningMode::Effort {
                        effort: reasoning_effort,
                    }),
                    exclude: Some(false),
                    enabled: Some(true),
                })
            }
            ThinkingMode::Sized(size) => Some(openai_api_rs::v1::chat_completion::Reasoning {
                mode: Some(
                    openai_api_rs::v1::chat_completion::ReasoningMode::MaxTokens {
                        max_tokens: *size as i64,
                    },
                ),
                exclude: Some(false),
                enabled: Some(true),
            }),
        }
    }

    #[cfg(feature = "ollama")]
    pub fn to_ollama(&self) -> bool {
        !matches!(self, ThinkingMode::None)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash, PartialOrd, Ord)]
pub enum ThinkingEffort {
    Low,
    Medium,
    High,
}

#[derive(Clone, Debug)]
pub enum ToolCallingMode {
    None,
    Auto,
    Any,
    Forced(Arc<dyn AnyFunction>),
}

impl Serialize for ToolCallingMode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let s = match self {
            ToolCallingMode::None => "none",
            ToolCallingMode::Auto => "auto",
            ToolCallingMode::Any => "any",
            ToolCallingMode::Forced(tool) => tool.name(),
        };
        serializer.serialize_str(s)
    }
}

impl<'de> Deserialize<'de> for ToolCallingMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "none" => Ok(ToolCallingMode::None),
            "auto" => Ok(ToolCallingMode::Auto),
            "any" => Ok(ToolCallingMode::Any),
            _ => Err(serde::de::Error::custom(format!(
                "Unknown tool calling mode: {}",
                s
            ))),
        }
    }
}

impl PartialEq<str> for ToolCallingMode {
    fn eq(&self, other: &str) -> bool {
        match self {
            ToolCallingMode::None => other == "none",
            ToolCallingMode::Auto => other == "auto",
            ToolCallingMode::Any => other == "any",
            ToolCallingMode::Forced(tool) => tool.name() == other,
        }
    }
}

impl PartialOrd<str> for ToolCallingMode {
    fn partial_cmp(&self, other: &str) -> Option<std::cmp::Ordering> {
        let self_str = match self {
            ToolCallingMode::None => "none",
            ToolCallingMode::Auto => "auto",
            ToolCallingMode::Any => "any",
            ToolCallingMode::Forced(tool) => tool.name(),
        };
        self_str.partial_cmp(other)
    }
}

/// Struct representing a Large Language Model (LLM) with its configuration and capabilities.
#[derive(Clone, Debug)]
pub struct LLM {
    pub model: String,
    pub system_prompt: String,
    pub provider: LLMProvider,
    pub authorization: Option<String>,
    pub endpoint: Option<String>,
    pub tools: Vec<Arc<dyn AnyFunction>>,
}

/// Configuration parameters for text generation by the LLM.
///
/// These parameters influence the behavior and output of the LLM during generation tasks.
///
/// - `temperature`: Controls the randomness of the output. Higher values yield more diverse results.
/// - `top_p`: Nucleus sampling parameter to limit the token selection to a subset of probable tokens.
/// - `top_k`: Limits the token selection to the top K most probable tokens.
/// - `max_output_tokens`: Maximum number of tokens to generate in the output.
/// - `thinking`: Mode for LLM thinking/planning before generating a response.
/// - `candidate_count`: Number of candidate responses to generate.
/// - `stop_sequences`: Optional sequences that, when generated, will stop further output.
/// - `output_schema`: Optional JSON schema to structure the output.
/// - `response_mime_type`: Optional MIME type for the response format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub max_output_tokens: i32,
    pub thinking: ThinkingMode,
    pub candidate_count: i32,
    pub stop_sequences: Option<Vec<String>>,
    pub output_schema: Option<Schema>,
    pub response_mime_type: Option<String>,
    pub tool_calling_mode: ToolCallingMode,
    pub include_thoughts: bool,
}

impl GenerationConfig {
    /// Creates a new `GenerationConfig` with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    pub fn with_top_k(mut self, top_k: i32) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn with_max_output_tokens(mut self, max_output_tokens: i32) -> Self {
        self.max_output_tokens = max_output_tokens;
        self
    }

    pub fn with_thinking(mut self, thinking: ThinkingMode) -> Self {
        self.thinking = thinking;
        self
    }

    pub fn with_candidate_count(mut self, candidate_count: i32) -> Self {
        self.candidate_count = candidate_count;
        self
    }

    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(stop_sequences);
        self
    }

    pub fn with_output_schema(mut self, output_schema: Schema) -> Self {
        self.output_schema = Some(output_schema);
        self
    }

    pub fn with_response_mime_type(mut self, mime_type: String) -> Self {
        self.response_mime_type = Some(mime_type);
        self
    }

    pub fn with_tool_calling_mode(mut self, mode: ToolCallingMode) -> Self {
        self.tool_calling_mode = mode;
        self
    }

    #[cfg(feature = "google")]
    pub fn to_google_tool_calling_mode(&self) -> FunctionCallingMode {
        match self.tool_calling_mode {
            ToolCallingMode::None => FunctionCallingMode::None,
            ToolCallingMode::Auto => FunctionCallingMode::Auto,
            _ => FunctionCallingMode::Auto,
        }
    }

    #[cfg(feature = "openai")]
    pub fn to_openai_tool_calling_mode(
        &self,
    ) -> openai_api_rs::v1::chat_completion::ToolChoiceType {
        match &self.tool_calling_mode {
            ToolCallingMode::None => openai_api_rs::v1::chat_completion::ToolChoiceType::None,
            ToolCallingMode::Auto => openai_api_rs::v1::chat_completion::ToolChoiceType::Auto,
            ToolCallingMode::Any => openai_api_rs::v1::chat_completion::ToolChoiceType::Required,
            ToolCallingMode::Forced(tool) => {
                openai_api_rs::v1::chat_completion::ToolChoiceType::ToolChoice {
                    tool: tool.openai_tool_definition(),
                }
            }
        }
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.5,
            top_p: 0.9,
            top_k: 40,
            max_output_tokens: 2048,
            thinking: ThinkingMode::Dynamic,
            candidate_count: 1,
            stop_sequences: None,
            output_schema: None,
            response_mime_type: None,
            tool_calling_mode: ToolCallingMode::Auto,
            include_thoughts: false,
        }
    }
}

#[async_trait::async_trait]
pub trait LLMGeneration {
    /// Performs text generation based on the provided history and inference.
    async fn generation(
        &self,
        history: &[Inference],
        inference: &Inference,
        config: GenerationConfig,
    ) -> crate::Result<Inference>;
}

#[async_trait::async_trait]
pub trait LLMStreaming {
    /// Streams text generation results based on the provided history and inference.
    async fn stream(
        &self,
        history: &[Inference],
        inference: &Inference,
        config: GenerationConfig,
    ) -> crate::Result<Pin<Box<dyn Stream<Item = crate::Result<Inference>> + Send + 'static>>>;
}

#[async_trait::async_trait]
pub trait LLMEmbedding {
    /// Generates an embedding for the given text.
    async fn embedding(&self, text: &str, dim: i32) -> crate::Result<Vec<f32>>;
}

impl LLM {
    /// Creates a new inference with this model.
    pub fn inference<'a>(&self, inference: impl Into<Inference> + 'a) -> InferenceTask<'a> {
        InferenceTask::new(inference.into(), Arc::new(self.clone()))
    }

    /// Creates a Gemini client configured for this LLM.
    #[cfg(feature = "google")]
    pub fn get_gemini_client(&self) -> crate::Result<Gemini> {
        let authorization = match &self.authorization {
            Some(a) => a,
            None => return Err(crate::Error::Unauthorized("Not auth".to_owned())),
        };

        let mut client =
            GeminiBuilder::new(authorization.clone()).with_model(format!("models/{}", self.model));

        if let Some(endpoint) = &self.endpoint {
            client = client.with_base_url(Url::parse(endpoint)?);
        }

        let client = client.build()?;

        Ok(client)
    }

    /// Creates an OpenAI client configured for this LLM.
    #[cfg(feature = "openai")]
    pub fn get_openai_client(&self) -> crate::Result<OpenAIClient> {
        let authorization = match &self.authorization {
            Some(a) => a,
            None => return Err(crate::Error::Unauthorized("Not auth".to_owned())),
        };

        let mut client = OpenAIClientBuilder::new().with_api_key(authorization.clone());

        if let Some(endpoint) = &self.endpoint {
            client = client.with_endpoint(endpoint.clone());
        }

        let client = client
            .build()
            .map_err(|e| crate::Error::Generic(format!("Failed to build OpenAI client: {}", e)))?;

        Ok(client)
    }

    /// Adds a single tool to the LLM.
    pub fn add_tool(&mut self, tool: Arc<dyn AnyFunction>) {
        self.tools.push(tool);
    }

    /// Adds multiple tools to the LLM.
    pub fn add_tools(&mut self, tools: Vec<Arc<dyn AnyFunction>>) {
        for tool in tools {
            self.tools.push(tool);
        }
    }

    pub fn exists_tool(&self, tool_name: &str) -> bool {
        for tool in &self.tools {
            if tool.name() == tool_name {
                return true;
            }
        }
        false
    }

    pub fn get_tool(&self, tool_name: &str) -> Option<Arc<dyn AnyFunction>> {
        for tool in &self.tools {
            if tool.name() == tool_name {
                return Some(tool.clone());
            }
        }
        None
    }

    /// Creates a new LLM builder.
    pub fn builder() -> LLMBuilder {
        LLMBuilder {
            model: None,
            system_prompt: None,
            provider: None,
            authorization: None,
            tools: vec![],
            endpoint: None,
        }
    }

    #[cfg(feature = "google")]
    pub fn new_google_request(
        &self,
        history: &[Inference],
        inference: &Inference,
        config: GenerationConfig,
    ) -> crate::Result<ContentBuilder> {
        let client = self.get_gemini_client()?;
        let mut history: Vec<gemini_rust::Message> =
            history.iter().map(|m| m.to_gemini_message()).collect();
        history.push(inference.to_gemini_message());
        let mut req = client
            .generate_content()
            .with_system_instruction(self.system_prompt.clone())
            .with_messages(history.clone())
            .with_thinking_budget(config.thinking.to_google())
            .with_generation_config(GeminiGenerationConfig {
                temperature: Some(config.temperature),
                top_p: Some(config.top_p),
                top_k: Some(config.top_k),
                max_output_tokens: Some(config.max_output_tokens),
                thinking_config: Some(ThinkingConfig {
                    thinking_budget: Some(config.thinking.to_google()),
                    include_thoughts: Some(config.include_thoughts),
                }),
                candidate_count: Some(config.candidate_count),
                ..Default::default()
            })
            .with_function_calling_mode(config.to_google_tool_calling_mode());

        if let Some(schema) = config.output_schema {
            req = req
                .with_response_mime_type("application/json".to_owned())
                .with_response_schema(schema.to_value());
        }

        if let Some(stop_sequences) = &config.stop_sequences {
            req = req.with_stop_sequences(stop_sequences.clone());
        }

        for tool in &self.tools {
            req = req.with_tool(tool.gemini_tool_definition());
        }

        Ok(req)
    }

    #[cfg(feature = "openai")]
    pub fn new_openai_request(
        &self,
        history: &[Inference],
        inference: &Inference,
        config: GenerationConfig,
    ) -> crate::Result<ChatCompletionRequest> {
        // Build messages with system prompt first
        let mut messages: Vec<ChatCompletionMessage> = vec![];
        if !self.system_prompt.is_empty() {
            messages.push(ChatCompletionMessage {
                role: openai_api_rs::v1::chat_completion::MessageRole::system,
                content: openai_api_rs::v1::chat_completion::Content::Text(
                    self.system_prompt.clone(),
                ),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }
        messages.extend(history.iter().map(|m| m.to_openai_message()));
        messages.push(inference.to_openai_message());

        let mut req = ChatCompletionRequest::new(self.model.clone(), messages)
            .max_tokens(config.max_output_tokens as i64)
            .temperature(config.temperature as f64)
            .top_p(config.top_p as f64)
            .n(config.candidate_count as i64)
            .tools(
                self.tools
                    .iter()
                    .map(|t| t.openai_tool_definition())
                    .collect(),
            )
            .tool_choice(config.to_openai_tool_calling_mode());

        if let Some(reasoning) = config.thinking.to_openai() {
            req = req.reasoning(reasoning);
        }

        if let Some(stop_sequences) = &config.stop_sequences {
            req = req.stop(stop_sequences.clone());
        }

        if let Some(output_schema) = config.output_schema {
            req = req.response_format(output_schema.to_value());
        }

        Ok(req)
    }

    #[cfg(feature = "openai")]
    pub fn new_openai_stream_request(
        &self,
        history: &[Inference],
        inference: &Inference,
        config: GenerationConfig,
    ) -> crate::Result<ChatCompletionStreamRequest> {
        // Build messages with system prompt first
        let mut messages: Vec<ChatCompletionMessage> = vec![];
        if !self.system_prompt.is_empty() {
            messages.push(ChatCompletionMessage {
                role: openai_api_rs::v1::chat_completion::MessageRole::system,
                content: openai_api_rs::v1::chat_completion::Content::Text(
                    self.system_prompt.clone(),
                ),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }
        messages.extend(history.iter().map(|m| m.to_openai_message()));
        messages.push(inference.to_openai_message());

        let mut req = ChatCompletionStreamRequest::new(self.model.clone(), messages)
            .max_tokens(config.max_output_tokens as i64)
            .temperature(config.temperature as f64)
            .top_p(config.top_p as f64)
            .n(config.candidate_count as i64)
            .tools(
                self.tools
                    .iter()
                    .map(|t| t.openai_tool_definition())
                    .collect(),
            )
            .tool_choice(config.to_openai_tool_calling_mode());

        if let Some(reasoning) = config.thinking.to_openai() {
            req = req.reasoning(reasoning);
        }

        if let Some(stop_sequences) = &config.stop_sequences {
            req = req.stop(stop_sequences.clone());
        }

        if let Some(output_schema) = config.output_schema {
            req = req.response_format(output_schema.to_value());
        }

        Ok(req)
    }

    #[cfg(feature = "ollama")]
    pub fn new_ollama_request(
        &self,
        history: &[Inference],
        inference: &Inference,
        config: GenerationConfig,
    ) -> crate::Result<ChatMessageRequest> {
        use ollama_rs::{generation::parameters::JsonStructure, models::ModelOptions};

        let mut ollama_messages: Vec<ChatMessage> =
            history.iter().map(|m| m.to_ollama_message()).collect();
        ollama_messages.push(inference.to_ollama_message());

        let mut req = ChatMessageRequest::new(self.model.clone(), ollama_messages)
            .think(config.thinking.to_ollama())
            .tools(
                self.tools
                    .iter()
                    .map(|t| t.ollama_tool_definition())
                    .collect(),
            )
            .options(
                ModelOptions::default()
                    .top_k(config.top_k as u32)
                    .top_p(config.top_p)
                    .temperature(config.temperature)
                    .stop(config.stop_sequences.clone().unwrap_or_default()),
            );

        if let Some(schema) = &config.output_schema {
            req = req.format(
                ollama_rs::generation::parameters::FormatType::StructuredJson(Box::from(
                    JsonStructure::new_for_schema(schema.clone()),
                )),
            );
        }

        Ok(req)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::function::AnyFunction;
    use serde_json::json;
    use std::sync::Arc;

    // --- Mocks ---

    #[derive(Debug)]
    struct MockTool {
        name: String,
    }

    #[async_trait::async_trait]
    impl AnyFunction for MockTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "mock description"
        }
        fn parameters_schema(&self) -> &schemars::Schema {
            lazy_static::lazy_static! {
                static ref SCHEMA: schemars::Schema = schemars::schema_for!(i32);
            }
            &SCHEMA
        }
        async fn execute(&self, _args: &serde_json::Value) -> crate::Result<serde_json::Value> {
            Ok(json!({"status": "ok"}))
        }

        #[cfg(feature = "google")]
        fn gemini_tool_definition(&self) -> gemini_rust::Tool {
            gemini_rust::Tool::Function {
                function_declarations: vec![],
            }
        }

        #[cfg(feature = "openai")]
        fn openai_tool_definition(&self) -> openai_api_rs::v1::chat_completion::Tool {
            openai_api_rs::v1::chat_completion::Tool {
                r#type: openai_api_rs::v1::chat_completion::ToolType::Function,
                function: openai_api_rs::v1::types::Function {
                    name: self.name.clone(),
                    description: None,
                    parameters: openai_api_rs::v1::types::FunctionParameters {
                        schema_type: openai_api_rs::v1::types::JSONSchemaType::Object,
                        properties: None,
                        required: None,
                    },
                },
            }
        }

        #[cfg(feature = "ollama")]
        fn ollama_tool_definition(&self) -> ollama_rs::generation::tools::ToolInfo {
            ollama_rs::generation::tools::ToolInfo {
                tool_type: ollama_rs::generation::tools::ToolType::Function,
                function: ollama_rs::generation::tools::ToolFunctionInfo {
                    name: self.name.clone(),
                    description: "".into(),
                    parameters: schemars::schema_for!(i32),
                },
            }
        }
    }

    fn create_test_llm(provider: LLMProvider) -> LLM {
        LLM {
            model: "test-model".into(),
            system_prompt: "You are a test assistant".into(),
            provider,
            authorization: Some("test-key".into()),
            endpoint: Some("http://localhost:8080".into()),
            tools: vec![],
        }
    }

    // --- Core LLM Logic Tests ---

    #[test]
    fn test_tool_registry_management() {
        let mut llm = create_test_llm(LLMProvider::OpenAI);
        let tool_name = "weather_api";

        llm.add_tool(Arc::new(MockTool {
            name: tool_name.into(),
        }));

        assert!(llm.exists_tool(tool_name));
        assert!(llm.get_tool(tool_name).is_some());
        assert_eq!(llm.get_tool(tool_name).unwrap().name(), tool_name);
        assert!(!llm.exists_tool("non_existent"));
    }

    #[test]
    fn test_generation_config_builder_flow() {
        let config = GenerationConfig::default()
            .with_temperature(0.7)
            .with_thinking(ThinkingMode::Sized(500))
            .with_stop_sequences(vec!["\n".into()]);

        assert_eq!(config.temperature, 0.7);
        assert!(matches!(config.thinking, ThinkingMode::Sized(500)));
        assert_eq!(config.stop_sequences.unwrap()[0], "\n");
    }

    // --- Provider Request Mapping Tests ---

    #[cfg(feature = "openai")]
    #[test]
    fn test_openai_request_composition() {
        use crate::{Role, inference::InferenceContent};

        let mut llm = create_test_llm(LLMProvider::OpenAI);
        llm.add_tool(Arc::new(MockTool {
            name: "test_tool".into(),
        }));

        let history = vec![];
        let inference = Inference {
            content: InferenceContent::from((Role::User, "Hello, how are you?")),
            ..Default::default()
        };
        let config = GenerationConfig::default();

        let req = llm
            .new_openai_request(&history, &inference, config)
            .unwrap();

        assert_eq!(req.model, "test-model");
        // System prompt + user inference = 2 messages
        assert_eq!(req.messages.len(), 2);
        assert!(req.tools.is_some());
        assert_eq!(req.tools.unwrap().len(), 1);
    }

    #[cfg(feature = "google")]
    #[test]
    fn test_google_thinking_budget_mapping() {
        let llm = create_test_llm(LLMProvider::Google);
        let history = vec![];
        let inference = Inference::default();

        // Test sized budget
        let config_sized = GenerationConfig::default().with_thinking(ThinkingMode::Sized(1024));
        let _req_sized = llm
            .new_google_request(&history, &inference, config_sized)
            .unwrap();
        // Since ContentBuilder is opaque, we verify it doesn't error and mapping is correct
        assert_eq!(ThinkingMode::Sized(1024).to_google(), 1024);
        // Test none
        assert_eq!(ThinkingMode::None.to_google(), 0);
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn test_ollama_request_composition() {
        let llm = create_test_llm(LLMProvider::Ollama);
        let config =
            GenerationConfig::default().with_thinking(ThinkingMode::Effort(ThinkingEffort::High));

        let req = llm
            .new_ollama_request(&[], &Inference::default(), config)
            .unwrap();

        // Ollama thinking is boolean in current implementation
        assert!(req.think.unwrap_or_default());
    }

    // --- Error Cases ---

    #[test]
    fn test_missing_authorization_error() {
        let llm = LLM {
            model: "model".into(),
            system_prompt: "".into(),
            provider: LLMProvider::OpenAI,
            authorization: None,
            endpoint: None,
            tools: vec![],
        };

        #[cfg(feature = "openai")]
        assert!(matches!(
            llm.get_openai_client(),
            Err(crate::Error::Unauthorized(_))
        ));

        #[cfg(feature = "google")]
        assert!(matches!(
            llm.get_gemini_client(),
            Err(crate::Error::Unauthorized(_))
        ));
    }

    #[test]
    fn test_thinking_mode_openai_effort_logic() {
        #[cfg(feature = "openai")]
        {
            let mode = ThinkingMode::Effort(ThinkingEffort::Low);
            let openai_reasoning = mode.to_openai().unwrap();

            assert!(openai_reasoning.enabled.unwrap());
            if let Some(openai_api_rs::v1::chat_completion::ReasoningMode::Effort { effort }) =
                openai_reasoning.mode
            {
                assert!(matches!(
                    effort,
                    openai_api_rs::v1::chat_completion::ReasoningEffort::Low
                ));
            } else {
                panic!("Reasoning mode effort mismatch");
            }
        }
    }
}
