use base64::Engine;
use gemini_rust::{Blob, Content, FunctionResponse, GenerationResponse, Part};
use ollama_rs::generation::chat::{ChatMessage, ChatMessageResponse, MessageRole};
#[cfg(feature = "openai")]
use openai_api_rs::v1::chat_completion::ChatCompletionChoice;
use openai_api_rs::v1::chat_completion::chat_completion_stream::ChatCompletionStreamResponse;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

use crate::{
    FinishReason, Image, Role,
    llm::{FunctionResult, function::FunctionCall},
};

/// A `Inference` in the conversation, which may include text, audio, images, and function calls/results.
/// This is a unified representation that can be converted to/from various LLM formats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inference {
    pub model: Option<String>,
    pub content: InferenceContent,

    pub thinking: Option<String>,
    pub function_calls: Vec<FunctionCall>,
    pub function_results: Vec<FunctionResult>,

    pub finish_reason: Option<FinishReason>,
    pub usage: Option<UsageMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageMetadata {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceContent {
    pub role: Role,
    pub text: Option<String>,
    pub audio: Option<Vec<u8>>,
    pub images: Option<Vec<Image>>,
}

impl Inference {
    pub fn new(content: impl Into<InferenceContent>) -> Self {
        Self {
            content: content.into(),
            ..Default::default()
        }
    }

    pub fn with_content(role: Role, text: impl ToString) -> Self {
        Self::new(InferenceContent {
            role,
            text: Some(text.to_string()),
            audio: None,
            images: None,
        })
    }

    pub fn as_user(text: impl ToString) -> Self {
        Self::with_content(Role::User, text)
    }

    pub fn as_assistant(text: impl ToString) -> Self {
        Self::with_content(Role::Assistant, text)
    }

    pub fn as_system(text: impl ToString) -> Self {
        Self::with_content(Role::System, text)
    }

    pub fn as_tool(text: impl ToString) -> Self {
        Self::with_content(Role::Tool, text)
    }

    pub fn with_function_results(results: Vec<FunctionResult>) -> Self {
        Self {
            content: InferenceContent {
                role: Role::Tool,
                text: None,
                audio: None,
                images: None,
            },
            function_results: results,
            ..Default::default()
        }
    }

    pub fn add_function_result(mut self, result: FunctionResult) -> Self {
        self.function_results.push(result);
        self
    }

    pub fn add_function_call(mut self, call: FunctionCall) -> Self {
        self.function_calls.push(call);
        self
    }

    pub fn with_audio(mut self, audio: Vec<u8>) -> Self {
        self.content.audio = Some(audio);
        self
    }

    pub fn with_images(mut self, images: Vec<Image>) -> Self {
        self.content.images = Some(images);
        self
    }

    pub fn add_image(mut self, image: Image) -> Self {
        if let Some(imgs) = &mut self.content.images {
            imgs.push(image);
        } else {
            self.content.images = Some(vec![image]);
        }
        self
    }

    pub fn with_thinking(mut self, thinking: String) -> Self {
        self.thinking = Some(thinking);
        self
    }

    pub fn is_thinking(&self) -> bool {
        self.thinking.is_some()
    }

    pub fn has_function_calls(&self) -> bool {
        !self.function_calls.is_empty()
    }

    pub fn has_function_results(&self) -> bool {
        !self.function_results.is_empty()
    }

    /// Converts this `Message` to a Gemini-compatible message.
    #[cfg(feature = "google")]
    pub fn to_gemini_message(&self) -> gemini_rust::Message {
        let role = match self.content.role {
            Role::User => gemini_rust::Role::User,
            _ => gemini_rust::Role::Model,
        };

        let mut content: Content = Content {
            role: Some(role.clone()),
            parts: vec![].into(),
        };

        if let Some(audio) = &self.content.audio
            && let Some(parts) = &mut content.parts
        {
            let base64 = base64::engine::general_purpose::STANDARD.encode(audio);
            parts.push(Part::InlineData {
                inline_data: Blob::new("audio/mp3", base64),
            });
        }

        if let Some(msg) = &self.content.text
            && let Some(parts) = &mut content.parts
        {
            parts.push(Part::Text {
                text: msg.clone(),
                thought: Some(self.thinking.is_some()),
                thought_signature: None,
            });
        }

        if !self.function_calls.is_empty()
            && let Some(parts) = &mut content.parts
        {
            for fc in &self.function_calls {
                parts.push(Part::FunctionCall {
                    function_call: fc.to_gemini(),
                    thought_signature: None,
                });
            }
        }

        if !self.function_results.is_empty()
            && let Some(parts) = &mut content.parts
        {
            for fr in &self.function_results {
                parts.push(Part::FunctionResponse {
                    function_response: FunctionResponse {
                        name: fr.name.clone(),
                        response: Some(fr.results.clone()),
                    },
                });
            }
        }

        gemini_rust::Message { role, content }
    }

    /// Converts this `Message` to an Ollama-compatible message.
    #[cfg(feature = "ollama")]
    pub fn to_ollama_message(&self) -> ChatMessage {
        let images = match &self.content.images {
            Some(imgs) => imgs.iter().map(|img| img.to_ollama()).collect(),
            None => vec![],
        };

        ChatMessage {
            role: match self.content.role {
                Role::User => MessageRole::User,
                Role::System => MessageRole::System,
                Role::Tool => MessageRole::Tool,
                Role::Assistant => MessageRole::Assistant,
            },
            content: self.content.text.clone().unwrap_or_default(),
            tool_calls: self
                .function_calls
                .iter()
                .map(|fc| fc.to_ollama())
                .collect(),
            images: if images.is_empty() {
                None
            } else {
                Some(images)
            },
            thinking: self.thinking.clone(),
        }
    }

    /// Converts this `Message` to an OpenAI-compatible message.
    #[cfg(feature = "openai")]
    pub fn to_openai_message(&self) -> openai_api_rs::v1::chat_completion::ChatCompletionMessage {
        use openai_api_rs::v1::chat_completion::{
            Content, MessageRole, ToolCall, ToolCallFunction,
        };

        let role = match self.content.role {
            Role::User => MessageRole::user,
            Role::Assistant => MessageRole::assistant,
            Role::System => MessageRole::system,
            Role::Tool => MessageRole::tool,
        };

        // Convert function_calls to OpenAI tool_calls
        let tool_calls = if self.function_calls.is_empty() {
            None
        } else {
            Some(
                self.function_calls
                    .iter()
                    .enumerate()
                    .map(|(i, fc)| ToolCall {
                        id: format!("call_{}", i),
                        r#type: "function".to_string(),
                        function: ToolCallFunction {
                            name: Some(fc.name.clone()),
                            arguments: Some(fc.arguments.to_string()),
                        },
                    })
                    .collect(),
            )
        };

        // For tool role (function results), we need tool_call_id
        let tool_call_id = if self.content.role == Role::Tool && !self.function_results.is_empty() {
            // Use the function name as identifier (OpenAI expects the call ID)
            self.function_results.first().map(|fr| fr.name.clone())
        } else {
            None
        };

        // For tool results, content is the result JSON
        let content = if self.content.role == Role::Tool && !self.function_results.is_empty() {
            Content::Text(
                self.function_results
                    .first()
                    .map(|fr| fr.results.to_string())
                    .unwrap_or_default(),
            )
        } else {
            Content::Text(self.content.text.clone().unwrap_or_default())
        };

        openai_api_rs::v1::chat_completion::ChatCompletionMessage {
            role,
            content,
            name: None,
            tool_call_id,
            tool_calls,
        }
    }

    /// Converts a Gemini `GenerationResponse` to this `Message` format.
    #[cfg(feature = "google")]
    pub fn from_gemini_response(gemini_message: GenerationResponse) -> Inference {
        let candidate = match gemini_message.candidates.first() {
            Some(c) => c,
            None => {
                let message = format!("No candidates in response: {:?}", gemini_message);

                return Inference {
                    model: gemini_message.model_version,
                    content: InferenceContent {
                        role: Role::Assistant,
                        text: Some(message),
                        audio: None,
                        images: None,
                    },
                    ..Default::default()
                };
            }
        };

        let mut function_calls = vec![];
        let mut message_parts = vec![];
        let mut audio_parts = vec![];
        if let Some(parts) = &candidate.content.parts {
            for part in parts {
                if let Part::FunctionCall { function_call, .. } = part {
                    function_calls.push(FunctionCall::from_gemini(function_call.clone()));
                }

                if let Part::Text { text, .. } = part {
                    message_parts.push(text.clone());
                }

                if let Part::InlineData { inline_data } = part
                    && inline_data.mime_type.starts_with("audio/")
                {
                    let decoded = base64::engine::general_purpose::STANDARD
                        .decode(&inline_data.data)
                        .ok();
                    if let Some(audio_data) = decoded {
                        audio_parts.push(audio_data);
                    }
                }
            }
        }

        let finish_reason = candidate
            .finish_reason
            .as_ref()
            .map(FinishReason::from_google);

        Inference {
            model: gemini_message.model_version.clone(),
            content: InferenceContent {
                role: Role::Assistant,
                text: if message_parts.is_empty() {
                    None
                } else {
                    Some(message_parts.join("\n"))
                },
                audio: if audio_parts.is_empty() {
                    None
                } else {
                    // For simplicity, only take the first audio part
                    Some(audio_parts.into_iter().next().unwrap())
                },
                images: None,
            },
            thinking: gemini_message.thoughts().join("\n").into(),
            function_calls,
            finish_reason,
            ..Default::default()
        }
    }

    /// Converts an Ollama `ChatMessageResponse` to this `Message` format.
    #[cfg(feature = "ollama")]
    pub fn from_ollama_response(ollama_message: ChatMessageResponse) -> Inference {
        let images = ollama_message.message.images.as_ref().map(|imgs| {
            imgs.iter()
                .map(|img| Image::from_ollama(img.clone()))
                .collect()
        });

        let finish_reason = if !ollama_message.message.tool_calls.is_empty() {
            Some(FinishReason::ToolCall)
        } else {
            None
        };

        Inference {
            model: Some(ollama_message.model),
            content: InferenceContent {
                role: match ollama_message.message.role {
                    MessageRole::User => Role::User,
                    MessageRole::System => Role::System,
                    MessageRole::Tool => Role::Tool,
                    MessageRole::Assistant => Role::Assistant,
                },
                text: Some(ollama_message.message.content.clone()),
                audio: None,
                images,
            },
            thinking: ollama_message.message.thinking,
            function_calls: ollama_message
                .message
                .tool_calls
                .into_iter()
                .map(FunctionCall::from_ollama)
                .collect(),
            finish_reason,
            ..Default::default()
        }
    }

    /// Converts an OpenAI `ChatCompletionMessage` to this `Message` format.
    #[cfg(feature = "openai")]
    pub fn from_openai_choice(choice: &ChatCompletionChoice) -> Inference {
        let function_calls = choice
            .message
            .tool_calls
            .as_ref()
            .map(|tcs| {
                tcs.iter()
                    .map(|tc| FunctionCall::from_openai(tc.clone()))
                    .collect()
            })
            .unwrap_or_default();

        let finish_reason = choice.finish_reason.as_ref().map(FinishReason::from_openai);

        Inference {
            // i think name is not the model name
            model: choice.message.name.clone(),
            content: InferenceContent {
                role: match choice.message.role {
                    openai_api_rs::v1::chat_completion::MessageRole::user => Role::User,
                    openai_api_rs::v1::chat_completion::MessageRole::system => Role::System,
                    openai_api_rs::v1::chat_completion::MessageRole::tool => Role::Tool,
                    openai_api_rs::v1::chat_completion::MessageRole::assistant => Role::Assistant,
                    _ => Role::Assistant,
                },
                text: choice.message.content.clone(),
                audio: None,
                images: None,
            },
            thinking: choice.message.reasoning_content.clone(),
            function_calls,
            finish_reason,
            ..Default::default()
        }
    }

    #[cfg(feature = "openai")]
    pub fn from_openai_stream_response(
        message: ChatCompletionStreamResponse,
    ) -> crate::Result<Inference> {
        match message {
            ChatCompletionStreamResponse::ToolCall(toolcalls) => Ok(Inference {
                content: InferenceContent {
                    role: Role::Assistant,
                    text: None,
                    audio: None,
                    images: None,
                },
                function_calls: toolcalls
                    .iter()
                    .map(|tc| FunctionCall::from_openai(tc.clone()))
                    .collect(),
                finish_reason: Some(FinishReason::ToolCall),
                ..Default::default()
            }),
            ChatCompletionStreamResponse::Content(content) => Ok(Inference {
                content: InferenceContent {
                    role: Role::Assistant,
                    text: Some(content),
                    audio: None,
                    images: None,
                },
                finish_reason: Some(FinishReason::Stop),
                ..Default::default()
            }),
            ChatCompletionStreamResponse::Done => Ok(Inference {
                content: InferenceContent {
                    role: Role::Assistant,
                    text: None,
                    audio: None,
                    images: None,
                },
                finish_reason: Some(FinishReason::Stop),
                ..Default::default()
            }),
        }
    }
}

impl Display for InferenceContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self)
    }
}

impl Default for InferenceContent {
    fn default() -> Self {
        Self {
            role: Role::Assistant,
            text: None,
            audio: None,
            images: None,
        }
    }
}

impl Display for Inference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self)
    }
}

impl Default for Inference {
    fn default() -> Self {
        Self {
            model: None,
            content: InferenceContent::default(),
            thinking: None,
            function_calls: vec![],
            function_results: vec![],
            finish_reason: None,
            usage: None,
        }
    }
}

impl From<&str> for Inference {
    fn from(s: &str) -> Self {
        Inference::with_content(Role::User, s)
    }
}

impl From<(Role, &str)> for Inference {
    fn from((role, s): (Role, &str)) -> Self {
        Inference::with_content(role, s)
    }
}

impl From<InferenceContent> for Inference {
    fn from(content: InferenceContent) -> Self {
        Inference::new(content)
    }
}

impl From<&str> for InferenceContent {
    fn from(s: &str) -> Self {
        InferenceContent {
            role: Role::User,
            text: Some(s.to_string()),
            audio: None,
            images: None,
        }
    }
}

impl From<(Role, &str)> for InferenceContent {
    fn from((role, s): (Role, &str)) -> Self {
        InferenceContent {
            role,
            text: Some(s.to_string()),
            audio: None,
            images: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::function::FunctionCall;
    use serde_json::json;

    // --- Helpers ---

    fn mock_function_call() -> FunctionCall {
        FunctionCall {
            name: "get_weather".to_string(),
            arguments: json!({"city": "London"}),
        }
    }

    fn mock_function_result() -> FunctionResult {
        FunctionResult {
            name: "get_weather".to_string(),
            results: json!({"temp": 22}),
        }
    }

    // --- Basic Inference Construction Tests ---

    #[test]
    fn test_inference_helpers() {
        let user = Inference::as_user("hello");
        assert_eq!(user.content.role, Role::User);
        assert_eq!(user.content.text.unwrap(), "hello");

        let assistant = Inference::as_assistant("hi");
        assert_eq!(assistant.content.role, Role::Assistant);

        let system = Inference::as_system("act as bot");
        assert_eq!(system.content.role, Role::System);
    }

    #[test]
    fn test_inference_with_tools() {
        let inf = Inference::as_assistant("using tool").add_function_call(mock_function_call());

        assert!(inf.has_function_calls());
        assert_eq!(inf.function_calls[0].name, "get_weather");
    }

    #[test]
    fn test_inference_with_results() {
        let results = vec![mock_function_result()];
        let inf = Inference::with_function_results(results);

        assert_eq!(inf.content.role, Role::Tool);
        assert!(inf.has_function_results());
        assert_eq!(inf.function_results[0].name, "get_weather");
    }

    // --- Provider Transformation Tests ---

    #[cfg(feature = "openai")]
    #[test]
    fn test_to_openai_message_with_tools() {
        let inf = Inference::as_assistant("calling tool").add_function_call(mock_function_call());

        let msg = inf.to_openai_message();

        assert!(msg.tool_calls.is_some());
        let tools = msg.tool_calls.unwrap();
        assert_eq!(tools[0].function.name.as_ref().unwrap(), "get_weather");
        // OpenAI tool calls IDs are generated as call_0, call_1...
        assert_eq!(tools[0].id, "call_0");
    }

    #[cfg(feature = "openai")]
    #[test]
    fn test_to_openai_tool_result_mapping() {
        let result = mock_function_result();
        let inf = Inference::with_function_results(vec![result]);

        let msg = inf.to_openai_message();

        assert!(matches!(
            msg.role,
            openai_api_rs::v1::chat_completion::MessageRole::tool
        ));
        // Verify tool_call_id is derived from function name as per implementation
        assert_eq!(msg.tool_call_id.unwrap(), "get_weather");

        if let openai_api_rs::v1::chat_completion::Content::Text(t) = msg.content {
            assert!(t.contains("22"));
        } else {
            panic!("Expected text content for tool result");
        }
    }

    #[cfg(feature = "google")]
    #[test]
    fn test_to_gemini_message_multimodal() {
        let audio_data = vec![1, 2, 3, 4];
        let inf = Inference::as_user("listen to this")
            .with_audio(audio_data)
            .with_thinking("processing audio".to_string());

        let msg = inf.to_gemini_message();

        let parts = msg.content.parts.unwrap();
        // Check for Audio (InlineData) and Text
        assert!(parts.iter().any(|p| matches!(p, Part::InlineData { .. })));
        assert!(parts.iter().any(|p| matches!(p, Part::Text { .. })));

        // Check thinking flag in Gemini Text part
        if let Part::Text { thought, .. } = &parts[1] {
            assert_eq!(thought, &Some(true));
        }
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn test_ollama_roundtrip_logic() {
        let mut inf = Inference::as_user("see this");
        inf = inf.add_image(Image::new("base64_data".into(), None));
        inf = inf.with_thinking("thinking...".into());

        let msg = inf.to_ollama_message();
        assert_eq!(msg.images.unwrap().len(), 1);
        assert_eq!(msg.thinking.unwrap(), "thinking...");
    }

    // --- Response Parsing Tests ---

    #[cfg(feature = "google")]
    #[test]
    fn test_from_gemini_response_parsing() {
        use gemini_rust::{Candidate, Content};

        let response = GenerationResponse {
            candidates: vec![Candidate {
                content: Content {
                    role: Some(gemini_rust::Role::Model),
                    parts: Some(vec![
                        Part::Text {
                            text: "Hello".to_string(),
                            thought: None,
                            thought_signature: None,
                        },
                        Part::FunctionCall {
                            function_call: gemini_rust::FunctionCall {
                                name: "test_fn".into(),
                                args: json!({}),
                                thought_signature: None,
                            },
                            thought_signature: None,
                        },
                    ]),
                },
                finish_reason: Some(gemini_rust::FinishReason::Stop),
                citation_metadata: None,
                index: None,
                safety_ratings: None,
            }],
            model_version: Some("gemini-1.5".into()),
            response_id: None,
            prompt_feedback: None,
            usage_metadata: None,
        };

        let inf = Inference::from_gemini_response(response);
        assert_eq!(inf.content.text.unwrap(), "Hello");
        assert_eq!(inf.function_calls[0].name, "test_fn");
        assert_eq!(inf.model.unwrap(), "gemini-1.5");
    }

    #[cfg(feature = "openai")]
    #[test]
    fn test_from_openai_stream_responses() {
        // Test Content chunk
        let chunk = ChatCompletionStreamResponse::Content("streaming text".into());
        let inf = Inference::from_openai_stream_response(chunk).unwrap();
        assert_eq!(inf.content.text.unwrap(), "streaming text");
        assert!(matches!(inf.finish_reason, Some(FinishReason::Stop)));

        // Test Done chunk
        let done = ChatCompletionStreamResponse::Done;
        let inf_done = Inference::from_openai_stream_response(done).unwrap();
        assert!(inf_done.content.text.is_none());
        assert!(matches!(inf_done.finish_reason, Some(FinishReason::Stop)));
    }

    // --- Traits and Conversions Tests ---

    #[test]
    fn test_inference_from_string_conversions() {
        let inf: Inference = "simple message".into();
        assert_eq!(inf.content.role, Role::User);
        assert_eq!(inf.content.text.unwrap(), "simple message");

        let tuple_inf: Inference = (Role::System, "init").into();
        assert_eq!(tuple_inf.content.role, Role::System);
    }
}
