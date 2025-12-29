#[cfg(feature = "google")]
pub mod google;
pub mod openai;

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{FinishReason, GenerationConfig, Inference, Role};

pub const DEFAULT_OPENAI_API_BASE: &str = "https://api.openai.com/v1";
pub const DEFAULT_GOOGLE_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";
pub const DEFAULT_ANTHROPIC_API_BASE: &str = "https://api.anthropic.com/v1";
pub const DEFAULT_BEDROCK_API_BASE: &str = "https://bedrock-runtime.us-east-1.amazonaws.com";
pub const DEFAULT_OPENROUTER_API_BASE: &str = "https://openrouter.ai/api/v1";
pub const DEFAULT_OLLAMA_API_BASE: &str = "http://localhost:11434/api";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum LLMProvider {
    Google,

    // Supported with OpenAI Compatible API
    OpenAI,
    Ollama,
    OpenRouter,

    // Not Supported Yet
    Anthropic,
    Bedrock,
}

impl LLMProvider {
    pub fn as_str(&self) -> &str {
        match self {
            LLMProvider::Google => "google",
            LLMProvider::OpenAI => "openai",
            LLMProvider::Anthropic => "anthropic",
            LLMProvider::Ollama => "ollama",
            LLMProvider::Bedrock => "bedrock",
            LLMProvider::OpenRouter => "openrouter",
        }
    }

    pub fn default_api_base(&self) -> &str {
        match self {
            LLMProvider::Google => DEFAULT_GOOGLE_API_BASE,
            LLMProvider::OpenAI => DEFAULT_OPENAI_API_BASE,
            LLMProvider::Anthropic => DEFAULT_ANTHROPIC_API_BASE,
            LLMProvider::Ollama => DEFAULT_OLLAMA_API_BASE,
            LLMProvider::Bedrock => DEFAULT_BEDROCK_API_BASE,
            LLMProvider::OpenRouter => DEFAULT_OPENROUTER_API_BASE,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub api_key: Option<String>,
    pub api_base: String,
}

pub trait ProviderAbstractionLayer<ROL, MSG, REQ, FIN, IMG, RES, THK, TOOL, FNCALL, FNMODE, CLIENT>
where
    ROL: Send + Sync,
    MSG: Send + Sync,
    RES: Send + Sync,
    REQ: Send + Sync,
    FIN: Send + Sync,
    IMG: Send + Sync,
    THK: Send + Sync,
    TOOL: Send + Sync,
    FNCALL: Send + Sync,
    FNMODE: Send + Sync,
    CLIENT: Send + Sync,
{
    fn client(&self) -> crate::Result<CLIENT>;

    fn to_native_role(role: &ROL) -> Role;
    fn to_provider_role(role: &Role) -> ROL;

    fn to_inference_from_response(
        response: &RES,
        config: Option<Arc<GenerationConfig>>,
    ) -> Inference;
    fn to_provider_message(message: &Inference) -> MSG;

    fn to_native_finish_reason(finish: &FIN) -> FinishReason;
    fn to_provider_finish_reason(finish: &FinishReason) -> FIN;

    fn to_native_image(image: &IMG) -> crate::Result<crate::Image>;
    fn to_provider_image(image: &crate::Image) -> crate::Result<IMG>;

    fn to_provider_thinking_mode(
        mode: &crate::llm::thinking_mode::ThinkingMode,
    ) -> crate::Result<THK>;

    fn to_native_function_call(call: &FNCALL) -> crate::Result<crate::FunctionCall>;
    fn to_provider_function_call(call: &crate::FunctionCall) -> crate::Result<FNCALL>;

    fn to_provider_tool_calling_mode(
        mode: &crate::llm::tool_calling_mode::ToolCallingMode,
    ) -> FNMODE;

    fn new_chat_request(&self, req: ChatCompletionRequest) -> crate::Result<REQ>;
}

#[derive(Debug, Clone)]
pub struct ChatCompletionRequest<'a> {
    pub model: &'a str,
    pub messages: &'a [Inference],
    pub inference: &'a Inference,
    pub config: &'a GenerationConfig,
    pub system_prompt: &'a str,
    pub tools: &'a [Arc<dyn crate::AnyFunction>],
}
