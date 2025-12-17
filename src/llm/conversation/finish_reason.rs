use serde::{Deserialize, Serialize};

/// Reason for finishing the generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,
    MaxTokens,
    Safety,
    Recitation,
    Blocklist,
    ProhibitedContent,
    Spii,
    MalformedFunctionCall,
    ImageSafety,
    UnexpectedToolCall,
    TooManyToolCalls,
    ToolCall,
    Other(String),
    Unspecified,
}

impl FinishReason {
    #[cfg(feature = "google")]
    pub fn from_google(reason: &gemini_rust::FinishReason) -> Self {
        match reason {
            gemini_rust::FinishReason::Stop => FinishReason::Stop,
            gemini_rust::FinishReason::MaxTokens => FinishReason::MaxTokens,
            gemini_rust::FinishReason::Safety => FinishReason::Safety,
            gemini_rust::FinishReason::Recitation => FinishReason::Recitation,
            gemini_rust::FinishReason::Blocklist => FinishReason::Blocklist,
            gemini_rust::FinishReason::ProhibitedContent => FinishReason::ProhibitedContent,
            gemini_rust::FinishReason::Spii => FinishReason::Spii,
            gemini_rust::FinishReason::MalformedFunctionCall => FinishReason::MalformedFunctionCall,
            gemini_rust::FinishReason::UnexpectedToolCall => FinishReason::UnexpectedToolCall,
            gemini_rust::FinishReason::TooManyToolCalls => FinishReason::TooManyToolCalls,
            gemini_rust::FinishReason::Language => FinishReason::Other("Language".to_string()),
            gemini_rust::FinishReason::ImageSafety => FinishReason::ImageSafety,
            gemini_rust::FinishReason::Other => FinishReason::Other("Other".to_string()),
            gemini_rust::FinishReason::FinishReasonUnspecified => FinishReason::Unspecified,
        }
    }

    #[cfg(feature = "openai")]
    pub fn from_openai(reason: &openai_api_rs::v1::chat_completion::FinishReason) -> Self {
        match reason {
            openai_api_rs::v1::chat_completion::FinishReason::stop => FinishReason::Stop,
            openai_api_rs::v1::chat_completion::FinishReason::length => FinishReason::MaxTokens,
            openai_api_rs::v1::chat_completion::FinishReason::content_filter => {
                FinishReason::Safety
            }
            openai_api_rs::v1::chat_completion::FinishReason::tool_calls => {
                FinishReason::UnexpectedToolCall
            }
            openai_api_rs::v1::chat_completion::FinishReason::null => {
                FinishReason::Other("Null".to_string())
            }
        }
    }
}
