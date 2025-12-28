use serde::{Deserialize, Serialize};

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

    #[cfg(feature = "openrouter")]
    pub fn to_openrouter(&self) -> Option<openrouter_rs::types::ReasoningConfig> {
        match self {
            ThinkingMode::None => Some(openrouter_rs::types::ReasoningConfig {
                enabled: Some(false),
                effort: None,
                exclude: None,
                max_tokens: None,
            }),
            ThinkingMode::Dynamic => Some(openrouter_rs::types::ReasoningConfig {
                enabled: Some(true),
                effort: Some(openrouter_rs::types::Effort::Medium),
                exclude: None,
                max_tokens: None,
            }),
            ThinkingMode::Effort(effort) => {
                let reasoning_effort = match effort {
                    ThinkingEffort::Low => openrouter_rs::types::Effort::Low,
                    ThinkingEffort::Medium => openrouter_rs::types::Effort::Medium,
                    ThinkingEffort::High => openrouter_rs::types::Effort::High,
                };

                Some(openrouter_rs::types::ReasoningConfig {
                    enabled: Some(true),
                    effort: Some(reasoning_effort),
                    exclude: None,
                    max_tokens: None,
                })
            }
            ThinkingMode::Sized(size) => Some(openrouter_rs::types::ReasoningConfig {
                enabled: Some(true),
                max_tokens: Some(*size as u32),
                effort: None,
                exclude: None,
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash, PartialOrd, Ord)]
pub enum ThinkingEffort {
    Low,
    Medium,
    High,
}
