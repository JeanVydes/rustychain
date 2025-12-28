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

    #[cfg(feature = "openrouter")]
    pub fn from_openrouter(reason: &openrouter_rs::types::completion::FinishReason) -> Self {
        match reason {
            openrouter_rs::types::completion::FinishReason::ToolCalls => {
                FinishReason::UnexpectedToolCall
            }
            openrouter_rs::types::completion::FinishReason::Stop => FinishReason::Stop,
            openrouter_rs::types::completion::FinishReason::Length => FinishReason::MaxTokens,
            openrouter_rs::types::completion::FinishReason::ContentFilter => FinishReason::Safety,
            openrouter_rs::types::completion::FinishReason::Error => {
                FinishReason::Other("Error".to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finish_reason_serialization() {
        let reason = FinishReason::MaxTokens;
        let serialized = serde_json::to_string(&reason).unwrap();
        assert_eq!(serialized, "\"MaxTokens\"");

        let other = FinishReason::Other("CustomReason".to_string());
        let serialized_other = serde_json::to_string(&other).unwrap();
        assert!(serialized_other.contains("Other"));
        assert!(serialized_other.contains("CustomReason"));
    }

    #[test]
    fn test_finish_reason_deserialization() {
        let json = "\"Safety\"";
        let deserialized: FinishReason = serde_json::from_str(json).unwrap();
        assert!(matches!(deserialized, FinishReason::Safety));

        let complex_json = "{\"Other\":\"NetworkError\"}";
        let deserialized_other: FinishReason = serde_json::from_str(complex_json).unwrap();
        if let FinishReason::Other(msg) = deserialized_other {
            assert_eq!(msg, "NetworkError");
        } else {
            panic!("Expected FinishReason::Other");
        }
    }

    #[cfg(feature = "google")]
    #[test]
    fn test_from_google_mapping() {
        use gemini_rust::FinishReason as GReason;

        let pairs = vec![
            (GReason::Stop, FinishReason::Stop),
            (GReason::MaxTokens, FinishReason::MaxTokens),
            (GReason::Safety, FinishReason::Safety),
            (GReason::Recitation, FinishReason::Recitation),
            (GReason::ImageSafety, FinishReason::ImageSafety),
            (GReason::FinishReasonUnspecified, FinishReason::Unspecified),
        ];

        for (google, internal) in pairs {
            let mapped = FinishReason::from_google(&google);
            assert_eq!(
                serde_json::to_value(&mapped).unwrap(),
                serde_json::to_value(&internal).unwrap()
            );
        }

        // Test specific "Other" string mapping
        let language_mapped = FinishReason::from_google(&GReason::Language);
        if let FinishReason::Other(s) = language_mapped {
            assert_eq!(s, "Language");
        } else {
            panic!("Expected Other('Language')");
        }
    }

    #[cfg(feature = "openai")]
    #[test]
    fn test_from_openai_mapping() {
        use openai_api_rs::v1::chat_completion::FinishReason as OReason;

        let stop_mapped = FinishReason::from_openai(&OReason::stop);
        assert!(matches!(stop_mapped, FinishReason::Stop));

        let length_mapped = FinishReason::from_openai(&OReason::length);
        assert!(matches!(length_mapped, FinishReason::MaxTokens));

        let filter_mapped = FinishReason::from_openai(&OReason::content_filter);
        assert!(matches!(filter_mapped, FinishReason::Safety));

        let tools_mapped = FinishReason::from_openai(&OReason::tool_calls);
        assert!(matches!(tools_mapped, FinishReason::UnexpectedToolCall));

        let null_mapped = FinishReason::from_openai(&OReason::null);
        if let FinishReason::Other(s) = null_mapped {
            assert_eq!(s, "Null");
        } else {
            panic!("Expected Other('Null')");
        }
    }
}
