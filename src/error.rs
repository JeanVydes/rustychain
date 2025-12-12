#[cfg(feature = "google")]
pub use gemini_rust::ClientError;
#[cfg(feature = "ollama")]
pub use ollama_rs::error::OllamaError;
use std::error::Error as StdError;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Box<dyn StdError + Send + Sync>>;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("{0}")]
    Generic(String),

    #[cfg(feature = "google")]
    #[error("Gemini API Error: {0}")]
    Gemini(#[from] ClientError),

    #[cfg(feature = "ollama")]
    #[error("Ollama Error: {0}")]
    Ollama(#[from] OllamaError),

    #[error("OpenAI Error: {0}")]
    OpenAI(String),

    #[error("Serialization Error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("HTTP Error: {0}")]
    HTTP(Box<dyn StdError + Send + Sync>),

    #[error("Internal Error: {0}")]
    Internal(Box<dyn StdError + Send + Sync>),

    #[error("Not Found: {0}")]
    NotFound(String),

    #[error("Database Error: {0}")]
    Sqlx(#[from] sqlx::Error),

    #[error("IO Error: {0}")]
    IO(#[from] std::io::Error),

    #[error("Unsupported: {0}")]
    Unsupported(String),

    #[error("Rate limit exceeded. Retry after {retry_after_secs} seconds")]
    RateLimit { retry_after_secs: u64 },
}

impl CoreError {
    /// Check if an error is a rate limit error (HTTP 429)
    pub fn is_rate_limit(error: &(dyn StdError + Send + Sync)) -> bool {
        let error_str = error.to_string();
        error_str.contains("429")
            || error_str.contains("RESOURCE_EXHAUSTED")
            || error_str.contains("rate limit")
    }

    /// Extract retry delay from error message if present
    pub fn extract_retry_delay(error: &(dyn StdError + Send + Sync)) -> Option<u64> {
        let error_str = error.to_string();

        // Try to find "retry in X.Xs" pattern
        if let Some(pos) = error_str.find("retry in ") {
            let after = &error_str[pos + 9..];
            if let Some(end) = after.find('s') {
                let num_str = &after[..end];
                if let Ok(secs) = num_str.parse::<f64>() {
                    return Some(secs.ceil() as u64);
                }
            }
        }

        // Try to find "retryDelay": "Xs" pattern
        if let Some(pos) = error_str.find("\"retryDelay\":") {
            let after = &error_str[pos + 13..];
            if let Some(start) = after.find('"') {
                let after_quote = &after[start + 1..];
                if let Some(end) = after_quote.find('s') {
                    let num_str = &after_quote[..end];
                    if let Ok(secs) = num_str.parse::<u64>() {
                        return Some(secs);
                    }
                }
            }
        }

        None
    }
}
