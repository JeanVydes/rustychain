//! Error
//!
//! Error handling

#[cfg(feature = "google")]
pub use gemini_rust::ClientError;
#[cfg(feature = "ollama")]
pub use ollama_rs::error::OllamaError;
use std::{any::Any, error::Error as StdError, sync::Arc};
use thiserror::Error;

/// A specialized Result type for the RustyChain core module.
pub type Result<T> = std::result::Result<T, Box<dyn StdError + Send + Sync>>;

/// Core errors for RustyChain
/// These errors cover various failure scenarios in the core functionality,
/// including interactions with LLM providers, serialization issues, and chain execution errors.
#[derive(Debug, Error)]
pub enum CoreError {
    /// Generic error with a message
    #[error("{0}")]
    Generic(String),

    // LLM PROVIDER ERRORS
    /// Gemini provider errors
    #[cfg(feature = "google")]
    #[error("Gemini API Error: {0}")]
    Gemini(#[from] ClientError),

    /// Ollama provider errors
    #[cfg(feature = "ollama")]
    #[error("Ollama Error: {0}")]
    Ollama(#[from] OllamaError),

    /// OpenAI provider errors
    #[error("OpenAI Error: {0}")]
    OpenAI(String),

    // OTHER ERRORS
    /// Serialization errors
    #[error("Serialization Error: {0}")]
    Serde(#[from] serde_json::Error),

    /// HTTP errors
    #[error("HTTP Error: {0}")]
    HTTP(Box<dyn StdError + Send + Sync>),

    #[cfg(feature = "tools")]
    #[error("Reqwest Error: {0}")]
    Reqwest(#[from] reqwest::Error),

    #[error("URL Parse Error: {0}")]
    URL(#[from] url::ParseError),

    #[error("HTML Parsing Error: {0}")]
    ParsingHTML(String),

    #[error("Scraper Selector Error: {0}")]
    Scraper(String),

    /// Internal errors
    #[error("Internal Error: {0}")]
    Internal(Box<dyn StdError + Send + Sync>),

    /// Not found errors
    #[error("Not Found: {0}")]
    NotFound(String),

    #[error("No Content")]
    NoContent,

    /// SQLx database errors
    #[error("Database Error: {0}")]
    Sqlx(#[from] sqlx::Error),

    /// IO errors
    #[error("IO Error: {0}")]
    IO(#[from] std::io::Error),

    /// Unsupported operation errors
    #[error("Unsupported: {0}")]
    Unsupported(String),

    /// Downcast errors
    #[error("Downcast Error")]
    Downcast(Arc<dyn Any + Send + Sync>),

    /// Rate limit errors
    #[error("Rate limit exceeded. Retry after {retry_after_secs} seconds")]
    RateLimit { retry_after_secs: u64 },

    // CHAIN ERRORS
    /// Errors occurring during chain step execution
    #[error("Error in step {index} ('{step_name:?}'): {source}")]
    StepError {
        index: usize,
        step_name: Option<String>,
        source: Box<dyn StdError + Send + Sync>,
    },

    /// Initial input to the chain was not set
    #[error("Chain initial input not set")]
    NotInput,

    /// Chain has not been finalized yet
    #[error("Chain has not been finalized yet")]
    ChainNotFinalized,
}

/// CoreError utility methods
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
