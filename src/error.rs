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
pub type Result<T> = std::result::Result<T, Error>;

/// Core errors for RustyChain
/// These errors cover various failure scenarios in the core functionality,
/// including interactions with LLM providers, serialization issues, and chain execution errors.
#[derive(Debug, Error)]
pub enum Error {
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

    #[cfg(feature = "openai")]
    #[error("OpenAI Error: {0}")]
    OpenAI(#[from] openai_api_rs::v1::error::APIError),

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

    #[error("Operation timed out")]
    Timeout,

    #[error("Maximum depth reached")]
    MaxDepthReached,

    /// SQLx database errors
    #[error("Database Error: {0}")]
    Sqlx(#[from] sqlx::Error),

    /// IO errors
    #[error("IO Error: {0}")]
    IO(Box<dyn StdError + Send + Sync>),

    #[error("Invalid Input: {0}")]
    Input(String),

    /// Unsupported operation errors
    #[error("Unsupported: {0}")]
    Unsupported(String),

    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    /// Downcast errors
    #[error("Downcast Error")]
    Downcast(Arc<dyn Any + Send + Sync>),

    #[error("Tokio Join Error: {0}")]
    TokioJoin(#[from] tokio::task::JoinError),

    #[error("Tool Error: {source}")]
    ToolError {
        #[source]
        source: Box<dyn StdError + Send + Sync>,
    },

    #[cfg(feature = "tools")]
    #[error("Argon2 Error: {source}")]
    Argon2 {
        source: Box<dyn StdError + Send + Sync>,
    },

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

    /// Initial input was not set
    #[error("Initial input not set")]
    NotInput,

    #[error("Chain input validation failed: {message}")]
    Validation { message: String },

    /// Chain has not been finalized yet
    #[error("Chain has not been finalized yet")]
    ChainNotFinalized,

    #[error("Search Error for query '{query:?}': {source:?}")]
    Search {
        query: Option<String>,
        source: Option<Box<dyn StdError + Send + Sync>>,
        feedback: Option<String>,
    },

    #[error("Tool functions results not returned results")]
    NotFunctionResults,
}

/// utility methods
impl Error {
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

    pub fn from_boxed(err: Box<dyn StdError + Send + Sync + 'static>) -> Self {
        match err.downcast::<Error>() {
            Ok(inner) => *inner,
            Err(err) => Error::Internal(err),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::IO(Box::new(err))
    }
}

impl From<base64::DecodeError> for Error {
    fn from(err: base64::DecodeError) -> Self {
        Error::Generic(format!("Base64 Decode Error: {}", err))
    }
}

#[cfg(feature = "tools")]
impl<'a> From<scraper::error::SelectorErrorKind<'a>> for Error {
    fn from(err: scraper::error::SelectorErrorKind<'a>) -> Self {
        Error::Scraper(format!("Selector Error: {}", err))
    }
}

#[cfg(feature = "tools")]
impl From<argon2::password_hash::Error> for Error {
    fn from(err: argon2::password_hash::Error) -> Self {
        Error::Argon2 {
            source: Box::from(err.to_string()),
        }
    }
}
