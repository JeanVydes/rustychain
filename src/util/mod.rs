//!! Utilities
//!
//! This module contains various utility functions and helpers

pub mod formatters;
pub mod http_agent;
pub mod url;

pub use http_agent::*;
pub use url::*;

#[cfg(any(feature = "openai", feature = "google", feature = "ollama"))]
pub fn tools_to_string(tools: &[std::sync::Arc<impl crate::AnyFunction>]) -> String {
    tools
        .iter()
        .map(|tool| {
            serde_json::json!({
                "name": tool.name(),
                "description": tool.description(),
                "parameters": tool.parameters_schema(),
            })
            .to_string()
        })
        .collect::<Vec<String>>()
        .join("\n")
}
