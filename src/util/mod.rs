//!! Utilities
//!
//! This module contains various utility functions and helpers

pub mod formatters;
pub mod http_agent;
pub mod url;
use std::sync::Arc;

pub use http_agent::*;
use serde_json::json;
pub use url::*;

use crate::AnyFunction;

pub fn tools_to_string(tools: &Vec<Arc<dyn AnyFunction>>) -> String {
    tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name(),
                "description": tool.description(),
                "parameters": tool.parameters_schema(),
            })
            .to_string()
        })
        .collect::<Vec<String>>()
        .join("\n")
}
