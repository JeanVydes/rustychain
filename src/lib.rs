//! RustyChain: A modular framework for building LLM-powered applications in Rust.
//! This crate provides core functionalities, including LLM integration, tool management,
//! document splitting, and vector storage.

#[cfg(any(feature = "openai", feature = "google", feature = "ollama"))]
pub mod agent;
#[cfg(target_os = "linux")]
#[cfg(feature = "audio")]
pub mod audio;
pub mod chain;
pub mod error;
#[cfg(any(feature = "openai", feature = "google", feature = "ollama"))]
pub mod llm;
pub mod orchestor;
pub mod splitters;
pub mod storage;
pub mod util;
#[cfg(feature = "tools")]
pub mod tools;
pub use error::*;
#[cfg(any(feature = "openai", feature = "google", feature = "ollama"))]
#[allow(ambiguous_glob_reexports)]
pub use llm::*;
pub use splitters::*;
pub use storage::*;

pub const BASE64_ENGINE: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

pub mod prelude {
    #[cfg(feature = "tools")]
    pub use crate::ToolArgs;
    pub use crate::error::CoreError;
    #[cfg(any(feature = "openai", feature = "google", feature = "ollama"))]
    pub use crate::function::{
        AnyFunction, FnDeclarator, FnExecutor, FunctionDeclaration, FunctionResult,
    };
    #[cfg(any(feature = "openai", feature = "google", feature = "ollama"))]
    pub use crate::llm::{LLM, LLMActions, LLMProvider};
    #[cfg(feature = "macros")]
    pub use rustychain_macros::*;
    pub use schemars::JsonSchema;
}
