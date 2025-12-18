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
pub mod orchestrator;
pub mod splitters;
pub mod storage;
#[cfg(feature = "tools")]
pub mod tools;
pub mod util;
pub use error::*;
#[cfg(any(feature = "openai", feature = "google", feature = "ollama"))]
#[allow(ambiguous_glob_reexports)]
pub use llm::*;
pub use splitters::*;
pub use storage::*;

pub const BASE64_ENGINE: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

pub mod prelude {
    //! Prelude
    //!
    //! This module re-exports commonly used types and traits.

    #[cfg(feature = "tools")]
    pub use crate::ToolArgs;
    pub use crate::error::Error;
    #[cfg(any(feature = "openai", feature = "google", feature = "ollama"))]
    pub use crate::function::{
        AnyFunction, FnDeclarator, FnExecutor, FunctionDeclaration, FunctionResult,
    };
    #[cfg(any(feature = "openai", feature = "google", feature = "ollama"))]
    pub use crate::llm::{LLM, LLMEmbedding, LLMGeneration, LLMProvider, LLMStreaming};
    pub use crate::util::formatters::Cleaner;
    pub use crate::util::formatters::Formatter;
    #[cfg(feature = "macros")]
    pub use rustychain_macros::*;
    pub use schemars::JsonSchema;
}
