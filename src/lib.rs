///! RustyChain: A modular framework for building LLM-powered applications in Rust.
///! This crate provides core functionalities, including LLM integration, tool management,
///! document splitting, and vector storage.

#[cfg(feature = "audio")]
pub mod audio;
pub mod error;
pub mod llm;
pub mod orchestor;
pub mod splitters;
pub mod storage;
#[cfg(feature = "tools")]
pub mod tools;
pub use error::*;
pub use llm::*;
pub use splitters::*;
pub use storage::*;

pub mod prelude {
    pub use schemars::JsonSchema;
    pub use crate::error::CoreError;
    pub use crate::function::{AnyFunction, FunctionDeclaration, FnDeclarator, FnExecutor, FunctionResult};
    pub use crate::llm::{LLM, LLMProvider, LLMActions};
    #[cfg(feature = "tools")]
    pub use crate::ToolArgs;
}