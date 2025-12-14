use std::sync::Arc;

use crate::LLM;

#[derive(Debug, Clone)]
pub struct Agent {
    pub name: String,
    pub description: String,
    pub llm: Arc<LLM>,
}
