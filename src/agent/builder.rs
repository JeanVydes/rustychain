use crate::{LLM, Message, agent::definitions::Agent};
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct AgentBuilder {
    pub name: Option<String>,
    pub description: Option<String>,
    pub llm: Arc<LLM>,
    pub history: Option<Arc<Mutex<Vec<Message>>>>,
    pub max_depth: i32,
}

impl AgentBuilder {
    pub fn new(llm: Arc<LLM>) -> Self {
        Self {
            name: None,
            description: None,
            llm,
            history: None,
            max_depth: 5,
        }
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn history(mut self, history: Arc<Mutex<Vec<Message>>>) -> Self {
        self.history = Some(history);
        self
    }
    pub fn max_depth(mut self, depth: i32) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn build(self) -> Agent {
        Agent {
            name: self.name,
            description: self.description,
            llm: self.llm,
            history: self.history.unwrap_or_else(|| Arc::new(Mutex::new(vec![]))),
            max_depth: self.max_depth,
        }
    }
}
