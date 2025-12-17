use crate::{Inference, LLM, agent::definitions::Agent};
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct AgentBuilder {
    pub name: Option<String>,
    pub description: Option<String>,
    pub llm: Arc<LLM>,
    pub history: Option<Arc<Mutex<Vec<Inference>>>>,
    pub max_depth: i32,
    pub generation_config: Option<crate::llm::GenerationConfig>,
    pub output_schema: Option<schemars::Schema>,
    pub initial: Option<Inference>,
}

impl AgentBuilder {
    pub fn new(llm: Arc<LLM>) -> Self {
        Self {
            name: None,
            description: None,
            llm,
            history: None,
            max_depth: 5,
            initial: None,
            generation_config: None,
            output_schema: None,
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

    pub fn history(mut self, history: Arc<Mutex<Vec<Inference>>>) -> Self {
        self.history = Some(history);
        self
    }
    pub fn max_depth(mut self, depth: i32) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn initial(mut self, inference: Inference) -> Self {
        self.initial = Some(inference);
        self
    }

    pub fn generation_config(mut self, config: crate::llm::GenerationConfig) -> Self {
        self.generation_config = Some(config);
        self
    }

    pub fn build(self) -> Agent {
        Agent {
            name: self.name,
            description: self.description,
            llm: self.llm,
            history: self.history.unwrap_or_else(|| Arc::new(Mutex::new(vec![]))),
            max_depth: self.max_depth,
            current: self.initial,
            generation_config: self.generation_config,
        }
    }
}
