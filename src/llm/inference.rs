use futures_core::stream::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;

use crate::{
    prelude::*,
    llm::{GenerationConfig, LLM, conversation::Message}
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum InferenceMode {
    OneShot,
    Streaming,
}

#[derive(Clone, Debug)]
pub struct Inference {
    pub history: Vec<Message>,
    pub message: Message,
    pub output_schema: Option<Value>,
    pub config: GenerationConfig,
    pub llm: Arc<LLM>,
}

impl Inference {
    pub fn new(message: Message, llm: Arc<LLM>) -> Self {
        Self {
            history: vec![],
            message,
            output_schema: None,
            config: GenerationConfig::default(),
            llm,
        }
    }

    pub fn with_history(mut self, history: Vec<Message>) -> Self {
        self.history = history;
        self
    }

    pub fn with_output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    pub fn with_config(mut self, config: GenerationConfig) -> Self {
        self.config = config;
        self
    }

    pub async fn generate(&mut self) -> crate::Result<Message> {
        return self
            .llm
            .generation(&mut self.history, self.message.clone(), self.config.clone())
            .await;
    }

    pub async fn stream(
        &mut self,
    ) -> crate::Result<Pin<Box<dyn Stream<Item = crate::Result<Message>> + Send + 'static>>> {
        return self
            .llm
            .stream(&mut self.history, self.message.clone(), self.config.clone())
            .await;
    }
}
