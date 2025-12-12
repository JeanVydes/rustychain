use crate::llm::llm::LLMActions;
use futures_core::stream::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;

use crate::{
    CoreError,
    llm::{GenerationConfig, LLM, conversation::Message},
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
    pub message: Option<Message>,
    pub audio: Option<Vec<u8>>,
    pub output_schema: Option<Value>,
    pub config: GenerationConfig,
    pub llm: Option<Arc<LLM>>,
}

impl Inference {
    pub fn new() -> Self {
        Self {
            history: vec![],
            message: None,
            audio: None,
            output_schema: None,
            config: GenerationConfig::default(),
            llm: None,
        }
    }

    pub fn with_message(mut self, message: Message) -> Self {
        self.message = Some(message);
        self
    }

    pub fn with_audio(mut self, audio: Vec<u8>) -> Self {
        self.audio = Some(audio);
        self
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

    pub fn with_llm(mut self, llm: Arc<LLM>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub async fn generate(&mut self) -> crate::Result<Message> {
        match &self.llm {
            Some(llm) => {
                if let Some(audio) = &self.audio {
                    return llm
                        .generation_with_audio_input(
                            &mut self.history,
                            audio.clone(),
                            self.config.clone(),
                        )
                        .await;
                } else if let Some(message) = &self.message {
                    return Ok(llm
                        .generation(&mut self.history, message.clone(), self.config.clone())
                        .await?);
                }

                Err(Box::from(CoreError::Generic(
                    "No message or audio provided".to_owned(),
                )))
            }
            None => Err(Box::from(CoreError::Generic("No LLM provided".to_owned()))),
        }
    }

    pub async fn stream(
        &mut self,
    ) -> crate::Result<Pin<Box<dyn Stream<Item = crate::Result<Message>> + Send + 'static>>> {
        match &self.llm {
            Some(llm) => {
                if let Some(audio) = &self.audio {
                    return llm
                        .stream_with_audio_input(
                            &mut self.history,
                            audio.clone(),
                            self.config.clone(),
                        )
                        .await;
                } else if let Some(message) = &self.message {
                    return llm
                        .stream(&mut self.history, message.clone(), self.config.clone())
                        .await;
                }

                Err(Box::from(CoreError::Generic(
                    "No message or audio provided".to_owned(),
                )))
            }
            None => Err(Box::from(CoreError::Generic("No LLM provided".to_owned()))),
        }
    }
}
