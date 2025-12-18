use futures_core::stream::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;

use crate::{
    llm::{GenerationConfig, LLM, conversation::Inference},
    prelude::*,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum InferenceMode {
    OneShot,
    Streaming,
}

#[derive(Clone, Debug)]
pub struct InferenceTask<'a> {
    pub history: &'a [Inference],
    pub inference: Inference,
    pub config: GenerationConfig,
    pub llm: Arc<LLM>,
}

impl<'a> InferenceTask<'a> {
    pub fn new(inference: Inference, llm: Arc<LLM>) -> Self {
        Self {
            history: &[],
            inference,
            config: GenerationConfig::default(),
            llm,
        }
    }

    pub fn with_history(mut self, history: &'a [Inference]) -> Self {
        self.history = history;
        self
    }

    pub fn with_config(mut self, config: GenerationConfig) -> Self {
        self.config = config;
        self
    }

    pub async fn generate(&mut self) -> crate::Result<Inference> {
        return self
            .llm
            .generation(self.history, &self.inference, self.config.clone())
            .await;
    }

    pub async fn stream(
        &mut self,
    ) -> crate::Result<Pin<Box<dyn Stream<Item = crate::Result<Inference>> + Send + 'static>>> {
        return self
            .llm
            .stream(self.history, &self.inference, self.config.clone())
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Role,
        llm::{GenerationConfig, LLMProvider},
    };
    use std::sync::Arc;

    // --- Helpers ---

    fn create_test_llm() -> Arc<LLM> {
        Arc::new(LLM {
            model: "test-model".into(),
            system_prompt: "test-prompt".into(),
            provider: LLMProvider::OpenAI,
            authorization: Some("test-key".into()),
            endpoint: None,
            tools: vec![],
        })
    }

    // --- Tests ---

    #[test]
    fn test_inference_task_initialization() {
        let llm = create_test_llm();
        let inference = Inference::as_user("hello");
        let task = InferenceTask::new(inference.clone(), llm.clone());

        assert_eq!(task.inference.content.text, Some("hello".into()));
        assert!(task.history.is_empty());
        assert!(Arc::ptr_eq(&task.llm, &llm));
    }

    #[test]
    fn test_inference_task_fluent_builders() {
        let llm = create_test_llm();
        let history = vec![Inference::as_assistant("previous context")];
        let config = GenerationConfig::default().with_temperature(0.5);

        let task = InferenceTask::new(Inference::as_user("new query"), llm)
            .with_history(&history)
            .with_config(config.clone());

        assert_eq!(task.history.len(), 1);
        assert_eq!(
            task.history[0].content.text,
            Some("previous context".into())
        );
        assert_eq!(task.config.temperature, 0.5);
    }

    #[test]
    fn test_inference_mode_serialization() {
        let mode_oneshot = InferenceMode::OneShot;
        let mode_streaming = InferenceMode::Streaming;

        assert_eq!(serde_json::to_string(&mode_oneshot).unwrap(), "\"oneshot\"");
        assert_eq!(
            serde_json::to_string(&mode_streaming).unwrap(),
            "\"streaming\""
        );
    }

    #[test]
    fn test_inference_task_lifetime_bounds() {
        let llm = create_test_llm();
        let history_data = vec![Inference::as_user("context")];

        let task =
            InferenceTask::new(Inference::as_assistant("res"), llm).with_history(&history_data);

        assert_eq!(task.history.len(), 1);
        assert_eq!(task.history[0].content.text, Some("context".into()));
    }

    #[tokio::test]
    async fn test_inference_task_generate_logic_flow() {
        let llm = create_test_llm();
        let mut task = InferenceTask::new(Inference::as_user("trigger"), llm);

        assert_eq!(task.config.temperature, 0.5); // Default
        task = task.with_config(GenerationConfig::default().with_temperature(1.0));
        assert_eq!(task.config.temperature, 1.0);
    }

    #[test]
    fn test_inference_task_composition() {
        let llm = create_test_llm();
        let inference = Inference::as_user("composing task");
        let task = InferenceTask::new(inference, llm);

        assert!(matches!(task.inference.content.role, Role::User));
    }
}
