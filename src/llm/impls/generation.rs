use crate::{CoreError, GenerationConfig, LLM, LLMGeneration, LLMProvider, Message};

#[async_trait::async_trait]
impl LLMGeneration for LLM {
    /// Performs text generation based on the provided history and message.
    #[allow(unreachable_patterns)]
    async fn generation(
        &self,
        history: &mut Vec<Message>,
        message: Message,
        config: GenerationConfig,
    ) -> crate::Result<Message> {
        match self.provider {
            #[cfg(feature = "google")]
            LLMProvider::Google => {
                use crate::CoreError;

                let req = self.new_google_request(history, message, config)?;
                match req.execute().await {
                    Ok(res) => Ok(Message::from_gemini(res)),
                    Err(err) => Err(Box::from(CoreError::Gemini(err))),
                }
            }
            #[cfg(feature = "openai")]
            LLMProvider::OpenAI => {
                use std::error::Error;

                let mut client = self.get_openai_client()?;
                let req = self.new_openai_request(history, message, config)?;
                let res = client.chat_completion(req).await.map_err(
                    |e| -> Box<dyn Error + Send + Sync> {
                        use crate::CoreError;

                        Box::from(CoreError::OpenAI(format!("OpenAI request failed: {:?}", e)))
                    },
                )?;

                if let Some(choice) = res.choices.first() {
                    Ok(Message::from_openai_chat_completion_message_for_response(
                        &choice.message,
                    )?)
                } else {
                    Err(Box::from(CoreError::Generic(
                        "No choices returned from OpenAI".to_owned(),
                    )))
                }
            }
            #[cfg(feature = "ollama")]
            LLMProvider::Ollama => {
                use ollama_rs::Ollama;

                let req = self.new_ollama_request(history, message, config)?;
                let res = Ollama::default().send_chat_messages(req).await?;
                Ok(Message::from_ollama(res))
            }
            #[cfg(feature = "anthropic")]
            LLMProvider::Anthropic => Err(Box::from(CoreError::Generic(
                "Anthropic provider not yet implemented".to_owned(),
            ))),
            _ => Err(Box::from(CoreError::Unsupported(
                "Provider not supported for generation".to_owned(),
            ))),
        }
    }
}
