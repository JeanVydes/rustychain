use crate::{GenerationConfig, LLM, LLMGeneration, LLMProvider, conversation::Inference};

#[async_trait::async_trait]
impl LLMGeneration for LLM {
    /// Performs text generation based on the provided history and message.
    #[allow(unreachable_patterns)]
    async fn generation(
        &self,
        history: &[Inference],
        inference: &Inference,
        config: GenerationConfig,
    ) -> crate::Result<Inference> {
        match self.provider {
            #[cfg(feature = "google")]
            LLMProvider::Google => {
                let req = self.new_google_request(history, inference, config)?;
                match req.execute().await {
                    Ok(res) => Ok(Inference::from_gemini_response(res)),
                    Err(err) => Err(crate::Error::from(err)),
                }
            }

            #[cfg(feature = "openai")]
            LLMProvider::OpenAI => {
                let mut client = self.get_openai_client()?;
                let req = self.new_openai_request(history, inference, config)?;
                let res = client.chat_completion(req).await?;

                if let Some(choice) = res.choices.first() {
                    Ok(Inference::from_openai_choice(choice))
                } else {
                    Err(crate::Error::Generic(
                        "No choices returned from OpenAI".to_owned(),
                    ))
                }
            }
            #[cfg(feature = "ollama")]
            LLMProvider::Ollama => {
                use ollama_rs::Ollama;

                let req = self.new_ollama_request(history, inference, config)?;
                let res = Ollama::default().send_chat_messages(req).await?;
                Ok(Inference::from_ollama_response(res))
            }
            #[cfg(feature = "anthropic")]
            LLMProvider::Anthropic => Err(crate::Error::Generic(
                "Anthropic provider not yet implemented".to_owned(),
            )),
            _ => Err(crate::Error::Unsupported(
                "Provider not supported for generation".to_owned(),
            )),
        }
    }
}
