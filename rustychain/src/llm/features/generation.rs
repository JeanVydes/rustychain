use std::sync::Arc;

use crate::prelude::*;

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
        let config = Arc::new(config);
        log::trace!("[LLM_GENERATION][REQUEST]");
        log::trace!("[LLM_GENERATION][HISTORY] {:?}", history);
        log::trace!("[LLM_STREAMING][PROVIDER] {:?}", self.provider);
        log::trace!("[LLM_GENERATION][INFERENCE] {:?}", inference);
        log::trace!("[LLM_GENERATION][GENERATION_CONFIG] {:?}", config);

        match self.provider {
            #[cfg(feature = "google")]
            LLMProvider::Google => {
                let req = self.new_google_request(history, inference, &config)?;
                let res = req.execute().await?;
                log::trace!("[LLM_GENERATION/GOOGLE][RAW_RESPONSE] {:?}", res);
                Ok(Inference::from_google_response(res, Some(config)))
            }

            #[cfg(feature = "openai")]
            LLMProvider::OpenAI => {
                let mut client = self.get_openai_client()?;
                let req = self.new_openai_request(history, inference, &config)?;
                let res = client.chat_completion(req).await?;

                log::trace!("[LLM_GENERATION/OPENAI][RAW_RESPONSE] {:?}", res);

                if let Some(choice) = res.choices.first() {
                    Ok(Inference::from_openai_choice(choice, Some(config)))
                } else {
                    Err(crate::Error::Generic(
                        "No choices returned from OpenAI".to_owned(),
                    ))
                }
            }
            #[cfg(feature = "ollama")]
            LLMProvider::Ollama => {
                let req = self.new_ollama_request(history, inference, &config)?;
                let res = ollama_rs::Ollama::default().send_chat_messages(req).await?;
                log::trace!("[LLM_GENERATION/OLLAMA][RAW_RESPONSE] {:?}", res);
                Ok(Inference::from_ollama_response(res, Some(config)))
            }
            #[cfg(feature = "anthropic")]
            LLMProvider::Anthropic => Err(crate::Error::Generic(
                "Anthropic provider not yet implemented".to_owned(),
            )),
            #[cfg(feature = "openrouter")]
            LLMProvider::OpenRouter => {
                let client = self.get_openrouter_client()?;
                let req = self.new_openrouter_request(history, inference, &config)?;
                let res = client.send_chat_completion(&req).await.map_err(|e| {
                    crate::Error::Generic(format!("OpenRouter request failed: {}", e))
                })?;

                log::trace!("[LLM_GENERATION/OPENROUTER][RAW_RESPONSE] {:?}", res);

                Ok(Inference::from_openrouter_response(&res, Some(config)))
            }
            _ => Err(crate::Error::Unsupported(
                "Provider not supported for generation".to_owned(),
            )),
        }
    }
}
