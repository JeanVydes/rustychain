use crate::prelude::*;
use futures_core::Stream;
use futures_util::stream::{StreamExt, TryStreamExt};
use ollama_rs::Ollama;
use std::{pin::Pin, sync::Arc};

#[async_trait::async_trait]
impl LLMStreaming for LLM {
    /// Streams text generation results based on the provided history and inference.
    #[allow(unreachable_patterns)]
    async fn stream(
        &self,
        history: &[Inference],
        inference: &Inference,
        config: Arc<GenerationConfig>,
    ) -> crate::Result<Pin<Box<dyn Stream<Item = crate::Result<Inference>> + Send + 'static>>> {
        log::trace!("[LLM_STREAMING][REQUEST]");
        log::trace!("[LLM_STREAMING][PROVIDER] {:?}", self.provider);
        log::trace!("[LLM_STREAMING][INFERENCE] {:?}", inference);
        log::trace!("[LLM_STREAMING][GENERATION_CONFIG] {:?}", config);

        match self.provider {
            #[cfg(feature = "google")]
            LLMProvider::Google => {
                let req = self.new_google_request(history, inference, &config)?;
                let stream = req.execute_stream().await?;
                let mapped = stream.into_stream().map(move |res| match res {
                    Ok(gen_resp) => Ok(Inference::from_google_response(
                        gen_resp,
                        Some(config.clone()),
                    )),
                    Err(e) => Err(crate::Error::from(e)),
                });

                Ok(Box::pin(mapped))
            }
            #[cfg(feature = "openai")]
            LLMProvider::OpenAI => {
                let mut client = self.get_openai_client()?;
                let req = self.new_openai_stream_request(history, inference, &config)?;
                let stream = client.chat_completion_stream(req).await?;
                let mapped = stream.map(Inference::from_openai_stream_response);
                Ok(Box::pin(mapped))
            }
            #[cfg(feature = "ollama")]
            LLMProvider::Ollama => {
                let req = self.new_ollama_request(history, inference, &config)?;
                let res = Ollama::default().send_chat_messages_stream(req).await?;
                let mapped = res.map(move |res| match res {
                    Ok(ollama_msg) => Ok(Inference::from_ollama_response(
                        ollama_msg,
                        Some(config.clone()),
                    )),
                    Err(_e) => Err(crate::Error::Generic("Ollama stream error".to_owned())),
                });

                Ok(Box::pin(mapped))
            }
            #[cfg(feature = "anthropic")]
            LLMProvider::Anthropic => Err(crate::Error::Generic(
                "Streaming not yet implemented for this provider".to_owned(),
            )),
            #[cfg(feature = "openrouter")]
            LLMProvider::OpenRouter => {
                let client = self.get_openrouter_client()?;
                let req = self.new_openrouter_request(history, inference, &config)?;
                let res = client.stream_chat_completion(&req).await.map_err(|e| {
                    crate::Error::Generic(format!("OpenRouter request failed: {}", e))
                })?;
                let mapped = res.map(move |res| match res {
                    Ok(openrouter_chunk) => Ok(Inference::from_openrouter_response(
                        &openrouter_chunk,
                        Some(config.clone()),
                    )),
                    Err(e) => Err(crate::Error::Generic(format!(
                        "OpenRouter stream error: {}",
                        e
                    ))),
                });
                Ok(Box::pin(mapped))
            }
            _ => Err(crate::Error::Unsupported(
                "Provider not supported for streaming".to_owned(),
            )),
        }
    }
}
