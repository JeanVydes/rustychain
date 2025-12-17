use crate::{GenerationConfig, LLM, LLMProvider, LLMStreaming, conversation::Inference};
use futures_core::Stream;
use futures_util::stream::{StreamExt, TryStreamExt};
use ollama_rs::Ollama;
use std::pin::Pin;

#[async_trait::async_trait]
impl LLMStreaming for LLM {
    /// Streams text generation results based on the provided history and inference.
    #[allow(unreachable_patterns)]
    async fn stream(
        &self,
        history: &[Inference],
        inference: &Inference,
        config: GenerationConfig,
    ) -> crate::Result<Pin<Box<dyn Stream<Item = crate::Result<Inference>> + Send + 'static>>> {
        match self.provider {
            #[cfg(feature = "google")]
            LLMProvider::Google => {
                let req = self.new_google_request(history, inference, config)?;
                let stream = req.execute_stream().await?;
                let mapped = stream.into_stream().map(|res| match res {
                    Ok(gen_resp) => Ok(Inference::from_gemini_response(gen_resp)),
                    Err(e) => Err(crate::Error::Gemini(e)),
                });

                Ok(Box::pin(mapped))
            }
            #[cfg(feature = "openai")]
            LLMProvider::OpenAI => {
                let mut client = self.get_openai_client()?;
                let req = self.new_openai_stream_request(history, inference, config)?;
                let stream = client.chat_completion_stream(req).await?;
                let mapped = stream.map(Inference::from_openai_stream_response);
                Ok(Box::pin(mapped))
            }
            #[cfg(feature = "ollama")]
            LLMProvider::Ollama => {
                let req = self.new_ollama_request(history, inference, config)?;
                let res = Ollama::default().send_chat_messages_stream(req).await?;

                let mapped = res.map(|res| match res {
                    Ok(ollama_msg) => Ok(Inference::from_ollama_response(ollama_msg)),
                    Err(_e) => Err(crate::Error::Generic("Ollama stream error".to_owned())),
                });

                Ok(Box::pin(mapped))
            }
            #[cfg(feature = "anthropic")]
            LLMProvider::Anthropic => Err(crate::Error::Generic(
                "Streaming not yet implemented for this provider".to_owned(),
            )),
            _ => Err(crate::Error::Unsupported(
                "Provider not supported for streaming".to_owned(),
            )),
        }
    }
}
