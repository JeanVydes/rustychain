use crate::{CoreError, GenerationConfig, LLM, LLMProvider, LLMStreaming, Message};
use futures_core::Stream;
use futures_util::stream::{StreamExt, TryStreamExt};
use ollama_rs::Ollama;
use std::{error::Error, pin::Pin};

#[async_trait::async_trait]
impl LLMStreaming for LLM {
    /// Streams text generation results based on the provided history and message.
    #[allow(unreachable_patterns)]
    async fn stream(
        &self,
        history: &[Message],
        message: &Message,
        config: GenerationConfig,
    ) -> crate::Result<Pin<Box<dyn Stream<Item = crate::Result<Message>> + Send + 'static>>> {
        match self.provider {
            #[cfg(feature = "google")]
            LLMProvider::Google => {
                let req = self.new_google_request(history, message, config)?;
                let stream = req.execute_stream().await?;
                let mapped = stream.into_stream().map(|res| match res {
                    Ok(gen_resp) => Ok(Message::from_gemini(gen_resp)),
                    Err(e) => Err(Box::new(CoreError::Gemini(e)) as Box<dyn Error + Send + Sync>),
                });

                Ok(Box::pin(mapped))
            }
            #[cfg(feature = "openai")]
            LLMProvider::OpenAI => {
                let mut client = self.get_openai_client()?;
                let req = self.new_openai_stream_request(history, message, config)?;
                let stream = client.chat_completion_stream(req).await.map_err(
                    |_| -> Box<dyn Error + Send + Sync> {
                        Box::from(CoreError::OpenAI("OpenAI stream request failed".to_owned()))
                    },
                )?;
                let mapped = stream.map(Message::from_openai_chat_completion_stream_response);
                Ok(Box::pin(mapped))
            }
            #[cfg(feature = "ollama")]
            LLMProvider::Ollama => {
                let req = self.new_ollama_request(history, message, config)?;
                let res = Ollama::default().send_chat_messages_stream(req).await?;

                let mapped = res.map(|res| match res {
                    Ok(ollama_msg) => Ok(Message::from_ollama(ollama_msg)),
                    Err(_e) => Err(
                        Box::new(CoreError::Generic("Ollama stream error".to_owned()))
                            as Box<dyn Error + Send + Sync>,
                    ),
                });

                Ok(Box::pin(mapped))
            }
            #[cfg(feature = "anthropic")]
            LLMProvider::Anthropic => Err(Box::from(CoreError::Generic(
                "Streaming not yet implemented for this provider".to_owned(),
            ))),
            _ => Err(Box::from(CoreError::Unsupported(
                "Provider not supported for streaming".to_owned(),
            ))),
        }
    }
}
