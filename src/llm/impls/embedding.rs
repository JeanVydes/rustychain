use crate::{CoreError, LLM, LLMEmbedding, LLMProvider};

#[async_trait::async_trait]
impl LLMEmbedding for LLM {
    /// Generates an embedding for the given text.
    async fn embedding(&self, text: &str, dim: i32) -> crate::Result<Vec<f32>> {
        match self.provider {
            #[cfg(feature = "google")]
            LLMProvider::Google => {
                let client = self.get_gemini_client()?;

                let res = client
                    .embed_content()
                    .with_output_dimensionality(dim)
                    .with_text(text)
                    .execute()
                    .await?;

                Ok(res.embedding.values)
            }
            #[cfg(feature = "openai")]
            LLMProvider::OpenAI => {
                use std::error::Error;

                let mut client = self.get_openai_client()?;

                let mut req = openai_api_rs::v1::embedding::EmbeddingRequest::new(
                    self.name.clone(),
                    vec![text.to_owned()],
                );
                req.dimensions = Some(dim);

                let res =
                    client
                        .embedding(req)
                        .await
                        .map_err(|e| -> Box<dyn Error + Send + Sync> {
                            Box::from(CoreError::OpenAI(format!(
                                "OpenAI embedding request failed: {:?}",
                                e
                            )))
                        })?;

                if let Some(data) = res.data.first() {
                    Ok(data.embedding.clone())
                } else {
                    Err(Box::from(CoreError::Generic(
                        "No embedding data returned from OpenAI".to_owned(),
                    )))
                }
            }
            _ => Err(Box::from(CoreError::Unsupported(
                "Embedding not supported for this provider".to_owned(),
            ))),
        }
    }
}
