#[cfg(feature = "google")]
use gemini_rust::Gemini;
#[cfg(feature = "openai")]
use openai_api_rs::v1::api::OpenAIClient;

use crate::LLM;

impl LLM {
    /// Creates a Gemini client configured for this LLM.
    #[cfg(feature = "google")]
    pub fn get_gemini_client(&self) -> crate::Result<Gemini> {
        use gemini_rust::GeminiBuilder;

        let authorization = match &self.authorization {
            Some(a) => a,
            None => return Err(crate::Error::Unauthorized("Not auth".to_owned())),
        };

        let mut client =
            GeminiBuilder::new(authorization.clone()).with_model(format!("models/{}", self.model));

        if let Some(endpoint) = &self.endpoint {
            client = client.with_base_url(url::Url::parse(endpoint)?);
        }

        let client = client.build()?;

        Ok(client)
    }

    /// Creates an OpenAI client configured for this LLM.
    #[cfg(feature = "openai")]
    pub fn get_openai_client(&self) -> crate::Result<OpenAIClient> {
        use openai_api_rs::v1::api::OpenAIClientBuilder;

        let authorization = match &self.authorization {
            Some(a) => a,
            None => return Err(crate::Error::Unauthorized("Not auth".to_owned())),
        };

        let mut client = OpenAIClientBuilder::new().with_api_key(authorization.clone());

        if let Some(endpoint) = &self.endpoint {
            client = client.with_endpoint(endpoint.clone());
        }

        let client = client.build()?;

        Ok(client)
    }

    #[cfg(feature = "openrouter")]
    pub fn get_openrouter_client(&self) -> crate::Result<openrouter_rs::OpenRouterClient> {
        let authorization = match &self.authorization {
            Some(a) => a,
            None => return Err(crate::Error::Unauthorized("Not auth".to_owned())),
        };

        let mut client = openrouter_rs::OpenRouterClient::builder();
        let mut client = client.api_key(authorization.clone());

        if let Some(endpoint) = &self.endpoint {
            client = client.base_url(endpoint.clone());
        }

        let client = client
            .build()
            .map_err(|e| crate::Error::Internal(e.to_string().into()))?;

        Ok(client)
    }
}
