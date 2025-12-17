use std::sync::Arc;

use rustychain::prelude::*;
use rustychain::{LLM, LLMProvider};

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let auth = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let llm = Arc::new(
        LLM::builder()
            .set_authorization(auth.clone())
            .set_model("gemini-embedding-001".to_owned())
            .set_provider(LLMProvider::Google)
            .build()?,
    );

    // The dimensionality depends on the model/provider used
    // And own requirements
    // In this example `gemini-embedding-001` supports up to 3072 dimensions, but we use 1536
    let vector = llm.embedding("Hello, world!", 1536).await?;

    log::info!("Embedding: {:?}", vector);

    Ok(())
}
