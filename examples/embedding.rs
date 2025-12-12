use std::sync::Arc;

use rustychain::prelude::*;
use rustychain::{LLM, LLMProvider};

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let auth = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let embedding = Arc::new(
        LLM::builder()
            .set_authorization(auth.clone())
            .set_name("gemini-embedding-001".to_owned())
            .set_provider(LLMProvider::Google)
            .build()?,
    );

    let vector = embedding.embedding("Hello, world!", 1536).await?;

    println!("Embedding: {:?}", vector);

    Ok(())
}
