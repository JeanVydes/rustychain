use rustychain::prelude::*;
use rustychain::{LLM, LLMProvider, Message};

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let auth = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let llm = LLM::builder()
        .set_authorization(auth)
        .set_name("gemini-2.5-flash".to_owned())
        .set_provider(LLMProvider::Google)
        .set_system_prompt("You are a helpful assistant.".to_owned())
        .build()?;

    let response = llm
        .inference()
        .with_message(Message::user("What is the capital of Colombia?"))
        .generate()
        .await?;

    println!("Response: {}", response);

    Ok(())
}
