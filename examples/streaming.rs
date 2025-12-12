use rustychain::prelude::*;
use rustychain::{LLM, LLMProvider, Message};
use futures_util::stream::StreamExt;

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let auth = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let llm = LLM::builder()
        .set_authorization(auth)
        .set_name("gpt-5.2-2025-12-11".to_owned())
        .set_provider(LLMProvider::OpenAI)
        .set_system_prompt("You are a helpful assistant.".to_owned())
        .build()?;

    let mut stream = llm
        .inference()
        .with_message(Message::user("Tell me a joke about computers."))
        .stream()
        .await?;

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(part) => {
                if let Some(text) = part.message {
                    print!("{}", text);
                }
            }
            Err(_) => break,
        }
    }

    Ok(())
}
