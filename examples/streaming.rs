use futures_util::stream::StreamExt;
use rustychain::{LLM, LLMProvider};

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let auth = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let llm = LLM::builder()
        .set_authorization(auth)
        .set_model("gpt-5.2-2025-12-11")
        .set_provider(LLMProvider::OpenAI)
        .set_system_prompt("You are a helpful assistant.")
        .build()?;

    let mut stream = llm
        .inference("Tell me a joke about computers.")
        .stream()
        .await?;

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(part) => {
                if let Some(text) = part.content.text {
                    log::info!("{}", text);
                }
            }
            Err(_) => break,
        }
    }

    Ok(())
}
