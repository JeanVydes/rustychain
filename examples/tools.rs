use rustychain::prelude::*;
use rustychain::tools::search::DuckDuckGoSearchTool;
use rustychain::{LLM, LLMProvider, Message};

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let auth = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let duckduckgo_tool = DuckDuckGoSearchTool::new();

    let llm = LLM::builder()
        .set_authorization(auth)
        .set_name("gemini-2.5-flash".to_owned())
        .set_provider(LLMProvider::Google)
        .set_system_prompt("You are a helpful assistant.".to_owned())
        .add_tool(duckduckgo_tool.declare().into())
        .build()?;

    let response = llm
        .inference()
        .with_message(Message::user("What is the capital of Colombia?"))
        .generate()
        .await?;

    log::info!("Response: {}", response);

    Ok(())
}
