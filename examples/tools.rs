use rustychain::prelude::*;
use rustychain::tools::integrations::search::DuckDuckGoSearchTool;
use rustychain::{LLM, LLMProvider};

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
        .set_model("gemini-2.5-flash")
        .set_provider(LLMProvider::Google)
        .set_system_prompt("You are a helpful assistant.")
        .add_tool(duckduckgo_tool.declare())
        .build()?;

    let response = llm
        .inference("What is the capital of Colombia?")
        .generate()
        .await?;

    log::info!("Response: {}", response);

    Ok(())
}
