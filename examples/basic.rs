use rustychain::{LLM, LLMProvider};

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let auth = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let llm = LLM::builder()
        .set_authorization(auth)
        .set_model("gemini-2.5-flash")
        .set_provider(LLMProvider::Google)
        .set_system_prompt("You are a helpful assistant.")
        .build()?;

    let response = llm.inference("Hello, how are you?").generate().await?;

    log::info!("Response: {}", response);

    Ok(())
}
