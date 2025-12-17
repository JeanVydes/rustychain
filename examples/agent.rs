use rustychain::agent::definitions::{Agent, AgentStep};
use rustychain::tools::scraping::ScrappingTool;
use rustychain::tools::search::DuckDuckGoSearchTool;
use rustychain::{GenerationConfig, Inference, prelude::*};
use rustychain::{LLM, LLMProvider};
use std::sync::Arc;

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_env_filter(
            tracing_subscriber::EnvFilter::new("debug")
                .add_directive("html5ever=off".parse().unwrap()) // Disable html5ever
                .add_directive("selectors=off".parse().unwrap()), // Disable selectors
        )
        .init();

    let auth = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let duckduckgo_tool = DuckDuckGoSearchTool::new();
    let scraping_tool = ScrappingTool::new();

    let llm = Arc::new(
        LLM::builder()
            .set_authorization(auth)
            .set_model("gemini-2.5-flash".to_owned())
            .set_provider(LLMProvider::Google)
            .set_system_prompt(SYSTEM_PROMPT.to_owned())
            .add_tool(duckduckgo_tool.declare().into())
            .add_tool(scraping_tool.declare().into())
            .build()?,
    );

    let mut agent = Agent::builder(llm)
        .name("BiographicalResearchAgent".to_owned())
        .generation_config(GenerationConfig::default())
        .initial(Inference::as_user("Research about the Pibe Valderrama"))
        .build();

    let mut step_count = 0;
    while let Some(step) = agent.next().await && step_count <= 20 {
        step_count += 1;
        if let Ok(step) = step {
            match step {
                AgentStep::Finished(_) => {
                    log::info!("🏆 Agent has completed its task in {} steps.", step_count);
                }
                AgentStep::ToolReturn(message) => {
                    for result in message.function_results {
                        log::info!("🛠 Tool Result: {}", result.results);
                    }
                }
            }
        }
    }
    Ok(())
}

pub const SYSTEM_PROMPT: &str = r#"You are an intelligent research assistant with web search and scraping capabilities.

## Your Capabilities
- **Search Tool (duckduckgo_search)**: Find relevant web pages and information
- **Scraping Tool (scrape_web)**: Extract detailed content from specific URLs

## Workflow Instructions
1. **Search First**: Use duckduckgo_search to find relevant sources
2. **Scrape Next**: Use scrape_web on promising URLs to get detailed information
3. **Extract Facts**: Carefully read scraped content to find specific information
4. **Verify**: Cross-reference information across multiple sources when possible
5. **Respond**: Only after gathering sufficient information, provide your final answer

## Tool Usage Guidelines

### DuckDuckGo Search
- Keep queries concise (2-8 words)
- Start broad, then refine if needed
- Example: "Pibe Valderrama footballer" or "Carlos Valderrama biography"

### Web Scraping
- **Mode Selection**:
  - Use `readability` mode for articles, biographies, and blogs
  - Use `selective` mode if you know specific CSS selectors
  - Use `text_only` for simple text extraction
- **Output Format**:
  - Use `markdown` for better readability
  - Use `plain_text` for simple information
- **Always** set `include_metadata: true` to get titles and descriptions
- **Example for biographical info**:
  ```json
  {
    "url": "https://example.com",
    "options": {
      "mode": "readability",
      "output_format": "markdown",
      "include_metadata": true,
      "extract_links": false,
      "clean_content": true
    }
  }
  ```

## Critical Rules
- **ALWAYS** scrape at least 5-8 sources before answering
- **NEVER** make up information - only use scraped content
- **CONTINUE** using tools until you have found the specific information requested
- **ONLY STOP** when you have a complete, verified answer
- If information is not found in scraped content, search again with different queries
- Extract specific facts (dates, names, numbers) directly from scraped text

## Response Format
When you have gathered all information:
1. State the answer clearly
2. Cite your sources (URLs)
3. Include relevant context if helpful

Remember: The conversation ends when you stop making tool calls, so ensure you have all needed information before responding with your final answer.
"#;
