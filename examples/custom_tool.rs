use rustychain::{Message, prelude::*};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(JsonSchema, Serialize, Deserialize, Debug, Clone, ToolArgs)]
pub struct SumArgs {
    #[schemars(description = "First integer to sum.")]
    pub a: i64,
    #[schemars(description = "Second integer to sum.")]
    pub b: i64,
}

#[declare_function(
    name = "sum_integers",
    description = "Returns the sum of two integers.",
    args = SumArgs,
    result = i64
)]
pub struct SumTool {}

impl SumTool {
    pub async fn execute(&self, args: SumArgs) -> rustychain::Result<i64> {
        Ok(args.a + args.b)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let auth = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let sum_tool = SumTool {};

    let llm = LLM::builder()
        .set_authorization(auth)
        .set_name("gemini-2.5-flash".to_owned())
        .set_provider(LLMProvider::Google)
        .set_system_prompt("You are a helpful assistant.".to_owned())
        .add_tool(sum_tool.declare().into())
        .build()?;

    let response = llm
        .inference()
        .with_message(Message::user("What is the sum of 42 and 58?"))
        .generate()
        .await?;

    println!("Response: {}", response);

    Ok(())
}
