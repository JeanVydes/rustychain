use rustychain::chain::definitions::ChainBuilder;
use rustychain::chain::step::Runnable;
use rustychain::prelude::*;
use rustychain::{LLM, LLMProvider, Message};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Input {
    text: String,
}

#[derive(Debug, Clone)]
pub struct Output {
    parts: Vec<String>,
}

pub struct Formatter {}

#[async_trait::async_trait]
impl Runnable<String, Output> for Formatter {
    async fn run(&self, input: Arc<String>) -> rustychain::Result<Output> {
        Ok(Output {
            parts: input
                .split(" ")
                .map(|line| line.to_lowercase().trim().to_string())
                .filter(|line| !line.is_empty())
                .collect(),
        })
    }
}

pub struct Summarizer {
    llm: Arc<LLM>,
}

#[async_trait::async_trait]
impl Runnable<Input, String> for Summarizer {
    async fn run(&self, input: Arc<Input>) -> rustychain::Result<String> {
        let prompt: String = format!(
            "Summarize the following text in one sentence:\n\n{}",
            input.text
        );
        let response = self
            .llm
            .inference()
            .with_message(Message::user(&prompt))
            .generate()
            .await?;

        Ok(response.message.unwrap_or_default())
    }
}

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let auth = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let llm = Arc::new(
        LLM::builder()
            .set_authorization(auth)
            .set_name("gemini-2.5-flash".to_owned())
            .set_provider(LLMProvider::Google)
            .set_system_prompt("You are a helpful assistant.".to_owned())
            .build()?,
    );

    let mut chain =
        ChainBuilder::starts("summarization".to_owned(), Summarizer { llm: llm.clone() })
            .add_step("formatting".to_owned(), Formatter {})
            .set_input(Input {
                text: EXAMPLE_TEST.to_owned(),
            });

    while let Some(r) = chain.next().await {
        let step_result = r?;
        match step_result.name.as_str() {
            "summarization" => {
                if let Some(input) = step_result.input.downcast_ref::<Input>() {
                    log::info!("📥 Summarization Input: {}", &input.text[..100]);
                }
                if let Some(output) = step_result.output.downcast_ref::<String>() {
                    log::info!("📤 Summarization Output: {}", output);
                }
            }
            "formatting" => {
                if let Some(input) = step_result.input.downcast_ref::<String>() {
                    log::info!("📥 Formatting Input: {}", input);
                }
                if let Some(output) = step_result.output.downcast_ref::<String>() {
                    log::info!("📤 Formatting Output: {}", output);
                }
            }
            _ => {}
        };
    }

    let final_output = chain.finalize()?;
    log::info!("Final Output: {:?}", final_output.parts);

    return Ok(());
}

const EXAMPLE_TEST: &str = r#"
Calculus Basics: Derivatives and Integrals

Calculus is fundamentally the mathematical study of change, motion, and accumulation. It is divided into two core concepts: differentiation (derivatives) and integration (integrals). These two concepts are inversely related, much like addition and subtraction.
1. Derivatives

A derivative is a tool used to determine the instantaneous rate of change of a function with respect to one of its variables. It answers the question, "How fast is this changing right now?" Geometrically, calculating the derivative of a function at a specific point gives the slope of the tangent line to the curve at that point . For someone just learning, the simplest analogy is a car's speedometer. If a function represents the distance a car has traveled over time, the derivative of that function is the speedometer's reading—the exact speed at that precise moment in time. It measures the change in distance over an infinitesimally small change in time.
2. Integrals

An integral is the opposite process, known as anti-differentiation. It is a method for summing up small slices of information to find the total accumulation or the net effect over a given interval. Most commonly, an integral is used to calculate the area under a curve . The analogy here is the car's odometer. If the function you are integrating represents the car's speed over time, the integral of that function will give you the total distance traveled from the start of the trip to the end. It accumulates all the small contributions of speed over the entire duration to find a total amount.

In essence, derivatives break down a problem to find the rate of change at a moment, while integrals build up small pieces to find the total accumulation over a period.
"#;
