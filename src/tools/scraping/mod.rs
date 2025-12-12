use crate::{FnDeclarator, FnExecutor, FunctionDeclaration, ToolArgs};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
pub struct ScrappingArgs {
    pub query: String,
}

impl ToolArgs for ScrappingArgs {}

#[derive(Debug, Clone)]
pub struct ScrappingTool;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrappingResult {
    pub data: String,
}

#[async_trait::async_trait]
impl FnExecutor<ScrappingArgs, ScrappingResult> for ScrappingTool {
    async fn call(&self, args: ScrappingArgs) -> crate::Result<ScrappingResult> {
        Ok(ScrappingResult {
            data: format!("Scraped data for query: {}", args.query),
        })
    }
}

impl FnDeclarator<ScrappingArgs, ScrappingResult> for ScrappingTool {
    fn declare(&self) -> FunctionDeclaration<ScrappingArgs, ScrappingResult> {
        FunctionDeclaration {
            name: "scrape_data",
            description: "Scrape data from the web based on the provided query.",
            parameters: schemars::schema_for!(ScrappingArgs),
            executor: std::sync::Arc::new(self.clone()),
        }
    }
}
