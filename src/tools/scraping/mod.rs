use crate::{FnDeclarator, FnExecutor, FunctionDeclaration, ToolArgs, util::UserAgentFactory};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
pub struct ScrappingArgs {
    #[schemars(description = "The URL to scrape data from.")]
    pub query: String,
    #[schemars(description = "Optional scrapping options.")]
    pub options: Option<ScrappingOptions>,
}

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
pub struct ScrappingOptions {
    pub user_agent: Option<String>,
    pub css_selectors: Option<Vec<String>>,
    pub exclude_selectors: Option<Vec<String>>,
}

impl ToolArgs for ScrappingArgs {}

#[derive(Debug, Clone)]
pub struct ScrappingTool;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrappingResult {
    pub html: String,
}

#[async_trait::async_trait]
impl FnExecutor<ScrappingArgs, ScrappingResult> for ScrappingTool {
    async fn call(&self, args: ScrappingArgs) -> crate::Result<ScrappingResult> {
        let url = url::Url::parse(&args.query).map_err(crate::CoreError::URL)?;

        let client = reqwest::Client::builder()
            .user_agent(
                args.options
                    .as_ref()
                    .and_then(|opts| opts.user_agent.clone())
                    .unwrap_or_else(UserAgentFactory::random),
            )
            .build()
            .map_err(crate::CoreError::Reqwest)?;

        let html = client.get(url).send().await?.text().await?;

        if html.is_empty() {
            return Err(Box::from(crate::CoreError::NoContent));
        }

        if let Some(options) = args.options {
            if let Some(selectors) = options.css_selectors {
                let document = scraper::Html::parse_document(&html);
                let mut extracted_html = String::new();
                for selector_str in selectors {
                    let selector = scraper::Selector::parse(&selector_str)
                        .map_err(|err| crate::CoreError::Scraper(err.to_string()))?;
                    for element in document.select(&selector) {
                        extracted_html.push_str(&element.html());
                    }
                }
                return Ok(ScrappingResult {
                    html: extracted_html,
                });
            }

            if let Some(exclude_selectors) = options.exclude_selectors {
                let document = scraper::Html::parse_document(&html);
                let mut filtered_html = html.clone();
                for selector_str in exclude_selectors {
                    let selector = scraper::Selector::parse(&selector_str)
                        .map_err(|err| crate::CoreError::Scraper(err.to_string()))?;
                    for element in document.select(&selector) {
                        filtered_html = filtered_html.replace(&element.html(), "");
                    }
                }
                return Ok(ScrappingResult {
                    html: filtered_html,
                });
            }
        }

        Ok(ScrappingResult {
            html: html.to_string(),
        })
    }
}

impl FnDeclarator<ScrappingArgs, ScrappingResult> for ScrappingTool {
    fn declare(&self) -> FunctionDeclaration<ScrappingArgs, ScrappingResult> {
        FunctionDeclaration {
            name: "scrape_web",
            description: "Scrape data from the web based on the provided url with optional config.",
            parameters: schemars::schema_for!(ScrappingArgs),
            executor: std::sync::Arc::new(self.clone()),
        }
    }
}
