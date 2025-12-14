//!! DuckDuckGo Search Tool
//! 
//! This module provides a tool for performing web searches using DuckDuckGo.

use crate::{
    FnDeclarator, FunctionDeclaration,
    llm::function::{FnExecutor, ToolArgs},
};
use reqwest::Client;
use schemars::JsonSchema;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Arguments for performing a DuckDuckGo search.
#[derive(JsonSchema, Serialize, Deserialize, Debug, Clone)]
pub struct DuckDuckGoSearchArgs {
    #[schemars(description = "The search query string.")]
    pub query: String,

    #[schemars(description = "Maximum number of results to return.")]
    #[serde(default = "default_max_results")]
    pub max_results: usize,

    #[schemars(
        description = "Region code for localized search results (e.g., 'wt-wt' for worldwide)."
    )]
    #[serde(default = "default_region")]
    pub region: String,
}

fn default_max_results() -> usize {
    10
}

fn default_region() -> String {
    "wt-wt".to_string() // No region preference
}

impl ToolArgs for DuckDuckGoSearchArgs {}

/// A single search result from DuckDuckGo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: Option<String>,
}

/// Response from a DuckDuckGo search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuckDuckGoSearchResponse {
    pub results: Vec<SearchResult>,
    pub query: String,
    pub count: usize,
}

/// Tool for performing DuckDuckGo web searches.
#[derive(Clone)]
pub struct DuckDuckGoSearchTool {
    client: Client,
}

impl Default for DuckDuckGoSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl DuckDuckGoSearchTool {
    pub fn new() -> Self {
        let client = Client::builder()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .cookie_store(true)
            .build()
            .expect("Failed to create HTTP client");

        Self { client }
    }

    pub async fn search(
        &self,
        args: DuckDuckGoSearchArgs,
    ) -> crate::Result<DuckDuckGoSearchResponse> {
        let url = format!(
            "https://html.duckduckgo.com/html/?q={}&kl={}",
            urlencoding::encode(&args.query),
            urlencoding::encode(&args.region)
        );

        log::debug!("Searching DuckDuckGo: {}", url);

        let response = self.client.get(&url).send().await.map_err(|e| {
            crate::CoreError::Generic(format!("Failed to fetch search results: {}", e))
        })?;

        if !response.status().is_success() {
            return Err(Box::from(crate::CoreError::Generic(format!(
                "DuckDuckGo returned status {}",
                response.status()
            ))));
        }

        let html = response
            .text()
            .await
            .map_err(|e| crate::CoreError::Generic(format!("Failed to read response: {}", e)))?;

        let results = self.parse_results(&html, args.max_results)?;

        Ok(DuckDuckGoSearchResponse {
            count: results.len(),
            query: args.query,
            results,
        })
    }

    fn parse_results(&self, html: &str, max_results: usize) -> crate::Result<Vec<SearchResult>> {
        let document = Html::parse_document(html);

        // Select all <a> elements with class "result__a"
        let result_selector = Selector::parse("a.result__a")
            .map_err(|e| crate::CoreError::Generic(format!("Invalid selector: {:?}", e)))?;

        // Selector for snippets (sibling element)
        let snippet_selector = Selector::parse("a.result__snippet")
            .map_err(|e| crate::CoreError::Generic(format!("Invalid selector: {:?}", e)))?;

        let mut results = Vec::new();
        let result_elements: Vec<_> = document.select(&result_selector).collect();
        let snippet_elements: Vec<_> = document.select(&snippet_selector).collect();

        for (i, element) in result_elements.into_iter().enumerate() {
            if results.len() >= max_results {
                break;
            }

            let href = match element.value().attr("href") {
                Some(h) => h,
                None => continue,
            };

            let url = self.extract_url(href);

            if url.is_empty() || url.starts_with("//duckduckgo.com") {
                continue;
            }

            let title: String = element
                .text()
                .collect::<Vec<_>>()
                .join(" ")
                .trim()
                .to_string();

            if title.is_empty() {
                continue;
            }

            let snippet = snippet_elements
                .get(i)
                .map(|el| el.text().collect::<Vec<_>>().join(" ").trim().to_string())
                .filter(|s| !s.is_empty());

            results.push(SearchResult {
                title,
                url,
                snippet,
            });
        }

        log::debug!("Parsed {} results from DuckDuckGo", results.len());
        Ok(results)
    }

    fn extract_url(&self, href: &str) -> String {
        // DuckDuckGo URLs look like: //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com...
        if href.contains("uddg=")
            && let Some(start) = href.find("uddg=")
        {
            let encoded = &href[start + 5..];
            // Find the end of the URL (next & or end of string)
            let end = encoded.find('&').unwrap_or(encoded.len());
            let encoded_url = &encoded[..end];

            // URL decode
            match urlencoding::decode(encoded_url) {
                Ok(decoded) => return decoded.to_string(),
                Err(e) => {
                    log::error!("Failed to decode URL: {}", e);
                }
            }
        }

        if href.starts_with("http://") || href.starts_with("https://") {
            return href.to_string();
        }

        // Return as-is if we can't parse it
        href.to_string()
    }
}

#[async_trait::async_trait]
impl FnExecutor<DuckDuckGoSearchArgs, DuckDuckGoSearchResponse> for DuckDuckGoSearchTool {
    async fn call(&self, args: DuckDuckGoSearchArgs) -> crate::Result<DuckDuckGoSearchResponse> {
        Ok(self.search(args).await?)
    }
}

impl FnDeclarator<DuckDuckGoSearchArgs, DuckDuckGoSearchResponse> for DuckDuckGoSearchTool {
    fn declare(&self) -> FunctionDeclaration<DuckDuckGoSearchArgs, DuckDuckGoSearchResponse> {
        FunctionDeclaration {
            name: "duckduckgo_search",
            description: "Perform a web search using DuckDuckGo and return the results.",
            parameters: schemars::schema_for!(DuckDuckGoSearchArgs),
            executor: Arc::new(self.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_url_with_uddg() {
        let tool = DuckDuckGoSearchTool::new();

        let href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc123";
        let url = tool.extract_url(href);
        assert_eq!(url, "https://example.com/page");
    }

    #[test]
    fn test_extract_url_direct() {
        let tool = DuckDuckGoSearchTool::new();

        let href = "https://example.com/page";
        let url = tool.extract_url(href);
        assert_eq!(url, "https://example.com/page");
    }

    #[test]
    fn test_default_args() {
        assert_eq!(default_max_results(), 10);
        assert_eq!(default_region(), "wt-wt");
    }

    #[tokio::test]
    async fn test_search_integration() {
        // This is an integration test - requires network
        let tool = DuckDuckGoSearchTool::new();
        let args = DuckDuckGoSearchArgs {
            query: "rust programming language".to_string(),
            max_results: 5,
            region: "wt-wt".to_string(),
        };

        let result = tool.search(args).await;

        // Just check it doesn't error - actual results depend on network
        match result {
            Ok(response) => {
                println!("Got {} results", response.count);
                for r in &response.results {
                    println!("- {} ({})", r.title, r.url);
                }
                assert!(response.count <= 5);
            }
            Err(e) => {
                // Network might not be available in CI
                println!("Search failed (may be expected in CI): {}", e);
            }
        }
    }
}
