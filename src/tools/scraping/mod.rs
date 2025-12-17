//!! Advanced Web Scraping Tool
//!
//! A comprehensive and LLM-friendly web scraping tool with intelligent content extraction,
//! cleaning, and formatting capabilities.

use crate::{
    FnDeclarator, FnExecutor, FunctionDeclaration, ToolArgs,
    prelude::{Cleaner, Formatter},
    util::{
        UserAgentFactory,
        formatters::{html::HtmlFormatter, whitespace::WhitespaceFormatter},
    },
};
use schemars::JsonSchema;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Extraction mode for scraping
#[derive(JsonSchema, Serialize, Deserialize, Debug, Clone, Copy, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionMode {
    /// Return full HTML content
    #[default]
    Full,
    /// Extract main content using readability algorithm
    Readability,
    /// Extract using CSS selectors
    Selective,
    /// Extract only text content
    TextOnly,
}

/// Output format for scraped content
#[derive(JsonSchema, Serialize, Deserialize, Debug, Clone, Copy, Default)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    /// HTML format
    #[default]
    Html,
    /// Markdown format
    Markdown,
    /// Plain text
    PlainText,
    /// JSON with structured data
    Json,
}

/// Arguments for web scraping
#[derive(JsonSchema, Serialize, Deserialize, Debug)]
pub struct ScrappingArgs {
    #[schemars(description = "The URL to scrape data from (must be a valid HTTP/HTTPS URL).")]
    pub url: String,

    #[schemars(description = "Optional scraping options for controlling extraction behavior.")]
    pub options: Option<ScrappingOptions>,
}

/// Options for customizing scraping behavior
#[derive(JsonSchema, Serialize, Deserialize, Debug, Default)]
pub struct ScrappingOptions {
    #[schemars(
        description = "Extraction mode: 'full' (entire page), 'readability' (main content), 'selective' (CSS selectors), or 'text_only'."
    )]
    pub mode: Option<ExtractionMode>,

    #[schemars(description = "Output format: 'html', 'markdown', 'plain_text', or 'json'.")]
    pub output_format: Option<OutputFormat>,

    #[schemars(
        description = "Custom User-Agent string. If not provided, a random one will be used."
    )]
    pub user_agent: Option<String>,

    #[schemars(
        description = "CSS selectors to extract specific elements. Used when mode is 'selective'."
    )]
    pub css_selectors: Option<Vec<String>>,

    #[schemars(description = "CSS selectors for elements to exclude from results.")]
    pub exclude_selectors: Option<Vec<String>>,

    #[schemars(
        description = "Whether to include page metadata (title, description, author, etc.)."
    )]
    pub include_metadata: Option<bool>,

    #[schemars(description = "Whether to extract all links from the page.")]
    pub extract_links: Option<bool>,

    #[schemars(description = "Whether to extract all images with their URLs and alt text.")]
    pub extract_images: Option<bool>,

    #[schemars(description = "Whether to extract and format tables.")]
    pub extract_tables: Option<bool>,

    #[schemars(
        description = "Maximum content length in characters. Content will be truncated if longer."
    )]
    pub max_length: Option<usize>,

    #[schemars(description = "Whether to remove extra whitespace and clean up formatting.")]
    pub clean_content: Option<bool>,

    #[schemars(description = "Request timeout in seconds (default: 30).")]
    pub timeout_secs: Option<u64>,

    #[schemars(description = "Whether to follow redirects (default: true).")]
    pub follow_redirects: Option<bool>,
}

impl ToolArgs for ScrappingArgs {}

/// Web scraping tool with advanced extraction capabilities
#[derive(Debug, Clone)]
pub struct ScrappingTool {}

impl Default for ScrappingTool {
    fn default() -> Self {
        Self::new()
    }
}

impl ScrappingTool {
    /// Create a new scraping tool instance
    pub fn new() -> Self {
        Self {}
    }

    /// Get or create an HTTP client with the specified configuration
    async fn get_client(&self, options: &ScrappingOptions) -> crate::Result<reqwest::Client> {
        let user_agent = options
            .user_agent
            .clone()
            .unwrap_or_else(UserAgentFactory::random);

        let timeout = Duration::from_secs(options.timeout_secs.unwrap_or(30));
        let follow_redirects = options.follow_redirects.unwrap_or(true);

        let client = reqwest::Client::builder()
            .user_agent(user_agent)
            .timeout(timeout)
            .redirect(if follow_redirects {
                reqwest::redirect::Policy::limited(10)
            } else {
                reqwest::redirect::Policy::none()
            })
            .build()?;

        Ok(client)
    }

    /// Perform the scraping operation
    pub async fn scrape(&self, args: ScrappingArgs) -> crate::Result<ScrappingResult> {
        // Validate URL
        let url = url::Url::parse(&args.url)?;

        if url.scheme() != "http" && url.scheme() != "https" {
            return Err(crate::Error::Validation {
                message: "URL must use HTTP or HTTPS protocol".to_string(),
            });
        }

        let options = args.options.unwrap_or_default();
        let client = self.get_client(&options).await?;

        log::trace!("Scraping URL: {}", url);

        // Fetch the page
        let response = client.get(url.as_str()).send().await?;

        // Check status
        if !response.status().is_success() {
            return Err(crate::Error::HTTP(
                format!(
                    "Received HTTP {} when fetching URL: {}",
                    response.status(),
                    url
                )
                .into(),
            ));
        }

        let html = response.text().await.map_err(crate::Error::Reqwest)?;

        if html.is_empty() {
            return Err(crate::Error::NoContent);
        }

        log::trace!("Fetched {} bytes of HTML", html.len());

        // Parse the HTML
        let document = Html::parse_document(&html);

        // Extract content based on mode
        let mode = options.mode.unwrap_or_default();
        let mut content = match mode {
            ExtractionMode::Full => self.extract_full(&document, &options)?,
            ExtractionMode::Readability => self.extract_readability(&document, &options)?,
            ExtractionMode::Selective => self.extract_selective(&document, &options)?,
            ExtractionMode::TextOnly => self.extract_text_only(&document, &options)?,
        };

        // Apply max length if specified
        if let Some(max_len) = options.max_length
            && content.len() > max_len
        {
            content.truncate(max_len);
            content.push_str("... [truncated]");
        }

        // Extract metadata if requested
        let metadata = if options.include_metadata.unwrap_or(false) {
            Some(self.extract_metadata(&document, url.as_str()))
        } else {
            None
        };

        // Extract links if requested
        let links = if options.extract_links.unwrap_or(false) {
            Some(self.extract_links(&document, url.as_str()))
        } else {
            None
        };

        // Extract images if requested
        let images = if options.extract_images.unwrap_or(false) {
            Some(self.extract_images(&document, url.as_str()))
        } else {
            None
        };

        // Extract tables if requested
        let tables = if options.extract_tables.unwrap_or(false) {
            Some(self.extract_tables(&document))
        } else {
            None
        };

        // Convert to requested output format
        let output_format = options.output_format.unwrap_or_default();
        let formatted_content = self.format_output(&content, output_format, &document)?;

        log::trace!(
            "Extraction complete: {} characters, format: {:?}",
            formatted_content.len(),
            output_format
        );

        Ok(ScrappingResult {
            content: formatted_content,
            metadata,
            links,
            images,
            tables,
            url: url.to_string(),
            format: output_format,
        })
    }

    /// Extract full HTML content with optional exclusions
    fn extract_full(&self, document: &Html, options: &ScrappingOptions) -> crate::Result<String> {
        let mut html = document.html();

        // Apply exclusions if specified
        if let Some(exclude_selectors) = &options.exclude_selectors {
            for selector_str in exclude_selectors {
                let selector = Selector::parse(selector_str)?;

                for element in document.select(&selector) {
                    html = html.replace(&element.html(), "");
                }
            }
        }

        if options.clean_content.unwrap_or(true) {
            html = WhitespaceFormatter::clean_html(&html);
        }

        Ok(html)
    }

    /// Extract main content using readability algorithm
    fn extract_readability(
        &self,
        document: &Html,
        options: &ScrappingOptions,
    ) -> crate::Result<String> {
        // Try to find main content containers
        let content_selectors = vec![
            "article",
            "main",
            "[role='main']",
            ".article-content",
            ".post-content",
            ".entry-content",
            "#content",
            ".content",
        ];

        let mut best_content = String::new();
        let mut max_text_length = 0;

        for selector_str in content_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                for element in document.select(&selector) {
                    let text = element.text().collect::<String>();
                    if text.len() > max_text_length {
                        max_text_length = text.len();
                        best_content = element.html();
                    }
                }
            }
        }

        // If no main content found, fall back to body
        if best_content.is_empty() {
            if let Ok(body_selector) = Selector::parse("body")
                && let Some(body) = document.select(&body_selector).next()
            {
                best_content = body.html();
            }
        }

        // Remove common noise elements
        let noise_selectors = vec![
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "aside",
            ".advertisement",
            ".ads",
            ".sidebar",
            ".social-share",
            ".comments",
            "#comments",
        ];

        for selector_str in noise_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                for element in Html::parse_fragment(&best_content).select(&selector) {
                    best_content = best_content.replace(&element.html(), "");
                }
            }
        }

        if options.clean_content.unwrap_or(true) {
            best_content = WhitespaceFormatter::clean_html(&best_content);
        }

        Ok(best_content)
    }

    /// Extract content using CSS selectors
    fn extract_selective(
        &self,
        document: &Html,
        options: &ScrappingOptions,
    ) -> crate::Result<String> {
        let selectors = options
            .css_selectors
            .as_ref()
            .ok_or_else(|| crate::Error::Validation {
                message: "css_selectors required for selective mode".to_string(),
            })?;

        let mut extracted = String::new();

        for selector_str in selectors {
            let selector =
                Selector::parse(selector_str).map_err(|e| crate::Error::Scraper(e.to_string()))?;

            for element in document.select(&selector) {
                extracted.push_str(&element.html());
                extracted.push('\n');
            }
        }

        // Apply exclusions
        if let Some(exclude_selectors) = &options.exclude_selectors {
            let fragment = Html::parse_fragment(&extracted);
            for selector_str in exclude_selectors {
                let selector = Selector::parse(selector_str)
                    .map_err(|e| crate::Error::Scraper(e.to_string()))?;

                for element in fragment.select(&selector) {
                    extracted = extracted.replace(&element.html(), "");
                }
            }
        }

        if options.clean_content.unwrap_or(true) {
            extracted = WhitespaceFormatter::clean_html(&extracted);
        }

        Ok(extracted)
    }

    /// Extract only text content, stripping all HTML
    fn extract_text_only(
        &self,
        document: &Html,
        options: &ScrappingOptions,
    ) -> crate::Result<String> {
        let body_selector =
            Selector::parse("body").map_err(|e| crate::Error::Scraper(e.to_string()))?;

        let mut text = String::new();

        if let Some(body) = document.select(&body_selector).next() {
            // Remove script and style tags first
            let mut cleaned_html = body.html();
            for tag in &["script", "style", "noscript"] {
                if let Ok(selector) = Selector::parse(tag) {
                    let temp_doc = Html::parse_fragment(&cleaned_html);
                    for element in temp_doc.select(&selector) {
                        cleaned_html = cleaned_html.replace(&element.html(), "");
                    }
                }
            }

            let cleaned_doc = Html::parse_fragment(&cleaned_html);
            text = cleaned_doc
                .root_element()
                .text()
                .collect::<Vec<_>>()
                .join(" ");
        }

        if options.clean_content.unwrap_or(true) {
            text = WhitespaceFormatter::clean_text(&text);
        }

        Ok(text)
    }

    /// Extract metadata from the page
    fn extract_metadata(&self, document: &Html, url: &str) -> PageMetadata {
        let mut metadata = PageMetadata {
            url: url.to_string(),
            ..Default::default()
        };

        // Extract title
        if let Ok(title_selector) = Selector::parse("title") {
            if let Some(title) = document.select(&title_selector).next() {
                metadata.title = Some(title.text().collect::<String>().trim().to_string());
            }
        }

        // Extract meta tags
        if let Ok(meta_selector) = Selector::parse("meta") {
            for meta in document.select(&meta_selector) {
                if let Some(name) = meta.value().attr("name") {
                    if let Some(content) = meta.value().attr("content") {
                        match name.to_lowercase().as_str() {
                            "description" => metadata.description = Some(content.to_string()),
                            "author" => metadata.author = Some(content.to_string()),
                            "keywords" => metadata.keywords = Some(content.to_string()),
                            _ => {}
                        }
                    }
                }

                // Open Graph tags
                if let Some(property) = meta.value().attr("property") {
                    if let Some(content) = meta.value().attr("content") {
                        match property.to_lowercase().as_str() {
                            "og:title" if metadata.title.is_none() => {
                                metadata.title = Some(content.to_string())
                            }
                            "og:description" if metadata.description.is_none() => {
                                metadata.description = Some(content.to_string())
                            }
                            "og:image" => metadata.image = Some(content.to_string()),
                            _ => {}
                        }
                    }
                }
            }
        }

        // Extract language
        if let Ok(html_selector) = Selector::parse("html") {
            if let Some(html_elem) = document.select(&html_selector).next()
                && let Some(lang) = html_elem.value().attr("lang")
            {
                metadata.language = Some(lang.to_string());
            }
        }

        metadata
    }

    /// Extract all links from the page
    fn extract_links(&self, document: &Html, base_url: &str) -> Vec<LinkInfo> {
        let mut links = Vec::new();
        let base = url::Url::parse(base_url).ok();

        if let Ok(selector) = Selector::parse("a[href]") {
            for element in document.select(&selector) {
                if let Some(href) = element.value().attr("href") {
                    // Resolve relative URLs
                    let absolute_url = if let Some(base) = &base {
                        base.join(href).ok().map(|u| u.to_string())
                    } else {
                        Some(href.to_string())
                    };

                    if let Some(url) = absolute_url {
                        let text = element.text().collect::<String>().trim().to_string();
                        links.push(LinkInfo {
                            url,
                            text: if text.is_empty() { None } else { Some(text) },
                            title: element.value().attr("title").map(String::from),
                        });
                    }
                }
            }
        }

        links
    }

    /// Extract all images from the page
    fn extract_images(&self, document: &Html, base_url: &str) -> Vec<ImageInfo> {
        let mut images = Vec::new();
        let base = url::Url::parse(base_url).ok();

        if let Ok(selector) = Selector::parse("img") {
            for element in document.select(&selector) {
                if let Some(src) = element.value().attr("src") {
                    // Resolve relative URLs
                    let absolute_url = if let Some(base) = &base {
                        base.join(src).ok().map(|u| u.to_string())
                    } else {
                        Some(src.to_string())
                    };

                    if let Some(url) = absolute_url {
                        images.push(ImageInfo {
                            url,
                            alt: element.value().attr("alt").map(String::from),
                            title: element.value().attr("title").map(String::from),
                            width: element.value().attr("width").map(String::from),
                            height: element.value().attr("height").map(String::from),
                        });
                    }
                }
            }
        }

        images
    }

    /// Extract and format tables
    fn extract_tables(&self, document: &Html) -> Vec<TableData> {
        let mut tables = Vec::new();

        if let Ok(table_selector) = Selector::parse("table") {
            for (index, table) in document.select(&table_selector).enumerate() {
                let mut rows = Vec::new();

                if let Ok(row_selector) = Selector::parse("tr") {
                    for row_elem in table.select(&row_selector) {
                        let mut cells = Vec::new();

                        if let Ok(cell_selector) = Selector::parse("th, td") {
                            for cell in row_elem.select(&cell_selector) {
                                cells.push(cell.text().collect::<String>().trim().to_string());
                            }
                        }

                        if !cells.is_empty() {
                            rows.push(cells);
                        }
                    }
                }

                if !rows.is_empty() {
                    tables.push(TableData {
                        index,
                        rows,
                        caption: None,
                    });
                }
            }
        }

        tables
    }

    /// Format content to the requested output format
    fn format_output(
        &self,
        content: &str,
        format: OutputFormat,
        _document: &Html,
    ) -> crate::Result<String> {
        match format {
            OutputFormat::Html => Ok(content.to_string()),
            OutputFormat::Markdown => HtmlFormatter::to_markdown(content),
            OutputFormat::PlainText => {
                let doc = Html::parse_fragment(content);
                let text = doc.root_element().text().collect::<Vec<_>>().join(" ");
                Ok(WhitespaceFormatter::clean_text(&text))
            }
            OutputFormat::Json => Ok(content.to_string()), // Return as-is for JSON wrapper
        }
    }
}

/// Result from a scraping operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrappingResult {
    /// The extracted content
    pub content: String,
    /// Page metadata (if requested)
    pub metadata: Option<PageMetadata>,
    /// Extracted links (if requested)
    pub links: Option<Vec<LinkInfo>>,
    /// Extracted images (if requested)
    pub images: Option<Vec<ImageInfo>>,
    /// Extracted tables (if requested)
    pub tables: Option<Vec<TableData>>,
    /// Original URL
    pub url: String,
    /// Output format used
    pub format: OutputFormat,
}

/// Page metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PageMetadata {
    pub url: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub author: Option<String>,
    pub keywords: Option<String>,
    pub image: Option<String>,
    pub language: Option<String>,
}

/// Link information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkInfo {
    pub url: String,
    pub text: Option<String>,
    pub title: Option<String>,
}

/// Image information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageInfo {
    pub url: String,
    pub alt: Option<String>,
    pub title: Option<String>,
    pub width: Option<String>,
    pub height: Option<String>,
}

/// Table data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableData {
    pub index: usize,
    pub rows: Vec<Vec<String>>,
    pub caption: Option<String>,
}

#[async_trait::async_trait]
impl FnExecutor<ScrappingArgs, ScrappingResult> for ScrappingTool {
    async fn call(&self, args: ScrappingArgs) -> crate::Result<ScrappingResult> {
        self.scrape(args).await
    }
}

impl FnDeclarator<ScrappingArgs, ScrappingResult> for ScrappingTool {
    fn declare(&self) -> FunctionDeclaration<ScrappingArgs, ScrappingResult> {
        FunctionDeclaration {
            name: "scrape_web",
            description: "Advanced web scraping tool that extracts content from URLs with multiple modes: 'full' (entire page), 'readability' (main article content - best for articles/blogs), 'selective' (specific CSS selectors), or 'text_only' (plain text). Supports multiple output formats (HTML, Markdown, PlainText, JSON) and can extract metadata, links, images, and tables. Use 'readability' mode for best LLM-friendly results when scraping articles or documentation.",
            parameters: schemars::schema_for!(ScrappingArgs),
            executor: std::sync::Arc::new(self.clone()),
        }
    }
}
