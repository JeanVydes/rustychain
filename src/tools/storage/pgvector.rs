//! pgvector
//!
//! This module provides tools for integrating PostgreSQL with pgvector
//! as a vector store.

use crate::llm::LLM;
use crate::llm::function::{FnDeclarator, FnExecutor, ToolArgs};
use crate::persistent::pgvector::{DocumentInput, SearchOptions};
use crate::prelude::*;
use crate::storage::persistent::pgvector::VectorStore;
use crate::{FunctionDeclaration, RecursiveCharacterTextSplitter, TextSplitter};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
pub struct SimpleRetrievalArgs {
    #[schemars(description = "The query string to search for in the vector database.")]
    pub query: String,
}

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
pub struct ComplexRetrievalArgs {
    #[schemars(description = "The query string to search for in the vector database.")]
    pub query: String,
    #[schemars(description = "Advanced search configuration options.")]
    pub config: SearchOptions,
}

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
pub struct AugmentedArgs {
    #[schemars(description = "The text content to be embedded and stored in the vector database.")]
    pub text: String,
    #[schemars(
        description = "Optional metadata to associate with the document, as a JSON object."
    )]
    pub metadata: Option<serde_json::Value>,
    #[schemars(
        description = "The name of the collection to add the document to. It is recommended to use different collections for granular data. Example: all data related to invoices of January 2024 Q1 can go into 'invoices_jan_2024_q1' collection."
    )]
    pub collection: Option<String>,
}

impl ToolArgs for SimpleRetrievalArgs {}
impl ToolArgs for ComplexRetrievalArgs {}
impl ToolArgs for AugmentedArgs {}

#[derive(Clone)]
pub struct PgVectorRetrievalTool {
    pub store: Arc<Mutex<VectorStore>>,
    pub llm: Arc<LLM>,
    pub table_name: Option<String>,
}

#[derive(Clone)]
pub struct PgVectorAugmentedTool {
    pub store: Arc<Mutex<VectorStore>>,
    pub llm: Arc<LLM>,
    pub table_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    pub query: String,
    pub results: Vec<(String, Option<serde_json::Value>)>,
}

#[async_trait::async_trait]
impl FnExecutor<SimpleRetrievalArgs, RetrievalResult> for PgVectorRetrievalTool {
    async fn call(&self, args: SimpleRetrievalArgs) -> crate::Result<RetrievalResult> {
        log::debug!("RetrievalTool called with args: {:?}", args);
        let query = self.llm.embedding(&args.query, 1536).await?;
        let store = self.store.lock().await;
        let results = store.similarity_search(query, 10).await?;

        Ok(RetrievalResult {
            query: args.query,
            results: results
                .into_iter()
                .map(|r| (r.document.text, r.document.metadata))
                .collect(),
        })
    }
}

#[async_trait::async_trait]
impl FnExecutor<ComplexRetrievalArgs, RetrievalResult> for PgVectorRetrievalTool {
    async fn call(&self, args: ComplexRetrievalArgs) -> crate::Result<RetrievalResult> {
        log::debug!("RetrievalTool called with args: {:?}", args);
        log::debug!("Generating embedding for query: {}", args.query);
        let query = self.llm.embedding(&args.query, 1536).await?;
        let store = self.store.lock().await;
        let context = store.search(query, args.config).await?;

        Ok(RetrievalResult {
            query: args.query,
            results: context
                .into_iter()
                .map(|r| (r.document.text, r.document.metadata))
                .collect(),
        })
    }
}

#[async_trait::async_trait]
impl FnExecutor<AugmentedArgs, serde_json::Value> for PgVectorAugmentedTool {
    async fn call(&self, args: AugmentedArgs) -> crate::Result<serde_json::Value> {
        log::debug!("Generating embedding for query: {}", args.text);

        let splitter = RecursiveCharacterTextSplitter::new(crate::SplitterConfig {
            chunk_size: 1024,
            chunk_overlap: 256,
            trim_chunks: true,
            track_indices: true,
            min_chunk_size: None,
        });

        let chunks = splitter.split(&args.text);

        let mut embeddings: HashMap<String, Vec<f32>> = HashMap::new();

        for chunk in chunks {
            let embedding = self.llm.embedding(&chunk.content, 1536).await?;
            embeddings.insert(chunk.content.clone(), embedding);
        }

        let store = self.store.lock().await;
        let documents = embeddings
            .into_iter()
            .map(|(text, embedding)| {
                DocumentInput(
                    text,
                    embedding,
                    Some("_default".to_string()),
                    Some(serde_json::json!({})),
                )
            })
            .collect::<Vec<_>>();

        store.add_documents_batch(documents).await?;

        log::debug!("Document added successfully");

        Ok(serde_json::json!({
            "status": "Document added successfully"
        }))
    }
}

impl FnDeclarator<SimpleRetrievalArgs, RetrievalResult> for PgVectorRetrievalTool {
    fn declare(&self) -> FunctionDeclaration<SimpleRetrievalArgs, RetrievalResult> {
        FunctionDeclaration {
            name: "retrieval_tool",
            description: "Use this tool to perform a similarity search in the vector store and retrieve relevant documents based on the user's query.",
            parameters: schema_for!(SimpleRetrievalArgs),
            executor: Arc::new(self.clone()),
        }
    }
}

impl FnDeclarator<ComplexRetrievalArgs, RetrievalResult> for PgVectorRetrievalTool {
    fn declare(&self) -> FunctionDeclaration<ComplexRetrievalArgs, RetrievalResult> {
        FunctionDeclaration {
            name: "complex_retrieval_tool",
            description: "Use this tool to perform a similarity search in the vector store with advanced search options and retrieve relevant documents based on the user's query.",
            parameters: schema_for!(ComplexRetrievalArgs),
            executor: Arc::new(PgVectorRetrievalTool {
                store: self.store.clone(),
                llm: self.llm.clone(),
                table_name: self.table_name.clone(),
            }),
        }
    }
}

impl FnDeclarator<AugmentedArgs, serde_json::Value> for PgVectorAugmentedTool {
    fn declare(&self) -> FunctionDeclaration<AugmentedArgs, serde_json::Value> {
        FunctionDeclaration {
            name: "augmented_tool",
            description: "Use this tool to add a new document to the vector store with its corresponding embedding.",
            parameters: schema_for!(AugmentedArgs),
            executor: Arc::new(PgVectorAugmentedTool {
                store: self.store.clone(),
                llm: self.llm.clone(),
                table_name: self.table_name.clone(),
            }),
        }
    }
}
