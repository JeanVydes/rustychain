use pgvector::Vector;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::{FromRow, Pool, Postgres, Row};

use crate::CoreError;

pub const DEFAULT_TABLE_NAME: &str = "rag_documents";
pub const DEFAULT_DIMENSIONS: i32 = 1536;

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct Document {
    pub id: i32,
    pub text: String,
    #[sqlx(default)]
    pub collection: Option<String>,
    #[sqlx(default)]
    pub metadata: Option<Value>,
    #[serde(skip)]
    #[sqlx(skip)]
    pub embedding: Option<Vector>,
}

#[derive(Debug, Clone)]
pub struct DocumentInput(
    pub String,
    pub Vec<f32>,
    pub Option<String>,
    pub Option<Value>,
);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub document: Document,
    pub distance: f32,
    pub score: f32, // 1.0 - distance for cosine
}

#[derive(JsonSchema, Serialize, Deserialize, Debug, Clone, Default)]
pub struct SearchOptions {
    #[schemars(description = "Number of nearest neighbors to retrieve")]
    pub k: i32,
    #[schemars(
        description = "Distance threshold for filtering results",
        range(min = 0.0, max = 2.0)
    )]
    pub threshold: Option<f32>,
    #[schemars(
        description = "The collection name to filter documents. If set, only documents from this collection will be considered."
    )]
    pub collection: Option<String>,
    #[schemars(
        description = "Metadata filter as a JSON object to filter documents. The filter should be a JSON object where keys are metadata fields and values are the expected values."
    )]
    pub metadata_filter: Option<Value>,
    #[schemars(
        description = "Whether to include embeddings in the results. Embeddings can be large, so enable only if needed."
    )]
    pub include_embeddings: bool,
    #[schemars(description = "Whether to include distances in the results.")]
    pub include_distances: bool,
}

impl SearchOptions {
    pub fn new(k: i32) -> Self {
        Self {
            k,
            ..Default::default()
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }

    pub fn with_collection(mut self, collection: impl Into<String>) -> Self {
        self.collection = Some(collection.into());
        self
    }

    pub fn with_metadata_filter(mut self, filter: Value) -> Self {
        self.metadata_filter = Some(filter);
        self
    }

    pub fn include_embeddings(mut self) -> Self {
        self.include_embeddings = true;
        self
    }

    pub fn include_distances(mut self) -> Self {
        self.include_distances = true;
        self
    }
}

#[derive(Clone)]
pub struct VectorStore {
    pub pool: Pool<Postgres>,
    pub table_name: Option<String>,
    pub dimensions: Option<i32>,
}

impl VectorStore {
    pub async fn new(
        db_url: &str,
        table_name: Option<String>,
        dimensions: Option<i32>,
    ) -> crate::Result<Self> {
        let pool = Pool::<Postgres>::connect(db_url).await?;
        let store = VectorStore {
            pool,
            table_name,
            dimensions,
        };
        store.ensure_table_exists().await?;
        Ok(store)
    }

    fn table(&self) -> &str {
        self.table_name.as_deref().unwrap_or(DEFAULT_TABLE_NAME)
    }

    fn dims(&self) -> i32 {
        self.dimensions.unwrap_or(DEFAULT_DIMENSIONS)
    }

    /// Ensure the vector table exists with proper schema
    async fn ensure_table_exists(&self) -> crate::Result<()> {
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(&self.pool)
            .await?;

        let query = format!(
            r#"
            CREATE TABLE IF NOT EXISTS {} (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                embedding vector({}) NOT NULL,
                collection VARCHAR(255),
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            "#,
            self.table(),
            self.dims()
        );
        sqlx::query(&query).execute(&self.pool).await?;

        // Index for similarity search
        let embedding_idx = format!(
            r#"
            CREATE INDEX IF NOT EXISTS {}_embedding_idx 
            ON {} 
            USING hnsw (embedding vector_cosine_ops)
            "#,
            self.table(),
            self.table()
        );
        sqlx::query(&embedding_idx).execute(&self.pool).await?;

        // Index for collection filtering
        let collection_idx = format!(
            "CREATE INDEX IF NOT EXISTS {}_collection_idx ON {} (collection)",
            self.table(),
            self.table()
        );
        sqlx::query(&collection_idx).execute(&self.pool).await?;

        // GIN index for metadata JSONB
        let metadata_idx = format!(
            "CREATE INDEX IF NOT EXISTS {}_metadata_idx ON {} USING GIN (metadata)",
            self.table(),
            self.table()
        );
        sqlx::query(&metadata_idx).execute(&self.pool).await?;

        Ok(())
    }

    /// Add a document with optional collection and metadata
    pub async fn add_document(
        &self,
        text: String,
        embedding: Vec<f32>,
        collection: Option<String>,
        metadata: Option<Value>,
    ) -> crate::Result<i32> {
        let vector = Vector::from(embedding);
        let query = format!(
            "INSERT INTO {} (text, embedding, collection, metadata) VALUES ($1, $2, $3, $4) RETURNING id",
            self.table()
        );

        let row = sqlx::query(&query)
            .bind(&text)
            .bind(&vector)
            .bind(&collection)
            .bind(&metadata)
            .fetch_one(&self.pool)
            .await?;

        Ok(row.get("id"))
    }

    /// Add multiple documents in a batch
    pub async fn add_documents_batch(
        &self,
        documents: Vec<DocumentInput>,
    ) -> crate::Result<Vec<i32>> {
        let mut ids = Vec::new();

        for d in documents {
            let id = self.add_document(d.0, d.1, d.2, d.3).await?;
            ids.push(id);
        }

        Ok(ids)
    }

    /// Advanced similarity search with options
    pub async fn search(
        &self,
        query_embedding: Vec<f32>,
        options: SearchOptions,
    ) -> crate::Result<Vec<SearchResult>> {
        let query_vector = Vector::from(query_embedding);

        let embedding_select = if options.include_embeddings {
            ", embedding"
        } else {
            ""
        };

        let distance_select = if options.include_distances {
            ", embedding <=> $1 as distance"
        } else {
            ""
        };

        // Build WHERE clauses
        let mut where_clauses = Vec::new();
        let mut param_idx = 2; // $1 is query_vector

        if options.collection.is_some() {
            where_clauses.push(format!("collection = ${}", param_idx));
            param_idx += 1;
        }

        if options.threshold.is_some() {
            where_clauses.push(format!("embedding <=> $1 < ${}", param_idx));
            param_idx += 1;
        }

        if let Some(ref filter) = options.metadata_filter {
            // Support simple key-value filters like {"key": "value"}
            for (key, value) in filter.as_object().unwrap_or(&serde_json::Map::new()) {
                where_clauses.push(format!("metadata->>'{}' = ${}", key, param_idx));
                param_idx += 1;
                let _ = value; // We'll bind these in order
            }
        }

        let where_sql = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };

        let query = format!(
            r#"
            SELECT id, text, collection, metadata {} {}
            FROM {}
            {}
            ORDER BY embedding <=> $1
            LIMIT ${}
            "#,
            embedding_select,
            distance_select,
            self.table(),
            where_sql,
            param_idx
        );

        let mut q = sqlx::query(&query).bind(&query_vector);

        if let Some(ref collection) = options.collection {
            q = q.bind(collection);
        }

        if let Some(threshold) = options.threshold {
            q = q.bind(threshold);
        }

        if let Some(ref filter) = options.metadata_filter {
            for (_key, value) in filter.as_object().unwrap_or(&serde_json::Map::new()) {
                if let Some(s) = value.as_str() {
                    q = q.bind(s.to_string());
                }
            }
        }

        q = q.bind(options.k);

        let rows = q.fetch_all(&self.pool).await?;

        let results = rows
            .into_iter()
            .map(|row| {
                let distance: f32 = if options.include_distances {
                    row.try_get("distance").unwrap_or(0.0)
                } else {
                    0.0
                };

                let embedding = if options.include_embeddings {
                    row.try_get::<Vector, _>("embedding").ok()
                } else {
                    None
                };

                SearchResult {
                    document: Document {
                        id: row.get("id"),
                        text: row.get("text"),
                        collection: row.get("collection"),
                        metadata: row.get("metadata"),
                        embedding,
                    },
                    distance,
                    score: 1.0 - distance,
                }
            })
            .collect();

        Ok(results)
    }

    /// Simple similarity search returning top k results
    ///
    /// This is a convenience wrapper around `search()` with default options.
    ///
    /// # Example
    /// ```ignore
    /// let results = store.similarity_search(embedding, 10).await?;
    /// for result in results {
    ///     println!("Score: {:.3} - {}", result.score, result.document.text);
    /// }
    /// ```
    pub async fn similarity_search(
        &self,
        query_embedding: Vec<f32>,
        k: i32,
    ) -> crate::Result<Vec<SearchResult>> {
        self.search(query_embedding, SearchOptions::new(k).include_distances())
            .await
    }

    /// Simple similarity search returning results as concatenated text
    ///
    /// Useful for RAG contexts where you need a single string of relevant documents.
    ///
    /// # Example
    /// ```ignore
    /// let context = store.similarity_search_text(embedding, 5).await?;
    /// let prompt = format!("Context:\n{}\n\nQuestion: {}", context, question);
    /// ```
    pub async fn similarity_search_text(
        &self,
        query_embedding: Vec<f32>,
        k: i32,
    ) -> crate::Result<String> {
        let results = self.search(query_embedding, SearchOptions::new(k)).await?;

        if results.is_empty() {
            return Err(Box::new(CoreError::NotFound(
                "No relevant documents found.".to_string(),
            )));
        }

        let context = results
            .into_iter()
            .map(|r| r.document.text)
            .collect::<Vec<_>>()
            .join("\n---\n");

        Ok(context)
    }

    /// Delete a document by ID
    pub async fn delete(&self, id: i32) -> crate::Result<bool> {
        let query = format!("DELETE FROM {} WHERE id = $1", self.table());
        let result = sqlx::query(&query).bind(id).execute(&self.pool).await?;

        Ok(result.rows_affected() > 0)
    }

    /// Delete all documents in a collection
    pub async fn delete_collection(&self, collection: &str) -> crate::Result<u64> {
        let query = format!("DELETE FROM {} WHERE collection = $1", self.table());
        let result = sqlx::query(&query)
            .bind(collection)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected())
    }

    /// Clear all documents (truncate table)
    pub async fn clear(&self) -> crate::Result<()> {
        let query = format!("TRUNCATE TABLE {} RESTART IDENTITY", self.table());
        sqlx::query(&query).execute(&self.pool).await?;
        Ok(())
    }

    /// Get document count
    pub async fn count(&self, collection: Option<&str>) -> crate::Result<i64> {
        let query = match collection {
            Some(_) => format!(
                "SELECT COUNT(*) as count FROM {} WHERE collection = $1",
                self.table()
            ),
            None => format!("SELECT COUNT(*) as count FROM {}", self.table()),
        };

        let row = match collection {
            Some(c) => sqlx::query(&query).bind(c).fetch_one(&self.pool).await?,
            None => sqlx::query(&query).fetch_one(&self.pool).await?,
        };

        Ok(row.get("count"))
    }

    /// List all collections
    pub async fn list_collections(&self) -> crate::Result<Vec<String>> {
        let query = format!(
            "SELECT DISTINCT collection FROM {} WHERE collection IS NOT NULL",
            self.table()
        );

        let rows = sqlx::query(&query).fetch_all(&self.pool).await?;

        let collections = rows
            .into_iter()
            .filter_map(|row| row.try_get::<String, _>("collection").ok())
            .collect();

        Ok(collections)
    }

    /// Rebuild the HNSW index for the table
    pub async fn reindex(&self) -> crate::Result<()> {
        let index_name = format!("{}_embedding_idx", self.table());

        // Drop existing index and recreate
        let drop_query = format!("DROP INDEX IF EXISTS {}", index_name);
        sqlx::query(&drop_query).execute(&self.pool).await?;

        let create_query = format!(
            "CREATE INDEX {} ON {} USING hnsw (embedding vector_cosine_ops)",
            index_name,
            self.table()
        );
        sqlx::query(&create_query).execute(&self.pool).await?;

        Ok(())
    }

    /// Get table statistics
    pub async fn stats(&self) -> crate::Result<TableStats> {
        let query = format!(
            r#"
            SELECT 
                pg_total_relation_size('{}') as total_size,
                pg_indexes_size('{}') as index_size,
                (SELECT count(*) FROM {}) as row_count
            "#,
            self.table(),
            self.table(),
            self.table()
        );

        let row = sqlx::query(&query).fetch_one(&self.pool).await?;

        Ok(TableStats {
            total_size_bytes: row.try_get::<i64, _>("total_size").unwrap_or(0),
            index_size_bytes: row.try_get::<i64, _>("index_size").unwrap_or(0),
            row_count: row.try_get::<i64, _>("row_count").unwrap_or(0),
        })
    }

    /// Delete documents matching a metadata filter
    pub async fn delete_by_metadata(
        &self,
        filter: &Value,
        collection: Option<&str>,
    ) -> crate::Result<u64> {
        let query = match collection {
            Some(_) => format!(
                "DELETE FROM {} WHERE metadata @> $1 AND collection = $2",
                self.table()
            ),
            None => format!("DELETE FROM {} WHERE metadata @> $1", self.table()),
        };

        let result = match collection {
            Some(coll) => {
                sqlx::query(&query)
                    .bind(filter)
                    .bind(coll)
                    .execute(&self.pool)
                    .await?
            }
            None => sqlx::query(&query).bind(filter).execute(&self.pool).await?,
        };

        Ok(result.rows_affected())
    }

    /// List documents with pagination
    pub async fn list(
        &self,
        limit: i32,
        offset: i32,
        collection: Option<&str>,
    ) -> crate::Result<Vec<Document>> {
        let query = match collection {
            Some(_) => format!(
                "SELECT id, text, collection, metadata FROM {} WHERE collection = $1 ORDER BY id LIMIT $2 OFFSET $3",
                self.table()
            ),
            None => format!(
                "SELECT id, text, collection, metadata FROM {} ORDER BY id LIMIT $1 OFFSET $2",
                self.table()
            ),
        };

        let rows: Vec<Document> = match collection {
            Some(coll) => {
                sqlx::query_as(&query)
                    .bind(coll)
                    .bind(limit)
                    .bind(offset)
                    .fetch_all(&self.pool)
                    .await?
            }
            None => {
                sqlx::query_as(&query)
                    .bind(limit)
                    .bind(offset)
                    .fetch_all(&self.pool)
                    .await?
            }
        };

        Ok(rows)
    }
}

/// Table statistics
#[derive(Debug, Clone)]
pub struct TableStats {
    pub total_size_bytes: i64,
    pub index_size_bytes: i64,
    pub row_count: i64,
}

impl TableStats {
    /// Get human-readable total size
    pub fn total_size_human(&self) -> String {
        format_bytes(self.total_size_bytes)
    }

    /// Get human-readable index size
    pub fn index_size_human(&self) -> String {
        format_bytes(self.index_size_bytes)
    }
}

fn format_bytes(bytes: i64) -> String {
    const KB: i64 = 1024;
    const MB: i64 = KB * 1024;
    const GB: i64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
