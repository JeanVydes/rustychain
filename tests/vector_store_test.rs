//! Integration tests for VectorStore
//!
//! These tests require a PostgreSQL instance with pgvector extension.
//! Tests use testcontainers to spin up an ephemeral database.

mod common;

use common::{
    TEST_DIMENSIONS, get_test_db_url, normalized_embedding, random_embedding, similar_embeddings,
    unique_table_name,
};
use pretty_assertions::assert_eq;
use rustychain::{
    persistent::pgvector::DocumentInput,
    storage::persistent::pgvector::{SearchOptions, VectorStore},
};
use serde_json::json;

/// Test basic VectorStore creation and table initialization
#[tokio::test]
async fn test_vector_store_creation() {
    let db_url = get_test_db_url().await;
    let table_name = unique_table_name();

    let store = VectorStore::new(db_url, Some(table_name.clone()), Some(TEST_DIMENSIONS))
        .await
        .expect("Failed to create VectorStore");

    // Verify we can get count (table exists)
    let count = store.count(None).await.expect("Failed to count");
    assert_eq!(count, 0);
}

/// Test adding a single document
#[tokio::test]
async fn test_add_document() {
    let db_url = get_test_db_url().await;
    let table_name = unique_table_name();

    let store = VectorStore::new(db_url, Some(table_name), Some(TEST_DIMENSIONS))
        .await
        .unwrap();

    let embedding = random_embedding(TEST_DIMENSIONS);
    let id = store
        .add_document(
            "Test document content".to_string(),
            embedding,
            Some("test_collection".to_string()),
            Some(json!({"source": "test", "page": 1})),
        )
        .await
        .expect("Failed to add document");

    assert!(id > 0);

    let count = store.count(None).await.unwrap();
    assert_eq!(count, 1);
}

/// Test adding multiple documents in batch
#[tokio::test]
async fn test_add_documents_batch() {
    let db_url = get_test_db_url().await;
    let table_name = unique_table_name();

    let store = VectorStore::new(db_url, Some(table_name), Some(TEST_DIMENSIONS))
        .await
        .unwrap();

    let documents: Vec<DocumentInput> = (0..10)
        .map(|i| {
            DocumentInput(
                format!("Document {}", i),
                random_embedding(TEST_DIMENSIONS),
                Some("batch_test".to_string()),
                Some(json!({"index": i})),
            )
        })
        .collect();

    let ids = store
        .add_documents_batch(documents)
        .await
        .expect("Failed to add batch");

    assert_eq!(ids.len(), 10);

    let count = store.count(None).await.unwrap();
    assert_eq!(count, 10);
}

/// Test simple similarity search
#[tokio::test]
async fn test_similarity_search() {
    let db_url = get_test_db_url().await;
    let table_name = unique_table_name();

    let store = VectorStore::new(db_url, Some(table_name), Some(TEST_DIMENSIONS))
        .await
        .unwrap();

    // Create a base embedding and similar variations
    let base = normalized_embedding(TEST_DIMENSIONS);
    let similar = similar_embeddings(&base, 5, 0.05);

    // Add the similar documents
    for (i, emb) in similar.iter().enumerate() {
        store
            .add_document(format!("Similar document {}", i), emb.clone(), None, None)
            .await
            .unwrap();
    }

    // Add some random (dissimilar) documents
    for i in 0..5 {
        store
            .add_document(
                format!("Random document {}", i),
                random_embedding(TEST_DIMENSIONS),
                None,
                None,
            )
            .await
            .unwrap();
    }

    // Search with the base embedding - should find similar ones first
    let results = store.similarity_search(base.clone(), 5).await.unwrap();

    // similarity_search now returns Vec<SearchResult>
    assert!(!results.is_empty(), "Results should not be empty");

    // Count how many results are "Similar document"
    let similar_count = results
        .iter()
        .filter(|r| r.document.text.contains("Similar document"))
        .count();

    assert!(
        similar_count >= 3,
        "Expected at least 3 similar documents in top 5 results, got {}",
        similar_count
    );

    // Verify results have distance/score populated
    for result in &results {
        assert!(result.distance >= 0.0, "Distance should be non-negative");
        assert!(
            result.score >= 0.0 && result.score <= 1.0,
            "Score should be between 0 and 1"
        );
    }
}

/// Test advanced search with options
#[tokio::test]
async fn test_search_with_options() {
    let db_url = get_test_db_url().await;
    let table_name = unique_table_name();

    let store = VectorStore::new(db_url, Some(table_name), Some(TEST_DIMENSIONS))
        .await
        .unwrap();

    // Add documents to different collections
    let embedding = normalized_embedding(TEST_DIMENSIONS);

    store
        .add_document(
            "Collection A doc 1".to_string(),
            embedding.clone(),
            Some("collection_a".to_string()),
            Some(json!({"type": "important"})),
        )
        .await
        .unwrap();

    store
        .add_document(
            "Collection A doc 2".to_string(),
            similar_embeddings(&embedding, 1, 0.01)[0].clone(),
            Some("collection_a".to_string()),
            Some(json!({"type": "normal"})),
        )
        .await
        .unwrap();

    store
        .add_document(
            "Collection B doc 1".to_string(),
            similar_embeddings(&embedding, 1, 0.01)[0].clone(),
            Some("collection_b".to_string()),
            Some(json!({"type": "important"})),
        )
        .await
        .unwrap();

    // Search with collection filter
    let options = SearchOptions::new(10)
        .with_collection("collection_a")
        .include_distances();

    let results = store.search(embedding.clone(), options).await.unwrap();

    assert_eq!(results.len(), 2);
    for result in &results {
        assert_eq!(result.document.collection, Some("collection_a".to_string()));
    }
}

/// Test search with metadata filter
#[tokio::test]
async fn test_search_with_metadata_filter() {
    let db_url = get_test_db_url().await;
    let table_name = unique_table_name();

    let store = VectorStore::new(db_url, Some(table_name), Some(TEST_DIMENSIONS))
        .await
        .unwrap();

    let embedding = normalized_embedding(TEST_DIMENSIONS);

    // Add documents with different metadata
    for i in 0..5 {
        store
            .add_document(
                format!("Important doc {}", i),
                similar_embeddings(&embedding, 1, 0.01)[0].clone(),
                None,
                Some(json!({"priority": "high", "index": i})),
            )
            .await
            .unwrap();
    }

    for i in 0..5 {
        store
            .add_document(
                format!("Normal doc {}", i),
                similar_embeddings(&embedding, 1, 0.01)[0].clone(),
                None,
                Some(json!({"priority": "low", "index": i})),
            )
            .await
            .unwrap();
    }

    // Search with metadata filter
    let options = SearchOptions::new(10).with_metadata_filter(json!({"priority": "high"}));

    let results = store.search(embedding, options).await.unwrap();

    assert_eq!(results.len(), 5);
    for result in &results {
        let metadata = result.document.metadata.as_ref().unwrap();
        assert_eq!(metadata["priority"], "high");
    }
}

/// Test document deletion by ID
#[tokio::test]
async fn test_delete_document() {
    let db_url = get_test_db_url().await;
    let table_name = unique_table_name();

    let store = VectorStore::new(db_url, Some(table_name), Some(TEST_DIMENSIONS))
        .await
        .unwrap();

    let id = store
        .add_document(
            "To be deleted".to_string(),
            random_embedding(TEST_DIMENSIONS),
            None,
            None,
        )
        .await
        .unwrap();

    assert_eq!(store.count(None).await.unwrap(), 1);

    store.delete(id).await.expect("Failed to delete");

    assert_eq!(store.count(None).await.unwrap(), 0);
}

/// Test delete by metadata
#[tokio::test]
async fn test_delete_by_metadata() {
    let db_url = get_test_db_url().await;
    let table_name = unique_table_name();

    let store = VectorStore::new(db_url, Some(table_name), Some(TEST_DIMENSIONS))
        .await
        .unwrap();

    // Add documents with different sources
    for i in 0..5 {
        store
            .add_document(
                format!("Doc {}", i),
                random_embedding(TEST_DIMENSIONS),
                None,
                Some(json!({"source": "to_delete"})),
            )
            .await
            .unwrap();
    }

    for i in 0..3 {
        store
            .add_document(
                format!("Keep doc {}", i),
                random_embedding(TEST_DIMENSIONS),
                None,
                Some(json!({"source": "keep"})),
            )
            .await
            .unwrap();
    }

    assert_eq!(store.count(None).await.unwrap(), 8);

    let deleted = store
        .delete_by_metadata(&json!({"source": "to_delete"}), None)
        .await
        .unwrap();

    assert_eq!(deleted, 5);
    assert_eq!(store.count(None).await.unwrap(), 3);
}

/// Test collection management
#[tokio::test]
async fn test_collections() {
    let db_url = get_test_db_url().await;
    let table_name = unique_table_name();

    let store = VectorStore::new(db_url, Some(table_name), Some(TEST_DIMENSIONS))
        .await
        .unwrap();

    // Add documents to different collections
    store
        .add_document(
            "Doc A1".to_string(),
            random_embedding(TEST_DIMENSIONS),
            Some("collection_a".to_string()),
            None,
        )
        .await
        .unwrap();

    store
        .add_document(
            "Doc A2".to_string(),
            random_embedding(TEST_DIMENSIONS),
            Some("collection_a".to_string()),
            None,
        )
        .await
        .unwrap();

    store
        .add_document(
            "Doc B1".to_string(),
            random_embedding(TEST_DIMENSIONS),
            Some("collection_b".to_string()),
            None,
        )
        .await
        .unwrap();

    // List collections
    let collections = store.list_collections().await.unwrap();
    assert_eq!(collections.len(), 2);
    assert!(collections.contains(&"collection_a".to_string()));
    assert!(collections.contains(&"collection_b".to_string()));

    // Count by collection
    assert_eq!(store.count(Some("collection_a")).await.unwrap(), 2);
    assert_eq!(store.count(Some("collection_b")).await.unwrap(), 1);

    // Delete collection
    store.delete_collection("collection_a").await.unwrap();
    assert_eq!(store.count(None).await.unwrap(), 1);
}

/// Test document listing with pagination
#[tokio::test]
async fn test_list_pagination() {
    let db_url = get_test_db_url().await;
    let table_name = unique_table_name();

    let store = VectorStore::new(db_url, Some(table_name), Some(TEST_DIMENSIONS))
        .await
        .unwrap();

    // Add 25 documents
    for i in 0..25 {
        store
            .add_document(
                format!("Document {}", i),
                random_embedding(TEST_DIMENSIONS),
                None,
                None,
            )
            .await
            .unwrap();
    }

    // Get first page
    let page1 = store.list(10, 0, None).await.unwrap();
    assert_eq!(page1.len(), 10);

    // Get second page
    let page2 = store.list(10, 10, None).await.unwrap();
    assert_eq!(page2.len(), 10);

    // Get third page (partial)
    let page3 = store.list(10, 20, None).await.unwrap();
    assert_eq!(page3.len(), 5);

    // Verify no overlap
    let page1_ids: Vec<i32> = page1.iter().map(|d| d.id).collect();
    let page2_ids: Vec<i32> = page2.iter().map(|d| d.id).collect();
    for id in &page1_ids {
        assert!(!page2_ids.contains(id));
    }
}

/// Test table statistics
#[tokio::test]
async fn test_stats() {
    let db_url = get_test_db_url().await;
    let table_name = unique_table_name();

    let store = VectorStore::new(db_url, Some(table_name), Some(TEST_DIMENSIONS))
        .await
        .unwrap();

    // Add some documents
    for i in 0..10 {
        store
            .add_document(
                format!("Stats test doc {}", i),
                random_embedding(TEST_DIMENSIONS),
                None,
                None,
            )
            .await
            .unwrap();
    }

    let stats = store.stats().await.unwrap();

    assert_eq!(stats.row_count, 10);
    assert!(stats.total_size_bytes > 0);

    // Test human-readable formatting
    let human_size = stats.total_size_human();
    assert!(human_size.contains("KB") || human_size.contains("MB") || human_size.contains("B"));
}

/// Test clear table
#[tokio::test]
async fn test_clear() {
    let db_url = get_test_db_url().await;
    let table_name = unique_table_name();

    let store = VectorStore::new(db_url, Some(table_name), Some(TEST_DIMENSIONS))
        .await
        .unwrap();

    // Add documents
    for i in 0..10 {
        store
            .add_document(
                format!("Clear test doc {}", i),
                random_embedding(TEST_DIMENSIONS),
                None,
                None,
            )
            .await
            .unwrap();
    }

    assert_eq!(store.count(None).await.unwrap(), 10);

    store.clear().await.unwrap();

    assert_eq!(store.count(None).await.unwrap(), 0);
}

/// Test reindex
#[tokio::test]
async fn test_reindex() {
    let db_url = get_test_db_url().await;
    let table_name = unique_table_name();

    let store = VectorStore::new(db_url, Some(table_name), Some(TEST_DIMENSIONS))
        .await
        .unwrap();

    // Add some documents
    for i in 0..5 {
        store
            .add_document(
                format!("Reindex test doc {}", i),
                random_embedding(TEST_DIMENSIONS),
                None,
                None,
            )
            .await
            .unwrap();
    }

    // Reindex should complete without error
    store.reindex().await.expect("Reindex failed");

    // Verify search still works after reindex
    let results = store
        .similarity_search(random_embedding(TEST_DIMENSIONS), 5)
        .await
        .unwrap();

    assert!(!results.is_empty());
}
