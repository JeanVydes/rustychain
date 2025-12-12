//! Benchmarks for VectorStore operations
//!
//! Run with: cargo bench
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use serde_json::Value;
use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;
use testcontainers::{
    core::{WaitFor, ContainerPort}, 
    runners::AsyncRunner, 
    GenericImage, 
    ImageExt,
    ContainerAsync
};
use rustychain::storage::persistent::pgvector::{VectorStore, DEFAULT_DIMENSIONS};

fn vector_store_benchmarks(c: &mut Criterion) {
    // Build a Tokio runtime to run async operations
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build runtime");
    
    // Start container and setup store inside the runtime
    let (db_url, _container) = rt.block_on(async {
        // Use pgvector/pgvector image which has the vector extension pre-installed
        let postgres_image = GenericImage::new("pgvector/pgvector", "pg16")
            .with_exposed_port(ContainerPort::Tcp(5432))
            .with_wait_for(WaitFor::message_on_stderr(
                "database system is ready to accept connections"
            ))
            .with_env_var("POSTGRES_PASSWORD", "postgres")
            .with_env_var("POSTGRES_DB", "postgres")
            .with_env_var("POSTGRES_USER", "postgres");
        
        let container: ContainerAsync<GenericImage> = postgres_image
            .start()
            .await
            .expect("Failed to start postgres container");
        
        // Get mapped port
        let host_port = container
            .get_host_port_ipv4(5432)
            .await
            .expect("Failed to get host port");
        
        let db_url = format!("postgresql://postgres:postgres@127.0.0.1:{}/postgres", host_port);
        
        // Wait a bit for postgres to be fully ready
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        
        (db_url, container)
    });
    
    // Create the store
    let store = rt.block_on(async { 
        VectorStore::new(&db_url, None, Some(DEFAULT_DIMENSIONS))
            .await
            .expect("Failed to create store") 
    });
    
    // Small bench: 100 documents, deterministic embeddings
    let n_small = 100usize;
    let dim = DEFAULT_DIMENSIONS as usize;
    let mut rng = StdRng::seed_from_u64(42);
    
    let docs_small: Vec<(String, Vec<f32>, Option<String>, Option<Value>)> = (0..n_small)
        .map(|i| {
            let mut emb = vec![0.0f32; dim];
            for v in emb.iter_mut() { 
                *v = rng.random::<f32>();
            }
            (format!("doc-{}", i), emb, Some("bench".to_string()), None)
        })
        .collect();
    
    // Use one of the embeddings as query
    let query_embedding = docs_small[0].1.clone();
    
    // Bench: add batch
    let store_add = store.clone();
    let docs_add = docs_small.clone();
    
    c.bench_function("vector_add_100", |b| {
        b.to_async(&rt).iter(|| {
            let s = store_add.clone();
            let docs = docs_add.clone();
            async move {
                let _ = s.clear().await;
                let _ = s.add_documents_batch(black_box(docs)).await.unwrap();
            }
        })
    });
    
    // Prepare data for search bench (populate once)
    rt.block_on(async {
        let _ = store.clear().await;
        let _ = store.add_documents_batch(docs_small.clone()).await.unwrap();
    });
    
    let store_search = store.clone();
    let query = query_embedding.clone();
    
    c.bench_function("vector_search_100", |b| {
        b.to_async(&rt).iter(|| {
            let s = store_search.clone();
            let q = query.clone();
            async move {
                let _ = s.similarity_search(black_box(q), black_box(10)).await.unwrap();
            }
        })
    });
    
    // Cleanup: drop container inside the runtime context
    rt.block_on(async {
        drop(_container);
    });
}

criterion_group!(benches, vector_store_benchmarks);
criterion_main!(benches);