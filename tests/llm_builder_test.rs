#![cfg(any(feature = "openai", feature = "google", feature = "ollama"))]
//! Unit tests for LLM Builder
//!
//! Tests the builder pattern for LLM construction

use rustychain::llm::{LLM, LLMProvider};

#[test]
fn test_builder_with_all_required_fields() {
    let llm = LLM::builder()
        .set_name("test-model".to_string())
        .set_provider(LLMProvider::Google)
        .build();

    assert!(llm.is_ok());
    let llm = llm.unwrap();
    assert_eq!(llm.name, "test-model");
    assert_eq!(llm.provider, LLMProvider::Google);
    assert_eq!(llm.system_prompt, ""); // Default empty
    assert!(llm.authorization.is_none());
    assert!(llm.tools.is_empty());
}

#[test]
fn test_builder_missing_name_fails() {
    let result = LLM::builder().set_provider(LLMProvider::Google).build();

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Name"));
}

#[test]
fn test_builder_missing_provider_fails() {
    let result = LLM::builder().set_name("test-model".to_string()).build();

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Provider"));
}

#[test]
fn test_builder_with_system_prompt() {
    let prompt = "You are a helpful assistant.";
    let llm = LLM::builder()
        .set_name("test-model".to_string())
        .set_provider(LLMProvider::Ollama)
        .set_system_prompt(prompt.to_string())
        .build()
        .unwrap();

    assert_eq!(llm.system_prompt, prompt);
}

#[test]
fn test_builder_with_authorization() {
    let auth = "sk-test-key-12345";
    let llm = LLM::builder()
        .set_name("gemini-pro".to_string())
        .set_provider(LLMProvider::Google)
        .set_authorization(auth.to_string())
        .build()
        .unwrap();

    assert_eq!(llm.authorization, Some(auth.to_string()));
}

#[test]
fn test_builder_chain_order_independent() {
    // Build with one order
    let llm1 = LLM::builder()
        .set_name("model".to_string())
        .set_provider(LLMProvider::Google)
        .set_system_prompt("prompt".to_string())
        .set_authorization("auth".to_string())
        .build()
        .unwrap();

    // Build with different order
    let llm2 = LLM::builder()
        .set_authorization("auth".to_string())
        .set_system_prompt("prompt".to_string())
        .set_provider(LLMProvider::Google)
        .set_name("model".to_string())
        .build()
        .unwrap();

    assert_eq!(llm1.name, llm2.name);
    assert_eq!(llm1.provider, llm2.provider);
    assert_eq!(llm1.system_prompt, llm2.system_prompt);
    assert_eq!(llm1.authorization, llm2.authorization);
}

#[test]
fn test_builder_provider_google() {
    let llm = LLM::builder()
        .set_name("gemini-2.5-flash".to_string())
        .set_provider(LLMProvider::Google)
        .build()
        .unwrap();

    assert_eq!(llm.provider, LLMProvider::Google);
}

#[test]
fn test_builder_provider_ollama() {
    let llm = LLM::builder()
        .set_name("llama3".to_string())
        .set_provider(LLMProvider::Ollama)
        .build()
        .unwrap();

    assert_eq!(llm.provider, LLMProvider::Ollama);
}

#[test]
fn test_provider_serialization() {
    // Test that providers serialize correctly
    let google_json = serde_json::to_string(&LLMProvider::Google).unwrap();
    let ollama_json = serde_json::to_string(&LLMProvider::Ollama).unwrap();

    assert_eq!(google_json, "\"google\"");
    assert_eq!(ollama_json, "\"ollama\"");

    // And deserialize back
    let google: LLMProvider = serde_json::from_str(&google_json).unwrap();
    let ollama: LLMProvider = serde_json::from_str(&ollama_json).unwrap();

    assert_eq!(google, LLMProvider::Google);
    assert_eq!(ollama, LLMProvider::Ollama);
}

#[test]
fn test_provider_equality() {
    assert_eq!(LLMProvider::Google, LLMProvider::Google);
    assert_eq!(LLMProvider::Ollama, LLMProvider::Ollama);
    assert_ne!(LLMProvider::Google, LLMProvider::Ollama);
}

#[test]
fn test_provider_ordering() {
    // Providers should have consistent ordering for use in collections
    let mut providers = [LLMProvider::Ollama, LLMProvider::Google];
    providers.sort();

    // Ordering is based on enum variant order
    assert_eq!(providers[0], LLMProvider::Google);
    assert_eq!(providers[1], LLMProvider::Ollama);
}
