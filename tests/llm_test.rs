//! Unit tests for LLM struct and LLMActions trait
//!
//! Tests LLM creation, configuration, and action methods

use rustychain::{
    ToolCallingMode,
    llm::{
        FnDeclarator, FnExecutor, FunctionDeclaration, GenerationConfig, LLM, LLMProvider, Message,
        Role, ThinkingMode, ToolArgs,
    },
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ============================================================================
// Helper to create messages
// ============================================================================

fn user_message(text: &str) -> Message {
    Message {
        role: Role::User,
        message: Some(text.to_string()),
        audio: None,
        images: None,
        thinking: None,
        function_calls: vec![],
        function_results: vec![],
    }
}

// ============================================================================
// Test Tool for LLM tests
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize, Debug, Clone)]
pub struct TestToolArgs {
    pub input: String,
}

impl ToolArgs for TestToolArgs {}

#[derive(Clone)]
pub struct TestTool;

#[async_trait::async_trait]
impl FnExecutor<TestToolArgs, serde_json::Value> for TestTool {
    async fn call(&self, args: TestToolArgs) -> rustychain::Result<serde_json::Value> {
        Ok(serde_json::json!({"processed": args.input}))
    }
}

impl FnDeclarator<TestToolArgs, serde_json::Value> for TestTool {
    fn declare(&self) -> FunctionDeclaration<TestToolArgs, serde_json::Value> {
        FunctionDeclaration {
            name: "test_tool",
            description: "A test tool",
            parameters: schemars::schema_for!(TestToolArgs),
            executor: Arc::new(TestTool),
        }
    }
}

// ============================================================================
// LLM Provider Tests
// ============================================================================

#[test]
fn test_llm_provider_variants() {
    let google = LLMProvider::Google;
    let ollama = LLMProvider::Ollama;

    // Pattern matching
    assert!(matches!(google, LLMProvider::Google));
    assert!(matches!(ollama, LLMProvider::Ollama));
}

#[test]
fn test_llm_provider_debug() {
    let google = LLMProvider::Google;
    let ollama = LLMProvider::Ollama;

    let google_debug = format!("{:?}", google);
    let ollama_debug = format!("{:?}", ollama);

    assert!(google_debug.contains("Google"));
    assert!(ollama_debug.contains("Ollama"));
}

#[test]
fn test_llm_provider_clone() {
    let google = LLMProvider::Google;
    let cloned = google.clone();

    assert_eq!(google, cloned);
}

// ============================================================================
// LLM Builder Pattern Tests
// ============================================================================

#[test]
fn test_llm_builder_minimal() {
    let result = LLM::builder()
        .set_name("test-model".to_owned())
        .set_provider(LLMProvider::Google)
        .build();

    assert!(result.is_ok());
    let llm = result.unwrap();
    assert_eq!(llm.name, "test-model");
}

#[test]
fn test_llm_builder_with_all_options() {
    let tool = TestTool;
    let tool_decl = FnDeclarator::<TestToolArgs, _>::declare(&tool);

    let result = LLM::builder()
        .set_name("gemini-pro".to_owned())
        .set_provider(LLMProvider::Google)
        .set_authorization("test-api-key".to_owned())
        .set_system_prompt("You are a helpful assistant".to_owned())
        .add_tool(Arc::new(tool_decl))
        .build();

    assert!(result.is_ok());
    let llm = result.unwrap();
    assert_eq!(llm.name, "gemini-pro");
    assert!(llm.authorization.is_some());
    assert!(!llm.system_prompt.is_empty());
    assert_eq!(llm.tools.len(), 1);
}

#[test]
fn test_llm_builder_missing_name_fails() {
    let result = LLM::builder().set_provider(LLMProvider::Google).build();

    assert!(result.is_err());
}

#[test]
fn test_llm_builder_missing_provider_fails() {
    let result = LLM::builder().set_name("test".to_owned()).build();

    assert!(result.is_err());
}

// ============================================================================
// LLM Configuration Tests
// ============================================================================

#[test]
fn test_llm_has_correct_name() {
    let llm = LLM::builder()
        .set_name("gemini-2.0-flash".to_owned())
        .set_provider(LLMProvider::Google)
        .build()
        .unwrap();

    assert_eq!(llm.name, "gemini-2.0-flash");
}

#[test]
fn test_llm_has_system_prompt() {
    let llm = LLM::builder()
        .set_name("test".to_owned())
        .set_provider(LLMProvider::Google)
        .set_system_prompt("Be concise".to_owned())
        .build()
        .unwrap();

    assert_eq!(llm.system_prompt, "Be concise".to_string());
}

#[test]
fn test_llm_has_authorization() {
    let llm = LLM::builder()
        .set_name("test".to_owned())
        .set_provider(LLMProvider::Google)
        .set_authorization("secret-key".to_owned())
        .build()
        .unwrap();

    assert_eq!(llm.authorization, Some("secret-key".to_string()));
}

#[test]
fn test_llm_has_multiple_tools() {
    let tool1 = Arc::new(FnDeclarator::<TestToolArgs, _>::declare(&TestTool));
    let tool2 = Arc::new(FnDeclarator::<TestToolArgs, _>::declare(&TestTool));

    let llm = LLM::builder()
        .set_name("test".to_owned())
        .set_provider(LLMProvider::Google)
        .add_tool(tool1)
        .add_tool(tool2)
        .build()
        .unwrap();

    assert_eq!(llm.tools.len(), 2);
}

// ============================================================================
// LLM Provider-Specific Tests
// ============================================================================

#[test]
fn test_llm_google_provider() {
    let llm = LLM::builder()
        .set_name("gemini-pro".to_owned())
        .set_provider(LLMProvider::Google)
        .build()
        .unwrap();

    assert!(matches!(llm.provider, LLMProvider::Google));
}

#[test]
fn test_llm_ollama_provider() {
    let llm = LLM::builder()
        .set_name("llama3".to_owned())
        .set_provider(LLMProvider::Ollama)
        .build()
        .unwrap();

    assert!(matches!(llm.provider, LLMProvider::Ollama));
}

// ============================================================================
// Inference Creation Tests
// ============================================================================

#[test]
fn test_inference_creation_for_llm() {
    let _llm = LLM::builder()
        .set_name("test".to_owned())
        .set_provider(LLMProvider::Google)
        .build()
        .unwrap();
    let llm = LLM::builder()
        .set_name("test".to_owned())
        .set_provider(LLMProvider::Google)
        .build()
        .unwrap();

    let inference = llm
        .inference(user_message("Hello"))
        .with_config(GenerationConfig {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            max_output_tokens: 1024,
            thinking: ThinkingMode::None,
            candidate_count: 1,
            stop_sequences: None,
            output_schema: None,
            response_mime_type: None,
            tool_calling_mode: ToolCallingMode::Auto,
        });

    // Inference should be compatible with any LLM
    assert_eq!(inference.message.role, Role::User);
    assert_eq!(inference.config.temperature, 0.7);
}

// ============================================================================
// Tool Management Tests
// ============================================================================

#[test]
fn test_llm_tools_are_accessible() {
    let tool = Arc::new(FnDeclarator::<TestToolArgs, _>::declare(&TestTool));

    let llm = LLM::builder()
        .set_name("test".to_owned())
        .set_provider(LLMProvider::Google)
        .add_tool(tool)
        .build()
        .unwrap();

    // Should be able to access tools
    assert_eq!(llm.tools.len(), 1);
    assert_eq!(llm.tools[0].name(), "test_tool");
}

#[tokio::test]
async fn test_llm_tool_execution() {
    let tool = Arc::new(FnDeclarator::<TestToolArgs, _>::declare(&TestTool));

    let llm = LLM::builder()
        .set_name("test".to_owned())
        .set_provider(LLMProvider::Google)
        .add_tool(tool)
        .build()
        .unwrap();

    // Execute the tool through AnyFunction
    let args = serde_json::json!({"input": "test data"});
    let result = llm.tools[0].execute(&args).await.unwrap();

    assert_eq!(result["processed"], "test data");
}

// ============================================================================
// LLM Equality and Comparison Tests
// ============================================================================

#[test]
fn test_llm_different_names_are_different() {
    let llm1 = LLM::builder()
        .set_name("model-a".to_owned())
        .set_provider(LLMProvider::Google)
        .build()
        .unwrap();

    let llm2 = LLM::builder()
        .set_name("model-b".to_owned())
        .set_provider(LLMProvider::Google)
        .build()
        .unwrap();

    assert_ne!(llm1.name, llm2.name);
}

#[test]
fn test_llm_same_config_same_name() {
    let llm1 = LLM::builder()
        .set_name("test".to_owned())
        .set_provider(LLMProvider::Google)
        .set_system_prompt("Be helpful".to_owned())
        .build()
        .unwrap();

    let llm2 = LLM::builder()
        .set_name("test".to_owned())
        .set_provider(LLMProvider::Google)
        .set_system_prompt("Be helpful".to_owned())
        .build()
        .unwrap();

    assert_eq!(llm1.name, llm2.name);
    assert_eq!(llm1.system_prompt, llm2.system_prompt);
}

// ============================================================================
// Debug Output Tests
// ============================================================================

#[test]
fn test_llm_debug_output() {
    let llm = LLM::builder()
        .set_name("gemini-pro".to_owned())
        .set_provider(LLMProvider::Google)
        .set_authorization("secret".to_owned())
        .build()
        .unwrap();

    let debug = format!("{:?}", llm);

    // Should contain key fields
    assert!(debug.contains("gemini-pro"));
    assert!(debug.contains("Google"));
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_llm_creation_stress() {
    // Create many LLMs to ensure no memory issues
    let llms: Vec<_> = (0..100)
        .map(|i| {
            LLM::builder()
                .set_name(format!("model-{}", i))
                .set_provider(LLMProvider::Google)
                .build()
                .unwrap()
        })
        .collect();

    assert_eq!(llms.len(), 100);

    // Verify each has unique name
    for (i, llm) in llms.iter().enumerate() {
        assert_eq!(llm.name, format!("model-{}", i));
    }
}

#[test]
fn test_llm_with_many_tools() {
    let mut builder = LLM::builder()
        .set_name("test".to_owned())
        .set_provider(LLMProvider::Google);

    // Add 50 tools
    for _ in 0..50 {
        let tool = Arc::new(FnDeclarator::<TestToolArgs, _>::declare(&TestTool));
        builder = builder.add_tool(tool);
    }

    let llm = builder.build().unwrap();
    assert_eq!(llm.tools.len(), 50);
}

// ============================================================================
// Builder Pattern Idempotency Tests
// ============================================================================

#[test]
fn test_llm_builder_idempotent() {
    let mut results = Vec::new();

    for _ in 0..100 {
        let llm = LLM::builder()
            .set_name("test-model".to_owned())
            .set_provider(LLMProvider::Google)
            .set_system_prompt("System".to_owned())
            .set_authorization("Key".to_owned())
            .build()
            .unwrap();

        results.push((
            llm.name.clone(),
            llm.system_prompt.clone(),
            llm.authorization.clone(),
        ));
    }

    // All results should be identical
    let first = &results[0];
    for result in &results {
        assert_eq!(result.0, first.0);
        assert_eq!(result.1, first.1);
        assert_eq!(result.2, first.2);
    }
}
