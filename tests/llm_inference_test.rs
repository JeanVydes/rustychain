//! Unit tests for Inference builder pattern
//!
//! Tests that Inference correctly builds requests for generate/stream

use rustychain::ToolCallingMode;
use rustychain::llm::{
    FunctionCall, FunctionResult, GenerationConfig, LLM, LLMProvider, Message, Role, ThinkingMode,
};

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

fn assistant_message(text: &str) -> Message {
    Message {
        role: Role::Assistant,
        message: Some(text.to_string()),
        audio: None,
        images: None,
        thinking: None,
        function_calls: vec![],
        function_results: vec![],
    }
}

fn system_message(text: &str) -> Message {
    Message {
        role: Role::System,
        message: Some(text.to_string()),
        audio: None,
        images: None,
        thinking: None,
        function_calls: vec![],
        function_results: vec![],
    }
}

fn tool_message(results: Vec<FunctionResult>) -> Message {
    Message {
        role: Role::Tool,
        message: None,
        audio: None,
        images: None,
        thinking: None,
        function_calls: vec![],
        function_results: results,
    }
}

fn dummy_llm() -> LLM {
    LLM {
        name: "test-model".to_string(),
        system_prompt: "".to_string(),
        provider: LLMProvider::OpenAI,
        authorization: None,
        endpoint: None,
        tools: vec![],
    }
}

// ============================================================================
// Inference Builder Tests
// ============================================================================

#[test]
fn test_inference_new_creates_empty() {
    let llm = dummy_llm();
    let inf = llm.inference(user_message(""));

    assert!(inf.history.is_empty());
    assert_eq!(inf.message.role, Role::User);
    assert!(inf.output_schema.is_none());
}

#[test]
fn test_inference_with_message() {
    let llm = dummy_llm();
    let inf = llm.inference(user_message("Hello"));

    assert_eq!(inf.message.role, Role::User);
    assert_eq!(inf.message.message.as_ref().unwrap(), "Hello");
}

#[test]
fn test_inference_with_history() {
    let history = vec![
        system_message("You are helpful"),
        user_message("Hi"),
        assistant_message("Hello!"),
    ];

    let llm = dummy_llm();
    let inf = llm.inference(user_message("")).with_history(history);

    assert_eq!(inf.history.len(), 3);
    assert_eq!(inf.history[0].role, Role::System);
    assert_eq!(inf.history[1].role, Role::User);
    assert_eq!(inf.history[2].role, Role::Assistant);
}

#[test]
fn test_inference_with_config() {
    let config = GenerationConfig {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        max_output_tokens: 1024,
        thinking: ThinkingMode::None,
        candidate_count: 1,
        stop_sequences: Some(vec!["STOP".to_string()]),
        output_schema: None,
        response_mime_type: None,
        tool_calling_mode: ToolCallingMode::Auto,
    };

    let llm = dummy_llm();
    let inf = llm.inference(user_message("")).with_config(config.clone());

    assert_eq!(inf.config.temperature, 0.7);
    assert_eq!(inf.config.top_p, 0.9);
    assert_eq!(inf.config.max_output_tokens, 1024);
}

#[test]
fn test_inference_with_audio() {
    let audio_data = vec![0u8; 1024];
    let msg = user_message("with audio").with_audio(audio_data.clone());
    let llm = dummy_llm();

    let inf = llm.inference(msg);

    assert!(inf.message.audio.is_some());
    assert_eq!(inf.message.audio.as_ref().unwrap().len(), 1024);
}

#[test]
fn test_inference_chained_building() {
    let msg = user_message("Explain Rust").with_audio(vec![1, 2, 3]);
    let llm = dummy_llm();

    let inf = llm
        .inference(msg)
        .with_history(vec![system_message("Be concise")])
        .with_config(GenerationConfig {
            temperature: 0.5,
            top_p: 0.9,
            top_k: 40,
            max_output_tokens: 500,
            thinking: ThinkingMode::None,
            candidate_count: 1,
            stop_sequences: None,
            output_schema: None,
            response_mime_type: None,
            tool_calling_mode: ToolCallingMode::Auto,
        });

    assert_eq!(inf.history.len(), 1);
    assert_eq!(inf.message.role, Role::User);
    assert!(inf.message.audio.is_some());
}

// ============================================================================
// GenerationConfig Tests
// ============================================================================

#[test]
fn test_generation_config_default() {
    let config = GenerationConfig::default();

    assert_eq!(config.temperature, 0.5);
    assert_eq!(config.top_p, 0.9);
    assert_eq!(config.top_k, 40);
    assert_eq!(config.max_output_tokens, 2048);
    assert!(matches!(config.thinking, ThinkingMode::None));
}

#[test]
fn test_generation_config_with_values() {
    let config = GenerationConfig {
        temperature: 0.8,
        top_p: 0.95,
        top_k: 50,
        max_output_tokens: 2048,
        thinking: ThinkingMode::Sized(1000),
        candidate_count: 1,
        stop_sequences: Some(vec!["END".to_string(), "STOP".to_string()]),
        output_schema: None,
        response_mime_type: None,
        tool_calling_mode: ToolCallingMode::Auto,
    };

    assert_eq!(config.temperature, 0.8);
    assert_eq!(config.top_p, 0.95);
    assert_eq!(config.max_output_tokens, 2048);
    assert_eq!(config.stop_sequences.as_ref().unwrap().len(), 2);
}

#[test]
fn test_thinking_mode_none() {
    let mode = ThinkingMode::None;
    assert!(matches!(mode, ThinkingMode::None));
}

#[test]
fn test_thinking_mode_dynamic() {
    let mode = ThinkingMode::Dynamic;
    assert!(matches!(mode, ThinkingMode::Dynamic));
}

#[test]
fn test_thinking_mode_sized() {
    let mode = ThinkingMode::Sized(2000);

    if let ThinkingMode::Sized(tokens) = mode {
        assert_eq!(tokens, 2000);
    } else {
        panic!("Expected Sized variant");
    }
}

// ============================================================================
// Function in Inference Tests
// ============================================================================

#[test]
fn test_inference_with_function_result() {
    let result = FunctionResult {
        name: "get_weather".to_string(),
        results: serde_json::json!({"temp": 25, "condition": "sunny"}),
    };

    let msg = tool_message(vec![result]);
    let llm = dummy_llm();
    let inf = llm.inference(msg);

    assert_eq!(inf.message.role, Role::Tool);
}

// ============================================================================
// Inference Immutability Tests
// ============================================================================

#[test]
fn test_inference_builder_is_consuming() {
    let llm = dummy_llm();
    let inf1 = llm.inference(user_message("Initial"));

    // Each call consumes and returns new inference
    let inf2 = inf1.with_history(vec![user_message("Hello")]);

    // inf2 should have history and keep the original message
    assert_eq!(inf2.history.len(), 1);
    assert_eq!(inf2.message.role, Role::User);
}

// ============================================================================
// Integration with LLM Builder Tests
// ============================================================================

#[test]
fn test_inference_with_llm_config_compatibility() {
    // Test that GenerationConfig works correctly
    let config = GenerationConfig {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        max_output_tokens: 1024,
        thinking: ThinkingMode::None,
        candidate_count: 1,
        stop_sequences: Some(vec!["STOP".to_string()]),
        output_schema: None,
        response_mime_type: None,
        tool_calling_mode: ToolCallingMode::Auto,
    };

    let llm = dummy_llm();

    // Build inference
    let inf = llm
        .inference(user_message("Test"))
        .with_config(config.clone());

    // Verify config is preserved
    assert_eq!(inf.config.temperature, 0.7);
    assert_eq!(inf.config.max_output_tokens, 1024);
}

// ============================================================================
// Message History Preservation Tests
// ============================================================================

#[test]
fn test_inference_preserves_history_order() {
    let history = vec![
        system_message("System"),
        user_message("User 1"),
        assistant_message("Model 1"),
        user_message("User 2"),
        assistant_message("Model 2"),
    ];

    let llm = dummy_llm();
    let inf = llm.inference(user_message("")).with_history(history);

    assert_eq!(inf.history[0].role, Role::System);
    assert_eq!(inf.history[1].role, Role::User);
    assert_eq!(inf.history[2].role, Role::Assistant);
    assert_eq!(inf.history[3].role, Role::User);
    assert_eq!(inf.history[4].role, Role::Assistant);
}

// ============================================================================
// Complex Conversation Tests
// ============================================================================

#[test]
fn test_inference_complex_conversation() {
    let call = FunctionCall {
        name: "search".to_string(),
        arguments: serde_json::json!({"query": "rust async"}),
    };

    let result = FunctionResult {
        name: "search".to_string(),
        results: serde_json::json!({"results": ["tokio", "async-std"]}),
    };

    // Message with function call
    let msg_with_call = Message {
        role: Role::Assistant,
        message: None,
        audio: None,
        images: None,
        thinking: None,
        function_calls: vec![call],
        function_results: vec![],
    };

    let history = vec![
        system_message("You are a search assistant"),
        user_message("Search for Rust async libraries"),
        msg_with_call,
        tool_message(vec![result]),
    ];

    let llm = dummy_llm();
    let inf = llm
        .inference(assistant_message("I found tokio and async-std"))
        .with_history(history);

    assert_eq!(inf.history.len(), 4);
    assert_eq!(inf.message.role, Role::Assistant);

    // Verify roles in history
    assert_eq!(inf.history[0].role, Role::System);
    assert_eq!(inf.history[1].role, Role::User);
    assert_eq!(inf.history[2].role, Role::Assistant);
    assert_eq!(inf.history[3].role, Role::Tool);
}

// ============================================================================
// Idempotency Tests
// ============================================================================

#[test]
fn test_inference_building_is_idempotent() {
    let msg = user_message("Hello");
    let config = GenerationConfig {
        temperature: 0.5,
        top_p: 0.9,
        top_k: 40,
        max_output_tokens: 100,
        thinking: ThinkingMode::None,
        candidate_count: 1,
        stop_sequences: None,
        output_schema: None,
        response_mime_type: None,
        tool_calling_mode: ToolCallingMode::Auto,
    };

    // Build multiple times with same input
    let mut results = Vec::new();
    let llm = dummy_llm();

    for _ in 0..100 {
        let inf = llm.inference(msg.clone()).with_config(config.clone());

        results.push((inf.message.role.clone(), inf.config.temperature));
    }

    // All results should be identical
    let first = &results[0];
    for result in &results {
        assert_eq!(result.0, first.0);
        assert_eq!(result.1, first.1);
    }
}

// ============================================================================
// Debug and Clone Tests
// ============================================================================

#[test]
fn test_generation_config_clone() {
    let config = GenerationConfig {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        max_output_tokens: 1024,
        thinking: ThinkingMode::Sized(500),
        candidate_count: 1,
        stop_sequences: Some(vec!["STOP".to_string()]),
        output_schema: None,
        response_mime_type: None,
        tool_calling_mode: ToolCallingMode::Auto,
    };

    let cloned = config.clone();

    assert_eq!(config.temperature, cloned.temperature);
    assert_eq!(config.top_p, cloned.top_p);
    assert_eq!(config.max_output_tokens, cloned.max_output_tokens);
    assert_eq!(config.stop_sequences, cloned.stop_sequences);
}

#[test]
fn test_generation_config_debug() {
    let config = GenerationConfig {
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
    };

    let debug = format!("{:?}", config);

    assert!(debug.contains("temperature"));
    assert!(debug.contains("0.7"));
    assert!(debug.contains("max_output_tokens"));
    assert!(debug.contains("1024"));
}
