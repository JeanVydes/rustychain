#![cfg(any(feature = "openai", feature = "google", feature = "ollama"))]
//! Unit tests for Conversation types
//!
//! Tests Inference, Role conversions and serialization consistency

use ollama_rs::generation::chat::MessageRole;
use rustychain::llm::{FunctionCall, FunctionResult, Inference, Role};
use serde_json::json;

// ============================================================================
// Role Tests
// ============================================================================

#[test]
fn test_role_serialization_lowercase() {
    assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
    assert_eq!(
        serde_json::to_string(&Role::Assistant).unwrap(),
        "\"assistant\""
    );
    assert_eq!(serde_json::to_string(&Role::System).unwrap(), "\"system\"");
    assert_eq!(serde_json::to_string(&Role::Tool).unwrap(), "\"tool\"");
}

#[test]
fn test_role_deserialization() {
    assert_eq!(
        serde_json::from_str::<Role>("\"user\"").unwrap(),
        Role::User
    );
    assert_eq!(
        serde_json::from_str::<Role>("\"assistant\"").unwrap(),
        Role::Assistant
    );
    assert_eq!(
        serde_json::from_str::<Role>("\"system\"").unwrap(),
        Role::System
    );
    assert_eq!(
        serde_json::from_str::<Role>("\"tool\"").unwrap(),
        Role::Tool
    );
}

#[test]
fn test_role_roundtrip_serialization() {
    let roles = [Role::User, Role::Assistant, Role::System, Role::Tool];

    for role in roles {
        let serialized = serde_json::to_string(&role).unwrap();
        let deserialized: Role = serde_json::from_str(&serialized).unwrap();
        assert_eq!(role, deserialized);
    }
}

#[test]
fn test_role_equality() {
    assert_eq!(Role::User, Role::User);
    assert_eq!(Role::Assistant, Role::Assistant);
    assert_ne!(Role::User, Role::Assistant);
}

#[test]
fn test_role_ordering_consistent() {
    // Run multiple times to ensure ordering is deterministic
    for _ in 0..10 {
        let mut roles = [Role::Tool, Role::User, Role::Assistant, Role::System];
        roles.sort();

        // Should always be in the same order (based on enum definition)
        assert_eq!(roles[0], Role::System);
        assert_eq!(roles[1], Role::Tool);
        assert_eq!(roles[2], Role::Assistant);
        assert_eq!(roles[3], Role::User);
    }
}

#[test]
fn test_role_hash_consistent() {
    use std::collections::HashSet;

    let mut set = HashSet::new();
    set.insert(Role::User);
    set.insert(Role::User); // Duplicate
    set.insert(Role::Assistant);

    assert_eq!(set.len(), 2); // Only unique values
    assert!(set.contains(&Role::User));
    assert!(set.contains(&Role::Assistant));
}

// ============================================================================
// Inference Tests
// ============================================================================

#[test]
fn test_inference_basic_creation() {
    let msg = Inference::as_user("Hello");

    assert_eq!(msg.content.role, Role::User);
    assert_eq!(msg.content.text, Some("Hello".to_string()));
}

#[test]
fn test_inference_serialization_roundtrip() {
    let mut original = Inference::as_user("Test Inference");
    original.thinking = Some("I'm thinking...".to_string());

    let serialized = serde_json::to_string(&original).unwrap();
    let deserialized: Inference = serde_json::from_str(&serialized).unwrap();

    assert_eq!(original.content.role, deserialized.content.role);
    assert_eq!(original.content.text, deserialized.content.text);
    assert_eq!(original.thinking, deserialized.thinking);
}

#[test]
fn test_inference_with_function_calls_roundtrip() {
    let mut original = Inference::as_assistant("");
    original = original.add_function_call(FunctionCall {
        name: "get_weather".to_string(),
        arguments: json!({"city": "London"}),
    });
    original = original.add_function_call(FunctionCall {
        name: "search".to_string(),
        arguments: json!({"query": "rust programming"}),
    });

    let serialized = serde_json::to_string(&original).unwrap();
    let deserialized: Inference = serde_json::from_str(&serialized).unwrap();

    assert_eq!(
        original.function_calls.len(),
        deserialized.function_calls.len()
    );
    assert_eq!(
        original.function_calls[0].name,
        deserialized.function_calls[0].name
    );
    assert_eq!(
        original.function_calls[0].arguments,
        deserialized.function_calls[0].arguments
    );
}

#[test]
fn test_inference_with_function_results_roundtrip() {
    let original = Inference::with_function_results(vec![FunctionResult {
        name: "get_weather".to_string(),
        results: json!({"temperature": 22, "condition": "sunny"}),
    }]);

    let serialized = serde_json::to_string(&original).unwrap();
    let deserialized: Inference = serde_json::from_str(&serialized).unwrap();

    assert_eq!(
        original.function_results.len(),
        deserialized.function_results.len()
    );
    assert_eq!(
        original.function_results[0].name,
        deserialized.function_results[0].name
    );
    assert_eq!(
        original.function_results[0].results,
        deserialized.function_results[0].results
    );
}

#[test]
fn test_inference_serialization_deterministic() {
    let msg = Inference::as_user("Hello");

    // Serialize multiple times and verify consistency
    let serialized1 = serde_json::to_string(&msg).unwrap();
    let serialized2 = serde_json::to_string(&msg).unwrap();
    let serialized3 = serde_json::to_string(&msg).unwrap();

    assert_eq!(serialized1, serialized2);
    assert_eq!(serialized2, serialized3);
}

#[test]
fn test_inference_clone() {
    let mut original = Inference::as_assistant("Response");
    original = original.with_audio(vec![1, 2, 3, 4]);
    original = original.add_function_call(FunctionCall {
        name: "test".to_string(),
        arguments: json!({}),
    });
    original.thinking = Some("Thinking".to_string());

    let cloned = original.clone();

    assert_eq!(original.content.role, cloned.content.role);
    assert_eq!(original.content.text, cloned.content.text);
    assert_eq!(original.content.audio, cloned.content.audio);
    assert_eq!(original.thinking, cloned.thinking);
    assert_eq!(original.function_calls.len(), cloned.function_calls.len());
}

// ============================================================================
// FunctionCall Tests
// ============================================================================

#[test]
fn test_function_call_serialization() {
    let fc = FunctionCall {
        name: "get_weather".to_string(),
        arguments: json!({"city": "New York", "unit": "celsius"}),
    };

    let serialized = serde_json::to_string(&fc).unwrap();
    let deserialized: FunctionCall = serde_json::from_str(&serialized).unwrap();

    assert_eq!(fc.name, deserialized.name);
    assert_eq!(fc.arguments, deserialized.arguments);
}

#[test]
fn test_function_call_complex_arguments() {
    let fc = FunctionCall {
        name: "complex_function".to_string(),
        arguments: json!({
            "nested": {
                "array": [1, 2, 3],
                "object": {"key": "value"}
            },
            "number": 42,
            "boolean": true,
            "null_value": null
        }),
    };

    let serialized = serde_json::to_string(&fc).unwrap();
    let deserialized: FunctionCall = serde_json::from_str(&serialized).unwrap();

    assert_eq!(fc.arguments["nested"]["array"][0], 1);
    assert_eq!(deserialized.arguments["nested"]["object"]["key"], "value");
}

#[test]
fn test_function_call_to_gemini_conversion() {
    let fc = FunctionCall {
        name: "search".to_string(),
        arguments: json!({"query": "test"}),
    };

    let gemini_fc = fc.to_gemini();

    assert_eq!(gemini_fc.name, "search");
    assert_eq!(gemini_fc.args["query"], "test");
}

#[test]
fn test_function_call_to_ollama_conversion() {
    let fc = FunctionCall {
        name: "calculate".to_string(),
        arguments: json!({"expression": "2+2"}),
    };

    let ollama_fc = fc.to_ollama();

    assert_eq!(ollama_fc.function.name, "calculate");
    assert_eq!(ollama_fc.function.arguments["expression"], "2+2");
}

#[test]
fn test_function_call_from_gemini_roundtrip() {
    let original = FunctionCall {
        name: "test_func".to_string(),
        arguments: json!({"arg1": "value1"}),
    };

    let gemini = original.to_gemini();
    let back = FunctionCall::from_gemini(gemini);

    assert_eq!(original.name, back.name);
    assert_eq!(original.arguments, back.arguments);
}

#[test]
fn test_function_call_from_ollama_roundtrip() {
    let original = FunctionCall {
        name: "test_func".to_string(),
        arguments: json!({"arg1": "value1"}),
    };

    let ollama = original.to_ollama();
    let back = FunctionCall::from_ollama(ollama);

    assert_eq!(original.name, back.name);
    assert_eq!(original.arguments, back.arguments);
}

// ============================================================================
// FunctionResult Tests
// ============================================================================

#[test]
fn test_function_result_serialization() {
    let fr = FunctionResult {
        name: "get_weather".to_string(),
        results: json!({"temperature": 25, "unit": "celsius"}),
    };

    let serialized = serde_json::to_string(&fr).unwrap();
    let deserialized: FunctionResult = serde_json::from_str(&serialized).unwrap();

    assert_eq!(fr.name, deserialized.name);
    assert_eq!(fr.results, deserialized.results);
}

#[test]
fn test_function_result_with_error() {
    let fr = FunctionResult {
        name: "failed_function".to_string(),
        results: json!({"error": "Something went wrong", "code": 500}),
    };

    let serialized = serde_json::to_string(&fr).unwrap();
    let deserialized: FunctionResult = serde_json::from_str(&serialized).unwrap();

    assert_eq!(deserialized.results["error"], "Something went wrong");
}

// ============================================================================
// Conversion Consistency Tests
// ============================================================================

#[test]
fn test_gemini_conversion_idempotent() {
    let msg = Inference::as_user("Test Inference");

    // Convert multiple times and verify consistency
    let gemini1 = msg.to_gemini_message();
    let gemini2 = msg.to_gemini_message();

    // Content should be the same
    assert_eq!(
        gemini1.content.parts.as_ref().map(|p| p.len()),
        gemini2.content.parts.as_ref().map(|p| p.len())
    );
}

#[test]
fn test_ollama_conversion_idempotent() {
    let msg = Inference::as_assistant("Response");

    // Convert multiple times
    let ollama1 = msg.to_ollama_message();
    let ollama2 = msg.to_ollama_message();

    assert_eq!(ollama1.content, ollama2.content);
    assert_eq!(ollama1.role, ollama2.role);
}

#[test]
fn test_role_to_gemini_mapping() {
    let user_msg = Inference::as_user("test");
    let assistant_msg = Inference::as_assistant("test");

    let gemini_user = user_msg.to_gemini_message();
    let gemini_assistant = assistant_msg.to_gemini_message();

    assert_eq!(gemini_user.role, gemini_rust::Role::User);
    assert_eq!(gemini_assistant.role, gemini_rust::Role::Model);
}

#[test]
fn test_role_to_ollama_mapping() {
    let roles = [
        (Role::User, MessageRole::User),
        (Role::Assistant, MessageRole::Assistant),
        (Role::System, MessageRole::System),
        (Role::Tool, MessageRole::Tool),
    ];

    for (our_role, expected_ollama_role) in roles {
        let msg = Inference::with_content(our_role, "test");

        let ollama = msg.to_ollama_message();
        assert_eq!(ollama.role, expected_ollama_role);
    }
}
