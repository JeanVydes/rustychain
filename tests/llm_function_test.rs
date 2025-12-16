#![cfg(any(feature = "openai", feature = "google", feature = "ollama"))]
//! Unit tests for Function Calling system
//!
//! Tests FunctionDeclaration, FnExecutor, FnDeclarator traits

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;

use rustychain::llm::{AnyFunction, FnDeclarator, FnExecutor, FunctionDeclaration, ToolArgs};

// ============================================================================
// Test Tool Definitions
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize, Debug, Clone)]
pub struct SimpleArgs {
    pub message: String,
}

impl ToolArgs for SimpleArgs {}

#[derive(JsonSchema, Serialize, Deserialize, Debug, Clone)]
pub struct ComplexArgs {
    pub query: String,
    pub limit: Option<i32>,
    pub filters: Option<Vec<String>>,
}

impl ToolArgs for ComplexArgs {}

#[derive(JsonSchema, Serialize, Deserialize, Debug, Clone)]
pub struct NestedArgs {
    pub config: ConfigArgs,
    pub data: Vec<DataItem>,
}

#[derive(JsonSchema, Serialize, Deserialize, Debug, Clone)]
pub struct ConfigArgs {
    pub enabled: bool,
    pub threshold: f64,
}

#[derive(JsonSchema, Serialize, Deserialize, Debug, Clone)]
pub struct DataItem {
    pub id: i32,
    pub value: String,
}

impl ToolArgs for NestedArgs {}

// Simple executor that echoes the input
#[derive(Clone)]
pub struct EchoExecutor;

#[async_trait::async_trait]
impl FnExecutor<SimpleArgs, serde_json::Value> for EchoExecutor {
    async fn call(&self, args: SimpleArgs) -> rustychain::Result<serde_json::Value> {
        Ok(json!({
            "echoed": args.message
        }))
    }
}

impl FnDeclarator<SimpleArgs, serde_json::Value> for EchoExecutor {
    fn declare(&self) -> FunctionDeclaration<SimpleArgs, serde_json::Value> {
        FunctionDeclaration {
            name: "echo",
            description: "Echoes the input message",
            parameters: schemars::schema_for!(SimpleArgs),
            executor: Arc::new(EchoExecutor),
        }
    }
}

// Complex executor
#[derive(Clone)]
pub struct SearchExecutor;

#[async_trait::async_trait]
impl FnExecutor<ComplexArgs, serde_json::Value> for SearchExecutor {
    async fn call(&self, args: ComplexArgs) -> rustychain::Result<serde_json::Value> {
        Ok(json!({
            "query": args.query,
            "limit": args.limit.unwrap_or(10),
            "filters_count": args.filters.map(|f| f.len()).unwrap_or(0)
        }))
    }
}

impl FnDeclarator<ComplexArgs, serde_json::Value> for SearchExecutor {
    fn declare(&self) -> FunctionDeclaration<ComplexArgs, serde_json::Value> {
        FunctionDeclaration {
            name: "search",
            description: "Search with optional filters",
            parameters: schemars::schema_for!(ComplexArgs),
            executor: Arc::new(SearchExecutor),
        }
    }
}

// ============================================================================
// FunctionDeclaration Tests
// ============================================================================

#[test]
fn test_function_declaration_creation() {
    let decl = EchoExecutor.declare();

    assert_eq!(decl.name, "echo");
    assert_eq!(decl.description, "Echoes the input message");
    assert!(!decl.parameters.to_value().is_null());
}

#[test]
fn test_function_declaration_debug() {
    let decl = EchoExecutor.declare();
    let debug_str = format!("{:?}", decl);

    assert!(debug_str.contains("echo"));
    assert!(debug_str.contains("Echoes the input message"));
    assert!(debug_str.contains("<executor>"));
}

#[test]
fn test_function_declaration_clone() {
    let decl = EchoExecutor.declare();
    let cloned = decl.clone();

    assert_eq!(decl.name, cloned.name);
    assert_eq!(decl.description, cloned.description);
    assert_eq!(decl.parameters, cloned.parameters);
}

#[test]
fn test_any_function_name() {
    let decl = EchoExecutor.declare();
    let any_fn: &dyn AnyFunction = &decl;

    assert_eq!(any_fn.name(), "echo");
}

#[test]
fn test_any_function_description() {
    let decl = EchoExecutor.declare();
    let any_fn: &dyn AnyFunction = &decl;

    assert_eq!(any_fn.description(), "Echoes the input message");
}

#[test]
fn test_any_function_parameters_schema() {
    let decl = EchoExecutor.declare();
    let any_fn: &dyn AnyFunction = &decl;
    let schema = any_fn.parameters_schema();

    // Schema should have properties
    assert!(schema.clone().to_value().is_object());
}

#[tokio::test]
async fn test_any_function_execute() {
    let decl = EchoExecutor.declare();
    let any_fn: &dyn AnyFunction = &decl;

    let args = json!({"message": "Hello, World!"});
    let result = any_fn.execute(&args).await.unwrap();

    assert_eq!(result["echoed"], "Hello, World!");
}

#[tokio::test]
async fn test_any_function_execute_with_complex_args() {
    let decl = SearchExecutor.declare();
    let any_fn: &dyn AnyFunction = &decl;

    let args = json!({
        "query": "rust programming",
        "limit": 5,
        "filters": ["language:rust", "stars:>100"]
    });

    let result = any_fn.execute(&args).await.unwrap();

    assert_eq!(result["query"], "rust programming");
    assert_eq!(result["limit"], 5);
    assert_eq!(result["filters_count"], 2);
}

#[tokio::test]
async fn test_any_function_execute_with_optional_args() {
    let decl = SearchExecutor.declare();
    let any_fn: &dyn AnyFunction = &decl;

    // Only required field
    let args = json!({"query": "test"});
    let result = any_fn.execute(&args).await.unwrap();

    assert_eq!(result["query"], "test");
    assert_eq!(result["limit"], 10); // Default
    assert_eq!(result["filters_count"], 0);
}

#[tokio::test]
async fn test_any_function_execute_invalid_args() {
    let decl = EchoExecutor.declare();
    let any_fn: &dyn AnyFunction = &decl;

    // Missing required field
    let args = json!({});
    let result = any_fn.execute(&args).await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("deserialize"));
}

#[tokio::test]
async fn test_any_function_execute_wrong_type() {
    let decl = EchoExecutor.declare();
    let any_fn: &dyn AnyFunction = &decl;

    // Wrong type for message (number instead of string)
    let args = json!({"message": 12345});
    let result = any_fn.execute(&args).await;

    assert!(result.is_err());
}

// ============================================================================
// Schema Generation Tests
// ============================================================================

#[test]
fn test_schema_contains_required_fields() {
    let schema = schemars::schema_for!(SimpleArgs);

    // Should have properties
    assert!(schema.clone().to_value()["properties"].is_object());
    assert!(schema.to_value()["properties"]["message"].is_object());
}

#[test]
fn test_schema_optional_fields() {
    let schema = schemars::schema_for!(ComplexArgs);

    // Should have properties for all fields
    assert!(schema.clone().to_value()["properties"]["query"].is_object());
    assert!(schema.clone().to_value()["properties"]["limit"].is_object());
    assert!(schema.clone().to_value()["properties"]["filters"].is_object());
}

#[test]
fn test_schema_nested_types() {
    let schema = schemars::schema_for!(NestedArgs);

    // Should handle nested types
    assert!(schema.clone().to_value()["properties"]["config"].is_object());
    assert!(schema.clone().to_value()["properties"]["data"].is_object());
}

// ============================================================================
// Gemini Tool Definition Tests
// ============================================================================

#[test]
fn test_gemini_tool_definition() {
    let decl = EchoExecutor.declare();
    let any_fn: &dyn AnyFunction = &decl;
    let gemini_tool = any_fn.gemini_tool_definition();

    // Should be a Function variant
    match gemini_tool {
        gemini_rust::Tool::Function {
            function_declarations,
        } => {
            assert_eq!(function_declarations.len(), 1);
            assert_eq!(function_declarations[0].name, "echo");
        }
        _ => panic!("Expected Function tool"),
    }
}

// ============================================================================
// Multiple Implementations Tests
// ============================================================================

#[derive(Clone)]
pub struct MultiImplTool;

#[async_trait::async_trait]
impl FnExecutor<SimpleArgs, serde_json::Value> for MultiImplTool {
    async fn call(&self, args: SimpleArgs) -> rustychain::Result<serde_json::Value> {
        Ok(json!({"simple": args.message}))
    }
}

#[async_trait::async_trait]
impl FnExecutor<ComplexArgs, serde_json::Value> for MultiImplTool {
    async fn call(&self, args: ComplexArgs) -> rustychain::Result<serde_json::Value> {
        Ok(json!({"complex": args.query}))
    }
}

#[async_trait::async_trait]
impl FnDeclarator<SimpleArgs, serde_json::Value> for MultiImplTool {
    fn declare(&self) -> FunctionDeclaration<SimpleArgs, serde_json::Value> {
        FunctionDeclaration {
            name: "multi_simple",
            description: "Multi tool simple variant",
            parameters: schemars::schema_for!(SimpleArgs),
            executor: Arc::new(MultiImplTool),
        }
    }
}

#[async_trait::async_trait]
impl FnDeclarator<ComplexArgs, serde_json::Value> for MultiImplTool {
    fn declare(&self) -> FunctionDeclaration<ComplexArgs, serde_json::Value> {
        FunctionDeclaration {
            name: "multi_complex",
            description: "Multi tool complex variant",
            parameters: schemars::schema_for!(ComplexArgs),
            executor: Arc::new(MultiImplTool),
        }
    }
}

#[test]
fn test_multiple_declarations_from_same_tool() {
    let tool = MultiImplTool;

    let simple_decl = FnDeclarator::<SimpleArgs, _>::declare(&tool);
    let complex_decl = FnDeclarator::<ComplexArgs, _>::declare(&tool);

    assert_eq!(simple_decl.name, "multi_simple");
    assert_eq!(complex_decl.name, "multi_complex");
}

#[tokio::test]
async fn test_multiple_executions_from_same_tool() {
    let tool = MultiImplTool;

    let simple_decl = FnDeclarator::<SimpleArgs, _>::declare(&tool);
    let complex_decl = FnDeclarator::<ComplexArgs, _>::declare(&tool);

    let simple_result = simple_decl
        .executor
        .call(SimpleArgs {
            message: "hello".to_string(),
        })
        .await
        .unwrap();

    let complex_result = complex_decl
        .executor
        .call(ComplexArgs {
            query: "search".to_string(),
            limit: None,
            filters: None,
        })
        .await
        .unwrap();

    assert_eq!(simple_result["simple"], "hello");
    assert_eq!(complex_result["complex"], "search");
}

// ============================================================================
// Executor Isolation Tests
// ============================================================================

#[tokio::test]
async fn test_executor_is_stateless() {
    let decl = EchoExecutor.declare();

    // Call multiple times
    let result1 = decl
        .executor
        .call(SimpleArgs {
            message: "first".to_string(),
        })
        .await
        .unwrap();
    let result2 = decl
        .executor
        .call(SimpleArgs {
            message: "second".to_string(),
        })
        .await
        .unwrap();
    let result3 = decl
        .executor
        .call(SimpleArgs {
            message: "first".to_string(),
        })
        .await
        .unwrap();

    // Results should be independent
    assert_eq!(result1["echoed"], "first");
    assert_eq!(result2["echoed"], "second");
    assert_eq!(result3["echoed"], "first");

    // Same input should give same output
    assert_eq!(result1, result3);
}
