use std::{collections::HashMap, fmt::Debug, sync::Arc};

use gemini_rust::{
    FunctionDeclaration as GeminiFunctionDeclaration, Tool as GeminiTool, tools::Behavior,
};
use ollama_rs::generation::tools::ToolCallFunction;
use openai_api_rs::v1::types::{
    Function as OpenAIFunction, FunctionParameters as OpenAIFunctionParameters, JSONSchemaDefine,
    JSONSchemaType,
};
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize, de};
use serde_json::Value;

// Experimental, this needs more work
/// Converts a schemars-generated JSON schema Value to OpenAI's FunctionParameters
fn schema_to_openai_parameters(schema: &Value) -> OpenAIFunctionParameters {
    let properties = schema
        .get("properties")
        .and_then(|p| p.as_object())
        .map(|props| {
            props
                .iter()
                .map(|(key, value)| (key.clone(), Box::new(value_to_json_schema_define(value))))
                .collect::<HashMap<String, Box<JSONSchemaDefine>>>()
        });

    let required = schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        });

    OpenAIFunctionParameters {
        schema_type: JSONSchemaType::Object,
        properties,
        required,
    }
}

/// Converts a serde_json::Value to OpenAI's JSONSchemaDefine
fn value_to_json_schema_define(value: &Value) -> JSONSchemaDefine {
    let schema_type = value.get("type").and_then(|t| t.as_str()).map(|t| match t {
        "string" => JSONSchemaType::String,
        "number" | "integer" => JSONSchemaType::Number,
        "boolean" => JSONSchemaType::Boolean,
        "array" => JSONSchemaType::Array,
        "object" => JSONSchemaType::Object,
        _ => JSONSchemaType::String,
    });

    let description = value
        .get("description")
        .and_then(|d| d.as_str())
        .map(|s| s.to_string());

    let enum_values = value.get("enum").and_then(|e| e.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect()
    });

    let properties = value
        .get("properties")
        .and_then(|p| p.as_object())
        .map(|props| {
            props
                .iter()
                .map(|(key, val)| (key.clone(), Box::new(value_to_json_schema_define(val))))
                .collect::<HashMap<String, Box<JSONSchemaDefine>>>()
        });

    let required = value.get("required").and_then(|r| r.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect()
    });

    let items = value
        .get("items")
        .map(|i| Box::new(value_to_json_schema_define(i)));

    JSONSchemaDefine {
        schema_type,
        description,
        enum_values,
        properties,
        required,
        items,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: Value,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResult {
    pub name: String,
    pub results: Value,
}

impl FunctionCall {
    #[cfg(feature = "google")]
    pub fn to_gemini(&self) -> gemini_rust::FunctionCall {
        gemini_rust::FunctionCall {
            name: self.name.clone(),
            args: self.arguments.clone(),
            thought_signature: None,
        }
    }

    #[cfg(feature = "ollama")]
    pub fn to_ollama(&self) -> ollama_rs::generation::tools::ToolCall {
        ollama_rs::generation::tools::ToolCall {
            function: ToolCallFunction {
                name: self.name.clone(),
                arguments: self.arguments.clone(),
            },
        }
    }

    #[cfg(feature = "openai")]
    pub fn to_openai(&self) -> openai_api_rs::v1::chat_completion::ToolCallFunction {
        openai_api_rs::v1::chat_completion::ToolCallFunction {
            name: Some(self.name.clone()),
            arguments: Some(self.arguments.to_string()),
        }
    }

    #[cfg(feature = "google")]
    pub fn from_gemini(gemini_call: gemini_rust::FunctionCall) -> FunctionCall {
        FunctionCall {
            name: gemini_call.name,
            arguments: gemini_call.args,
        }
    }

    #[cfg(feature = "ollama")]
    pub fn from_ollama(ollama_call: ollama_rs::generation::tools::ToolCall) -> FunctionCall {
        FunctionCall {
            name: ollama_call.function.name,
            arguments: ollama_call.function.arguments,
        }
    }

    #[cfg(feature = "openai")]
    pub fn from_openai(openai_call: openai_api_rs::v1::chat_completion::ToolCall) -> FunctionCall {
        FunctionCall {
            name: openai_call.function.name.unwrap_or_default(),
            arguments: openai_call
                .function
                .arguments
                .as_ref()
                .and_then(|args_str| serde_json::from_str(args_str).ok())
                .unwrap_or(Value::Null),
        }
    }
}

pub trait ToolArgs: JsonSchema + Serialize + Send + Sync {}

#[async_trait::async_trait]
pub trait FnExecutor<A, R>: Send + Sync
where
    A: ToolArgs + Send + Sync,
    R: Serialize + Send + Sync,
{
    async fn call(&self, args: A) -> crate::Result<R>;
}

pub trait FnDeclarator<A, R>: Send + Sync
where
    A: ToolArgs + Send + Sync,
    R: Serialize + Send + Sync,
{
    fn declare(&self) -> FunctionDeclaration<A, R>;
}

#[async_trait::async_trait]
pub trait AnyFunction: Send + Debug + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> &Schema;

    async fn execute(&self, args: &serde_json::Value) -> crate::Result<serde_json::Value>;

    #[cfg(feature = "google")]
    fn gemini_tool_definition(&self) -> GeminiTool;
    #[cfg(feature = "openai")]
    fn openai_tool_definition(&self) -> openai_api_rs::v1::chat_completion::Tool;
    #[cfg(feature = "ollama")]
    fn ollama_tool_definition(&self) -> ollama_rs::generation::tools::ToolInfo;
}

impl<A, R> From<FunctionDeclaration<A, R>> for Arc<dyn AnyFunction>
where
    A: de::DeserializeOwned + Debug + ToolArgs + 'static,
    R: Serialize + Send + Sync + 'static,
{
    fn from(decl: FunctionDeclaration<A, R>) -> Self {
        Arc::new(decl)
    }
}

#[async_trait::async_trait]
impl<A, R> AnyFunction for FunctionDeclaration<A, R>
where
    A: de::DeserializeOwned + Debug + ToolArgs + 'static,
    R: Serialize + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        self.description
    }

    fn parameters_schema(&self) -> &Schema {
        &self.parameters
    }

    async fn execute(&self, args: &serde_json::Value) -> crate::Result<serde_json::Value> {
        let concrete_args: A = serde_json::from_value(args.clone())?;

        let result = self.executor.call(concrete_args).await?;
        Ok(serde_json::to_value(result)?)
    }

    #[cfg(feature = "google")]
    fn gemini_tool_definition(&self) -> GeminiTool {
        GeminiTool::Function {
            function_declarations: vec![
                GeminiFunctionDeclaration::new(
                    self.name,
                    self.description,
                    Some(Behavior::Blocking),
                )
                .with_parameters::<A>()
                .with_response::<serde_json::Value>(),
            ],
        }
    }

    #[cfg(feature = "openai")]
    fn openai_tool_definition(&self) -> openai_api_rs::v1::chat_completion::Tool {
        let schema_val = serde_json::to_value(&self.parameters).unwrap_or(Value::Null);
        openai_api_rs::v1::chat_completion::Tool {
            r#type: openai_api_rs::v1::chat_completion::ToolType::Function,
            function: OpenAIFunction {
                name: String::from(self.name),
                description: Some(String::from(self.description)),
                parameters: schema_to_openai_parameters(&schema_val),
            },
        }
    }

    #[cfg(feature = "ollama")]
    fn ollama_tool_definition(&self) -> ollama_rs::generation::tools::ToolInfo {
        ollama_rs::generation::tools::ToolInfo {
            tool_type: ollama_rs::generation::tools::ToolType::Function,
            function: ollama_rs::generation::tools::ToolFunctionInfo {
                name: self.name.to_string(),
                description: self.description.to_string(),
                parameters: self.parameters.clone(),
            },
        }
    }
}

#[derive(Clone)]
pub struct FunctionDeclaration<A, R>
where
    A: ToolArgs,
    R: Serialize + Send + Sync,
{
    pub name: &'static str,
    pub description: &'static str,
    pub parameters: Schema,
    pub executor: Arc<dyn FnExecutor<A, R>>,
}

impl<A, R> Debug for FunctionDeclaration<A, R>
where
    A: ToolArgs,
    R: Serialize + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionDeclaration")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .field("executor", &"<executor>")
            .finish()
    }
}

#[macro_export]
macro_rules! declare_function {
    ($name:expr, $description:expr, $args_ty:ty, $result_ty:ty, $executor:expr) => {{
        let schema = schemars::schema_for!($args_ty);
        FunctionDeclaration {
            name: $name,
            description: $description,
            parameters: schema.schema,
            executor: std::sync::Arc::new($executor),
        }
    }};
}
