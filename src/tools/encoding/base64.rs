//! base64
//!
//! Tool for Base64 encoding and decoding.

use crate::llm::function::{FnDeclarator, FnExecutor, ToolArgs};
use crate::{BASE64_ENGINE, FunctionDeclaration};
use base64::Engine as _;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
pub struct Base64Args {
    #[schemars(description = "The action to perform: 'encode' or 'decode'.")]
    pub action: String,
    #[schemars(description = "The data string to process.")]
    pub data: String,
}

impl ToolArgs for Base64Args {}

#[derive(Clone, Default)]
pub struct Base64Tool;

#[async_trait::async_trait]
impl FnExecutor<Base64Args, String> for Base64Tool {
    async fn call(&self, args: Base64Args) -> crate::Result<String> {
        match args.action.to_lowercase().as_str() {
            "encode" => Ok(BASE64_ENGINE.encode(args.data)),
            "decode" => {
                let decoded_bytes = BASE64_ENGINE
                    .decode(&args.data)
                    .map_err(|e| crate::Error::Internal(format!("Invalid Base64: {}", e).into()))?;
                Ok(String::from_utf8_lossy(&decoded_bytes).to_string())
            }
            _ => Err(crate::Error::Internal(
                "Invalid action. Use 'encode' or 'decode'.".into(),
            )),
        }
    }
}

impl FnDeclarator<Base64Args, String> for Base64Tool {
    fn declare(&self) -> FunctionDeclaration<Base64Args, String> {
        FunctionDeclaration {
            name: "base64_tool",
            description: "Encodes data to Base64 format or decodes Base64 strings back to text.",
            parameters: schema_for!(Base64Args),
            executor: Arc::new(self.clone()),
        }
    }
}
