//! hex
//!
//! Tool for hex encoding and decoding.

use crate::FunctionDeclaration;
use crate::llm::function::{FnDeclarator, FnExecutor};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
pub struct HexArgs {
    #[schemars(description = "The action to perform: 'encode' or 'decode'.")]
    pub action: String,
    #[schemars(description = "The string to process.")]
    pub data: String,
}

#[derive(Clone, Default)]
pub struct HexTool;

#[async_trait::async_trait]
impl FnExecutor<HexArgs, String> for HexTool {
    async fn call(&self, args: HexArgs) -> crate::Result<String> {
        match args.action.to_lowercase().as_str() {
            "encode" => Ok(hex::encode(args.data)),
            "decode" => {
                let decoded = hex::decode(&args.data)?;
                Ok(String::from_utf8_lossy(&decoded).to_string())
            }
            _ => Err(crate::Error::Internal(
                "Invalid action. Use 'encode' or 'decode'.".into(),
            )),
        }
    }
}

impl FnDeclarator<HexArgs, String> for HexTool {
    fn declare(&self) -> FunctionDeclaration<HexArgs, String> {
        FunctionDeclaration {
            name: "hex_tool",
            description: "Encodes text to hexadecimal or decodes hexadecimal back to text.",
            parameters: schema_for!(HexArgs),
            executor: Arc::new(self.clone()),
        }
    }
}
