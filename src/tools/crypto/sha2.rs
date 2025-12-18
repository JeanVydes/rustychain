//! hash
use crate::FunctionDeclaration;
use crate::llm::function::{FnDeclarator, FnExecutor};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256, Sha512};
use std::sync::Arc;

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
pub struct HashArgs {
    pub data: String,
    #[schemars(description = "Algorithm: 'sha256' or 'sha512'")]
    pub algorithm: String,
}

#[derive(Clone, Default)]
pub struct HashTool;

#[async_trait::async_trait]
impl FnExecutor<HashArgs, String> for HashTool {
    async fn call(&self, args: HashArgs) -> crate::Result<String> {
        match args.algorithm.to_lowercase().as_str() {
            "sha256" => Ok(format!("{:x}", Sha256::digest(args.data.as_bytes()))),
            "sha512" => Ok(format!("{:x}", Sha512::digest(args.data.as_bytes()))),
            _ => Err(crate::Error::Internal("Unsupported algorithm".into())),
        }
    }
}

impl FnDeclarator<HashArgs, String> for HashTool {
    fn declare(&self) -> FunctionDeclaration<HashArgs, String> {
        FunctionDeclaration {
            name: "hash_tool",
            description: "Generates a cryptographic hash (SHA256/512) for the given data.",
            parameters: schema_for!(HashArgs),
            executor: Arc::new(self.clone()),
        }
    }
}
