//! argon2
use crate::llm::function::{FnDeclarator, FnExecutor, ToolArgs};
use crate::{BASE64_ENGINE, FunctionDeclaration};
use argon2::password_hash::Salt;
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use base64::Engine as _;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
pub struct ArgonArgs {
    #[schemars(description = "Action to perform: 'hash' or 'verify'")]
    pub action: String,
    pub password: String,
    #[schemars(description = "Salt string (required for 'hash')")]
    pub salt: Option<String>,
    #[schemars(description = "The previously generated hash string (required for 'verify')")]
    pub encoded_hash: Option<String>,
}

impl ToolArgs for ArgonArgs {}

#[derive(Clone, Default)]
pub struct ArgonTool;

#[async_trait::async_trait]
impl FnExecutor<ArgonArgs, String> for ArgonTool {
    async fn call(&self, args: ArgonArgs) -> crate::Result<String> {
        let argon2 = Argon2::default();

        match args.action.to_lowercase().as_str() {
            "hash" => {
                let salt_str = args
                    .salt
                    .ok_or_else(|| crate::Error::Internal("Salt is required for hashing".into()))?;
                let salt_b64 = BASE64_ENGINE.encode(salt_str.as_bytes());
                let salt = Salt::from_b64(&salt_b64)?;

                let password_hash = argon2.hash_password(args.password.as_bytes(), salt)?;

                Ok(password_hash.to_string())
            }
            "verify" => {
                let hash_str = args.encoded_hash.ok_or_else(|| {
                    crate::Error::Internal("Encoded hash is required for verification".into())
                })?;
                let parsed_hash = PasswordHash::new(&hash_str)?;

                match argon2.verify_password(args.password.as_bytes(), &parsed_hash) {
                    Ok(_) => Ok("valid".to_string()),
                    Err(_) => Ok("invalid".to_string()),
                }
            }
            _ => Err(crate::Error::Internal(
                "Invalid action. Use 'hash' or 'verify'.".into(),
            )),
        }
    }
}

impl FnDeclarator<ArgonArgs, String> for ArgonTool {
    fn declare(&self) -> FunctionDeclaration<ArgonArgs, String> {
        FunctionDeclaration {
            name: "argon2_tool",
            description: "Hashes passwords or verifies them against an existing Argon2 hash.",
            parameters: schema_for!(ArgonArgs),
            executor: Arc::new(self.clone()),
        }
    }
}
