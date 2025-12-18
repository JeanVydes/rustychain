//! symmetric
use crate::FunctionDeclaration;
use crate::llm::function::{FnDeclarator, FnExecutor};
use aes_gcm::{
    Aes256Gcm, Key, Nonce,
    aead::{Aead, KeyInit},
};
use rand::RngCore;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
pub struct SymmetricArgs {
    pub action: String, // "encrypt" or "decrypt"
    pub text: String,
    #[schemars(description = "32-byte key in hex format")]
    pub key_hex: String,
}

#[derive(Clone, Default)]
pub struct SymmetricTool;

#[async_trait::async_trait]
impl FnExecutor<SymmetricArgs, String> for SymmetricTool {
    async fn call(&self, args: SymmetricArgs) -> crate::Result<String> {
        let key_bytes = hex::decode(&args.key_hex)?;
        let key = Key::<Aes256Gcm>::from_slice(&key_bytes);
        let cipher = Aes256Gcm::new(key);

        match args.action.to_lowercase().as_str() {
            "encrypt" => {
                let mut nonce_bytes = [0u8; 12];
                rand::rng().fill_bytes(&mut nonce_bytes);
                let nonce = Nonce::from_slice(&nonce_bytes);
                let ciphertext = cipher.encrypt(nonce, args.text.as_bytes())?;
                Ok(format!(
                    "{}:{}",
                    hex::encode(nonce_bytes),
                    hex::encode(ciphertext)
                ))
            }
            "decrypt" => {
                let parts: Vec<&str> = args.text.split(':').collect();
                if parts.len() != 2 {
                    return Err(crate::Error::Internal(
                        "Invalid encrypted format. Expected nonce:ciphertext".into(),
                    ));
                }
                let nonce = hex::decode(parts[0])?;
                let nonce = Nonce::from_slice(&nonce);
                let ciphertext = hex::decode(parts[1])?;
                let plaintext = cipher.decrypt(nonce, ciphertext.as_ref())?;
                Ok(String::from_utf8_lossy(&plaintext).to_string())
            }
            _ => Err(crate::Error::Internal("Invalid action".into())),
        }
    }
}

impl FnDeclarator<SymmetricArgs, String> for SymmetricTool {
    fn declare(&self) -> FunctionDeclaration<SymmetricArgs, String> {
        FunctionDeclaration {
            name: "symmetric_crypto_tool",
            description: "Encrypts or decrypts text using AES-256-GCM. Format: 'nonce:ciphertext' in hex.",
            parameters: schema_for!(SymmetricArgs),
            executor: Arc::new(self.clone()),
        }
    }
}
