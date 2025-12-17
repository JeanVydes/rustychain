//! gmail_send
use crate::FunctionDeclaration;
use crate::llm::function::{FnDeclarator, FnExecutor, ToolArgs};
use lettre::transport::smtp::authentication::Credentials;
use lettre::{Message, SmtpTransport, Transport};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
pub struct GmailArgs {
    #[schemars(description = "The sender's Gmail address.")]
    pub sender_email: String,
    #[schemars(description = "The Gmail App Password (NOT the regular account password).")]
    pub app_password: String,
    pub recipient_email: String,
    pub subject: String,
    pub body: String,
}

impl ToolArgs for GmailArgs {}

#[derive(Clone, Default)]
pub struct GmailSendTool;

#[async_trait::async_trait]
impl FnExecutor<GmailArgs, String> for GmailSendTool {
    async fn call(&self, args: GmailArgs) -> crate::Result<String> {
        let email = Message::builder()
            .from(
                args.sender_email
                    .parse()
                    .map_err(|_| crate::Error::Input("Invalid sender".into()))?,
            )
            .to(args
                .recipient_email
                .parse()
                .map_err(|_| crate::Error::Input("Invalid recipient".into()))?)
            .subject(args.subject)
            .body(args.body)
            .map_err(|e| crate::Error::Internal(e.into()))?;

        let creds = Credentials::new(args.sender_email, args.app_password);

        // Open a remote connection to gmail
        let mailer = SmtpTransport::relay("smtp.gmail.com")
            .map_err(|e| crate::Error::Internal(e.into()))?
            .credentials(creds)
            .build();

        // Send the email
        match mailer.send(&email) {
            Ok(_) => Ok("Email sent successfully!".to_string()),
            Err(e) => Err(crate::Error::Internal(e.into())),
        }
    }
}

impl FnDeclarator<GmailArgs, String> for GmailSendTool {
    fn declare(&self) -> FunctionDeclaration<GmailArgs, String> {
        FunctionDeclaration {
            name: "gmail_send_tool",
            description: "Sends an email via Gmail SMTP using an App Password.",
            parameters: schema_for!(GmailArgs),
            executor: Arc::new(self.clone()),
        }
    }
}
