use base64::Engine;
#[cfg(feature = "google")]
use gemini_rust::{Blob, Content, FunctionResponse, Part};
#[cfg(feature = "ollama")]
use ollama_rs::generation::chat::ChatMessage;

use crate::{Inference, Role};

impl Inference {
    /// Converts this `Message` to a Gemini-compatible message.
    #[cfg(feature = "google")]
    pub fn to_gemini_message(&self) -> gemini_rust::Message {
        let role = match self.content.role {
            Role::User => gemini_rust::Role::User,
            Role::Tool => gemini_rust::Role::User,
            _ => gemini_rust::Role::Model,
        };

        let mut content: Content = Content {
            role: Some(role.clone()),
            parts: vec![].into(),
        };

        if let Some(audio) = &self.content.audio
            && let Some(parts) = &mut content.parts
        {
            let base64 = base64::engine::general_purpose::STANDARD.encode(audio);
            parts.push(Part::InlineData {
                inline_data: Blob::new("audio/mp3", base64),
            });
        }

        if let Some(msg) = &self.content.text
            && let Some(parts) = &mut content.parts
        {
            let is_thought = !self.thoughts.is_empty();
            parts.push(Part::Text {
                text: msg.clone(),
                thought: if is_thought { Some(true) } else { None },
                thought_signature: self
                    .thoughts
                    .first()
                    .and_then(|t| t.context.as_ref())
                    .and_then(|ctx| ctx.as_str())
                    .map(|s| s.to_string()),
            });
        }

        if !self.function_calls.is_empty()
            && let Some(parts) = &mut content.parts
        {
            for fc in &self.function_calls {
                parts.push(Part::FunctionCall {
                    function_call: fc.to_gemini(),
                    thought_signature: fc.context.clone(),
                });
            }
        }

        if !self.function_results.is_empty()
            && let Some(parts) = &mut content.parts
        {
            for fr in &self.function_results {
                parts.push(Part::FunctionResponse {
                    function_response: FunctionResponse {
                        name: fr.name.clone(),
                        response: Some(fr.results.clone()),
                    },
                });
            }
        }

        gemini_rust::Message { role, content }
    }

    /// Converts this `Message` to an Ollama-compatible message.
    #[cfg(feature = "ollama")]
    pub fn to_ollama_message(&self) -> ChatMessage {
        let role = Role::to_ollama(&self.content.role);
        let images = self
            .content
            .images
            .as_ref()
            .map(|imgs| imgs.iter().map(|img| img.to_ollama()).collect::<Vec<_>>());

        let tool_calls = self
            .function_calls
            .iter()
            .map(|fc| fc.to_ollama())
            .collect();

        let thinking = match self.thoughts.is_empty() {
            true => None,
            false => Some(
                self.thoughts
                    .iter()
                    .map(|t| t.text.clone())
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
        };

        let content = if self.content.role == Role::Tool && !self.function_results.is_empty() {
            // For tool role (function results), content is the result JSON
            self.function_results
                .first()
                .map(|fr| fr.results.to_string())
                .unwrap_or_default()
        } else {
            self.content.text.clone().unwrap_or_default()
        };

        ChatMessage {
            role,
            content,
            tool_calls,
            images,
            thinking,
        }
    }

    /// Converts this `Message` to an OpenAI-compatible message.
    #[cfg(feature = "openai")]
    pub fn to_openai_message(
        &self,
    ) -> crate::Result<openai_api_rs::v1::chat_completion::ChatCompletionMessage> {
        use openai_api_rs::v1::chat_completion::{Content, ToolCall, ToolCallFunction};

        // Convert function_calls to OpenAI tool_calls
        let tool_calls = if self.function_calls.is_empty() {
            None
        } else {
            let mut calls = Vec::new();

            for fc in self.function_calls.iter().cloned() {
                if fc.context.is_none() {
                    return Err(crate::Error::Generic(
                        "OpenAI tool call don't include context field".to_string(),
                    ));
                }

                calls.push(ToolCall {
                    // unwrap() is safe here due to the check above
                    id: fc.context.unwrap(),
                    r#type: "function_call".to_string(),
                    function: ToolCallFunction {
                        name: Some(fc.name.clone()),
                        arguments: Some(fc.arguments.to_string()),
                    },
                });
            }

            Some(calls)
        };

        // For tool role (function results), we need tool_call_id
        let tool_call_id = if self.content.role == Role::Tool && !self.function_results.is_empty() {
            Some(
                self.function_results
                    .first()
                    .map(|fr| fr.name.clone())
                    .unwrap_or_default(),
            )
        } else {
            None
        };

        // For tool results, content is the result JSON
        let content = if self.content.role == Role::Tool && !self.function_results.is_empty() {
            Content::Text(
                self.function_results
                    .first()
                    .map(|fr| fr.results.to_string())
                    .unwrap_or_default(),
            )
        } else {
            Content::Text(self.content.text.clone().unwrap_or_default())
        };

        Ok(openai_api_rs::v1::chat_completion::ChatCompletionMessage {
            role: Role::to_openai(&self.content.role),
            content,
            name: None,
            tool_call_id,
            tool_calls,
        })
    }

    #[cfg(feature = "openrouter")]
    pub fn to_openrouter_message(&self) -> crate::Result<openrouter_rs::api::chat::Message> {
        let content = match self.content.role {
            Role::Tool => {
                if !self.function_results.is_empty() {
                    self.function_results
                        .iter()
                        .map(|fr| fr.results.to_string())
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    self.content.text.clone().unwrap_or_default()
                }
            }
            _ => self.content.text.clone().unwrap_or_default(),
        };

        // use the first function result as tool_call_id and name
        let (name, tool_call_id) = self
            .function_results
            .first()
            .map(|fr| (Some(fr.name.clone()), fr.context.clone()))
            .unwrap_or((None, None));

        let tool_calls: Option<Vec<_>> = Some(
            self.function_calls
                .iter()
                .map(|fc| openrouter_rs::types::ToolCall {
                    id: fc.context.clone().unwrap_or_else(|| fc.name.clone()),
                    type_: "function".to_string(),
                    function: fc.to_openrouter(),
                })
                .collect::<Vec<_>>(),
        )
        .filter(|v| !v.is_empty());


        let res = openrouter_rs::api::chat::Message {
            role: Role::to_openrouter(&self.content.role),
            content,
            name,
            tool_call_id,
            tool_calls,
        };

        log::info!("Converted to OpenRouter message: {:?}", res);

        Ok(res)
    }
}
