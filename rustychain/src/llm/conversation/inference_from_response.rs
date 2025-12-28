use crate::{inference::Thought, prelude::InferenceContent};
use base64::Engine;
#[cfg(feature = "google")]
use gemini_rust::{GenerationResponse, Part};
#[cfg(feature = "ollama")]
use ollama_rs::generation::chat::ChatMessageResponse;
#[cfg(feature = "openai")]
use openai_api_rs::v1::chat_completion::ChatCompletionChoice;
#[cfg(feature = "openai")]
use openai_api_rs::v1::chat_completion::chat_completion_stream::ChatCompletionStreamResponse;
use serde_json::Value;
use std::sync::Arc;

#[cfg(feature = "google")]
use crate::GenerationConfig;
use crate::{FinishReason, Image, Inference, Role, llm::function::FunctionCall};

impl Inference {
    #[cfg(feature = "openrouter")]
    pub fn from_openrouter_response(
        openrouter_message: &openrouter_rs::types::CompletionsResponse,
        config: Option<Arc<GenerationConfig>>,
    ) -> Self {
        let choice = &openrouter_message.choices[0];
        let finish_reason = choice.finish_reason().map(FinishReason::from_openrouter);
        let mut function_calls: Vec<FunctionCall> = choice
            .tool_calls()
            .unwrap_or(&[])
            .iter()
            .map(|fc| FunctionCall::from_openrouter(fc))
            .collect();

        let mut text = choice.content().map(|s| s.to_string());
        if let Some(cfg) = &config
            && let Some(content) = serde_json::from_str::<Value>(
                &choice.content().map(|s| s.to_string()).unwrap_or_default(),
            )
            .ok()
            && !cfg.native_tool_handling
        {
            use crate::non_native_function_calling::{
                extract_content_from_value_opt, extract_function_calls_from_value_opt,
            };

            if let Some(inner_content) = extract_content_from_value_opt(&content) {
                text = inner_content;
            }

            if let Some(fc) = extract_function_calls_from_value_opt(&content) {
                for call in fc {
                    function_calls.push(call);
                }
            }
        }

        Inference {
            model: Some(openrouter_message.model.clone()),
            content: InferenceContent {
                role: Role::from_openrouter_str(choice.role().unwrap_or_default()),
                text,
                audio: None,
                images: None,
            },
            thoughts: choice
                .reasoning()
                .map(|thinking| {
                    vec![Thought {
                        text: thinking.to_string(),
                        context: None,
                    }]
                })
                .unwrap_or_default(),
            function_calls,
            function_results: vec![],
            finish_reason,
            usage: None,
        }
    }

    /// Converts a Gemini `GenerationResponse` to this `Message` format.
    #[cfg(feature = "google")]
    pub fn from_google_response(
        gemini_message: GenerationResponse,
        config: Option<Arc<GenerationConfig>>,
    ) -> Inference {
        let candidate = match gemini_message.candidates.first() {
            Some(c) => c,
            None => {
                let message = format!("No candidates in response: {:?}", gemini_message);

                return Inference {
                    model: gemini_message.model_version,
                    content: InferenceContent {
                        role: Role::Assistant,
                        text: Some(message),
                        audio: None,
                        images: None,
                    },
                    ..Default::default()
                };
            }
        };

        let mut function_calls = vec![];
        let mut message_parts = vec![];
        let mut audio_parts = vec![];
        let mut thoughts: Vec<Thought> = vec![];
        if let Some(parts) = &candidate.content.parts {
            for part in parts {
                if let Part::FunctionCall { function_call, .. } = part {
                    function_calls.push(FunctionCall::from_gemini(function_call.clone()));
                }

                if let Part::Text {
                    text,
                    thought,
                    thought_signature,
                } = part
                {
                    if let Some(true) = thought {
                        thoughts.push(Thought {
                            text: text.clone(),
                            context: thought_signature.as_ref().map(|s| Value::String(s.clone())),
                        });
                    } else if let Some(cfg) = &config
                        && !cfg.native_tool_handling
                    {
                        use crate::non_native_function_calling::extract_content_from_value_opt;

                        if let Ok(value) = serde_json::from_str::<Value>(text) {
                            if let Some(inner_content) = extract_content_from_value_opt(&value) {
                                message_parts.push(inner_content);
                            } else {
                                message_parts.push(text.clone());
                            }

                            if let Some(calls) =
                                    crate::non_native_function_calling::extract_function_calls_from_value_opt(
                                        &value,
                                    )
                                {
                                    for call in calls {
                                        function_calls.push(call);
                                    }
                                }
                        }
                    } else {
                        message_parts.push(text.clone());
                    }
                }

                if let Part::InlineData { inline_data } = part
                    && inline_data.mime_type.starts_with("audio/")
                {
                    let decoded = base64::engine::general_purpose::STANDARD
                        .decode(&inline_data.data)
                        .ok();
                    if let Some(audio_data) = decoded {
                        audio_parts.push(audio_data);
                    }
                }

                if let Part::FunctionResponse { .. } = part {
                    // Currently, we do not extract function results from Gemini responses.
                }
            }
        }

        let finish_reason = candidate
            .finish_reason
            .as_ref()
            .map(FinishReason::from_google);

        Inference {
            model: gemini_message.model_version.clone(),
            content: InferenceContent {
                role: Role::from_google(
                    &candidate
                        .content
                        .role
                        .clone()
                        .unwrap_or(gemini_rust::Role::Model),
                ),
                text: if message_parts.is_empty() {
                    None
                } else {
                    Some(message_parts.join("\n"))
                },
                audio: audio_parts.into_iter().next(),
                images: None,
            },
            thoughts,
            function_calls,
            finish_reason,
            ..Default::default()
        }
    }

    /// Converts an Ollama `ChatMessageResponse` to this `Message` format.
    #[cfg(feature = "ollama")]
    pub fn from_ollama_response(
        ollama_message: ChatMessageResponse,
        config: Option<Arc<GenerationConfig>>,
    ) -> Inference {
        use crate::prelude::InferenceContent;

        let images = ollama_message.message.images.as_ref().map(|imgs| {
            imgs.iter()
                .map(|img| Image::from_ollama(img.clone()))
                .collect()
        });

        let finish_reason = if !ollama_message.message.tool_calls.is_empty() {
            Some(FinishReason::ToolCall)
        } else {
            None
        };

        let mut function_calls: Vec<FunctionCall> = ollama_message
            .message
            .tool_calls
            .into_iter()
            .map(FunctionCall::from_ollama)
            .collect();

        let mut text = ollama_message.message.content.clone();
        if let Some(cfg) = &config
            && let Some(content) =
                serde_json::from_str::<Value>(&ollama_message.message.content).ok()
            && !cfg.native_tool_handling
        {
            use crate::non_native_function_calling::{
                extract_content_from_value_opt, extract_function_calls_from_value_opt,
            };

            if let Some(inner_content) = extract_content_from_value_opt(&content) {
                text = inner_content;
            }

            if let Some(fc) = extract_function_calls_from_value_opt(&content) {
                for call in fc {
                    function_calls.push(call);
                }
            }
        }

        Inference {
            model: Some(ollama_message.model),
            content: InferenceContent {
                role: Role::from_ollama(&ollama_message.message.role),
                text: Some(text),
                audio: None,
                images,
            },
            thoughts: if let Some(thinking) = &ollama_message.message.thinking {
                vec![Thought {
                    text: thinking.clone(),
                    context: None,
                }]
            } else {
                vec![]
            },
            function_calls,
            finish_reason,
            ..Default::default()
        }
    }

    /// Converts an OpenAI `ChatCompletionMessage` to this `Message` format.
    #[cfg(feature = "openai")]
    pub fn from_openai_choice(
        choice: &ChatCompletionChoice,
        config: Option<Arc<GenerationConfig>>,
    ) -> Inference {
        let mut function_calls: Vec<FunctionCall> = choice
            .message
            .tool_calls
            .as_ref()
            .map(|tcs| {
                tcs.iter()
                    .map(|tc| FunctionCall::from_openai(tc.clone()))
                    .collect()
            })
            .unwrap_or_default();

        let finish_reason = choice.finish_reason.as_ref().map(FinishReason::from_openai);
        let thoughts = if let Some(text) = &choice.message.reasoning_content {
            vec![Thought {
                text: text.clone(),
                context: None,
            }]
        } else {
            vec![]
        };

        let mut text: Option<String> = choice.message.content.clone();
        if let Some(cfg) = &config
            && !cfg.native_tool_handling
        {
            // this means that the `text` is following schema NonNativeFunctionCallingSchema (in theory)
            // so we should try to parse it as such to extract the `calls` and `original`, original goes directly to message_parts and calls to function_calls
            if let Some(content) = &choice.message.content {
                use crate::non_native_function_calling::{
                    extract_content_from_value_opt, extract_function_calls_from_value_opt,
                };

                if let Ok(value) = serde_json::from_str::<Value>(content) {
                    if let Some(inner_content) = extract_content_from_value_opt(&value) {
                        text = Some(inner_content);
                    }

                    if let Some(calls) = extract_function_calls_from_value_opt(&value) {
                        for call in calls {
                            function_calls.push(call);
                        }
                    }
                }
            }
        }

        Inference {
            // i think name is not the model name
            model: choice.message.name.clone(),
            content: InferenceContent {
                role: Role::from_openai(&choice.message.role),
                text,
                audio: None,
                images: None,
            },
            thoughts,
            function_calls,
            finish_reason,
            ..Default::default()
        }
    }

    // OpenAI Streaming doesn't support non native function calling yet, so no config here, and probably no will ever, because we need the full response to parse
    // So the convertion with non native function calling should be done after the full response is reassembled from the stream, not here
    #[cfg(feature = "openai")]
    pub fn from_openai_stream_response(
        message: ChatCompletionStreamResponse,
    ) -> crate::Result<Inference> {
        match message {
            ChatCompletionStreamResponse::ToolCall(toolcalls) => Ok(Inference {
                content: InferenceContent {
                    role: Role::Assistant,
                    text: None,
                    audio: None,
                    images: None,
                },
                function_calls: toolcalls
                    .iter()
                    .map(|tc| FunctionCall::from_openai(tc.clone()))
                    .collect(),
                finish_reason: Some(FinishReason::ToolCall),
                ..Default::default()
            }),
            ChatCompletionStreamResponse::Content(content) => Ok(Inference {
                content: InferenceContent {
                    role: Role::Assistant,
                    text: Some(content),
                    audio: None,
                    images: None,
                },
                finish_reason: Some(FinishReason::Stop),
                ..Default::default()
            }),
            ChatCompletionStreamResponse::Done => Ok(Inference {
                content: InferenceContent {
                    role: Role::Assistant,
                    text: None,
                    audio: None,
                    images: None,
                },
                finish_reason: Some(FinishReason::Stop),
                ..Default::default()
            }),
        }
    }
}
