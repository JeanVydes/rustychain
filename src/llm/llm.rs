use std::{error::Error, sync::Arc};

use futures_core::stream::Stream;
use futures_util::{StreamExt, TryStreamExt};
use gemini_rust::{
    FunctionCallingMode, Gemini, GeminiBuilder, GenerationConfig as GeminiGenerationConfig,
    ThinkingConfig,
};
use ollama_rs::{
    Ollama,
    generation::chat::{ChatMessage, request::ChatMessageRequest},
};
use openai_api_rs::v1::{
    api::{OpenAIClient, OpenAIClientBuilder},
    chat_completion::{
        ChatCompletionMessage,
        chat_completion::ChatCompletionRequest,
        chat_completion_stream::{ChatCompletionStreamRequest, ChatCompletionStreamResponse},
    },
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;
use url::Url;

use crate::{
    CoreError, FunctionCall,
    llm::{
        builder::LLMBuilder,
        conversation::{Message, Role},
        function::AnyFunction,
        inference::Inference,
    },
};

/// Enum representing supported LLM providers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum LLMProvider {
    Google,
    OpenAI,
    Anthropic,
    Ollama,
}

/// Different modes for LLM "thinking" or planning.
///
/// - `None`: No additional thinking time.
/// - `Dynamic`: Let the LLM decide how much time to spend thinking.
/// - `Sized(i32)`: Allocate a specific amount of time or budget for thinking.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash, PartialOrd, Ord)]
pub enum ThinkingMode {
    None,
    Dynamic,
    Sized(i32),
}

/// Struct representing a Large Language Model (LLM) with its configuration and capabilities.
#[derive(Clone, Debug)]
pub struct LLM {
    pub name: String,
    pub system_prompt: String,
    pub provider: LLMProvider,
    pub authorization: Option<String>,
    pub endpoint: Option<String>,
    pub tools: Vec<Arc<dyn AnyFunction>>,
}

/// Configuration parameters for text generation by the LLM.
///
/// These parameters influence the behavior and output of the LLM during generation tasks.
///
/// - `temperature`: Controls the randomness of the output. Higher values yield more diverse results.
/// - `top_p`: Nucleus sampling parameter to limit the token selection to a subset of probable tokens.
/// - `top_k`: Limits the token selection to the top K most probable tokens.
/// - `max_output_tokens`: Maximum number of tokens to generate in the output.
/// - `thinking`: Mode for LLM thinking/planning before generating a response.
/// - `candidate_count`: Number of candidate responses to generate.
/// - `stop_sequences`: Optional sequences that, when generated, will stop further output.
/// - `output_schema`: Optional JSON schema to structure the output.
/// - `response_mime_type`: Optional MIME type for the response format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub max_output_tokens: i32,
    pub thinking: ThinkingMode,
    pub candidate_count: i32,
    pub stop_sequences: Option<Vec<String>>,
    pub output_schema: Option<Value>,
    pub response_mime_type: Option<String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.5,
            top_p: 0.9,
            top_k: 40,
            max_output_tokens: 2048,
            thinking: ThinkingMode::None,
            candidate_count: 1,
            stop_sequences: None,
            output_schema: None,
            response_mime_type: None,
        }
    }
}

/// Trait defining the actions that can be performed by an LLM.
pub trait LLMActions {
    /// Creates a new inference with this model.
    /// This serve as a gateway to perform various LLM operations.
    fn inference(&self) -> Inference;

    /// Performs text generation based on the provided history and message.
    fn generation(
        &self,
        history: &mut Vec<Message>,
        message: Message,
        config: GenerationConfig,
    ) -> impl Future<Output = crate::Result<Message>> + Send;

    /// Streams text generation results based on the provided history and message.
    fn stream(
        &self,
        history: &mut Vec<Message>,
        message: Message,
        config: GenerationConfig,
    ) -> impl Future<
        Output = crate::Result<
            Pin<Box<dyn Stream<Item = crate::Result<Message>> + Send + 'static>>,
        >,
    > + Send;

    /// Streams text generation results with audio input.
    fn stream_with_audio_input(
        &self,
        history: &mut Vec<Message>,
        audio: Vec<u8>,
        config: GenerationConfig,
    ) -> impl Future<
        Output = crate::Result<
            Pin<Box<dyn Stream<Item = crate::Result<Message>> + Send + 'static>>,
        >,
    > + Send;

    /// Performs text generation with audio input.
    fn generation_with_audio_input(
        &self,
        history: &mut Vec<Message>,
        audio: Vec<u8>,
        config: GenerationConfig,
    ) -> impl Future<Output = crate::Result<Message>> + Send;

    /// Generates an embedding for the given text.
    fn embedding(
        &self,
        text: &str,
        dim: i32,
    ) -> impl Future<Output = crate::Result<Vec<f32>>> + Send;
}

impl LLM {
    /// Creates a Gemini client configured for this LLM.
    #[cfg(feature = "google")]
    pub fn get_gemini_client(&self) -> crate::Result<Gemini> {
        let authorization = match &self.authorization {
            Some(a) => a,
            None => return Err(Box::from(CoreError::Generic("Not auth".to_owned()))),
        };

        let mut client =
            GeminiBuilder::new(authorization.clone()).with_model(format!("models/{}", self.name));

        if let Some(endpoint) = &self.endpoint {
            client = client.with_base_url(Url::parse(endpoint).map_err(
                |err| -> Box<dyn Error + Send + Sync> {
                    Box::from(CoreError::Generic(format!(
                        "Invalid Gemini endpoint URL: {}",
                        err
                    )))
                },
            )?);
        }

        let client = match client.build() {
            Ok(c) => c,
            Err(e) => return Err(Box::from(CoreError::Gemini(e))),
        };

        return Ok(client);
    }

    /// Creates an OpenAI client configured for this LLM.
    #[cfg(feature = "openai")]
    pub fn get_openai_client(&self) -> crate::Result<OpenAIClient> {
        let authorization = match &self.authorization {
            Some(a) => a,
            None => return Err(Box::from(CoreError::Generic("Not auth".to_owned()))),
        };

        let mut client = OpenAIClientBuilder::new().with_api_key(authorization.clone());

        if let Some(endpoint) = &self.endpoint {
            client = client.with_endpoint(endpoint.clone());
        }

        let client = client
            .build()
            .map_err(|err| -> Box<dyn Error + Send + Sync> {
                Box::from(CoreError::OpenAI(format!("OpenAI client error: {}", err)))
            })?;

        return Ok(client);
    }

    /// Adds a single tool to the LLM.
    pub fn add_tool<R>(&mut self, tool: Arc<dyn AnyFunction>) {
        self.tools.push(tool);
    }

    /// Adds multiple tools to the LLM.
    pub fn add_tools(&mut self, tools: Vec<Arc<dyn AnyFunction>>) {
        for tool in tools {
            self.tools.push(tool);
        }
    }

    /// Creates a new LLM builder.
    pub fn builder() -> LLMBuilder {
        LLMBuilder {
            name: None,
            system_prompt: None,
            provider: None,
            authorization: None,
            tools: vec![],
            endpoint: None,
        }
    }
}

impl LLMActions for LLM {
    /// Creates a new inference with this model.
    fn inference(&self) -> Inference {
        Inference::new().with_llm(Arc::new(self.clone()))
    }

    /// Performs text generation based on the provided history and message.
    async fn generation(
        &self,
        history: &mut Vec<Message>,
        message: Message,
        config: GenerationConfig,
    ) -> crate::Result<Message> {
        match self.provider {
            #[cfg(feature = "google")]
            LLMProvider::Google => {
                let client = self.get_gemini_client()?;

                history.push(message);
                let history: Vec<gemini_rust::Message> =
                    history.iter().map(|m| m.to_gemini()).collect();

                let thinking_budget: i32 = match config.thinking {
                    ThinkingMode::None => 0,
                    ThinkingMode::Dynamic => -1,
                    ThinkingMode::Sized(size) => size,
                };

                let function_calling_mode = match self.tools.is_empty() {
                    true => FunctionCallingMode::None,
                    false => FunctionCallingMode::Auto,
                };

                let mut req = client
                    .generate_content()
                    .with_system_instruction(self.system_prompt.clone())
                    .with_messages(history)
                    .with_thinking_budget(0)
                    .with_generation_config(GeminiGenerationConfig {
                        temperature: Some(config.temperature),
                        top_p: Some(config.top_p),
                        top_k: Some(config.top_k),
                        max_output_tokens: Some(config.max_output_tokens),
                        thinking_config: Some(ThinkingConfig {
                            thinking_budget: Some(thinking_budget),
                            ..Default::default()
                        }),
                        candidate_count: Some(config.candidate_count),
                        stop_sequences: config.stop_sequences.clone(),
                        response_mime_type: config.response_mime_type.clone(),
                        response_schema: config.output_schema.clone(),
                        ..Default::default()
                    })
                    .with_function_calling_mode(function_calling_mode);

                if config.output_schema.is_some() {
                    req = req
                        .with_response_mime_type("application/json".to_owned())
                        .with_response_schema(config.output_schema.clone().unwrap());
                }

                for tool in &self.tools {
                    req = req.with_tool(tool.gemini_tool_definition());
                }

                match req.execute().await {
                    Ok(res) => Ok(Message::from_gemini(res)),
                    Err(err) => Err(Box::from(CoreError::Gemini(err))),
                }
            }
            #[cfg(feature = "openai")]
            LLMProvider::OpenAI => {
                let mut client = self.get_openai_client()?;

                // Build messages with system prompt first
                let mut messages: Vec<ChatCompletionMessage> = vec![];
                if !self.system_prompt.is_empty() {
                    messages.push(ChatCompletionMessage {
                        role: openai_api_rs::v1::chat_completion::MessageRole::system,
                        content: openai_api_rs::v1::chat_completion::Content::Text(
                            self.system_prompt.clone(),
                        ),
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                messages.extend(history.iter().map(|m| m.to_openai()));
                messages.push(message.to_openai());

                let req = ChatCompletionRequest::new(self.name.clone(), messages)
                    .max_tokens(config.max_output_tokens as i64)
                    .temperature(config.temperature as f64)
                    .top_p(config.top_p as f64)
                    .n(config.candidate_count as i64)
                    .tools(
                        self.tools
                            .iter()
                            .map(|t| t.openai_tool_definition())
                            .collect(),
                    );

                let res = client.chat_completion(req).await.map_err(
                    |e| -> Box<dyn Error + Send + Sync> {
                        Box::from(CoreError::OpenAI(format!("OpenAI request failed: {:?}", e)))
                    },
                )?;

                if let Some(choice) = res.choices.first() {
                    // Extract function calls if present
                    let function_calls = choice
                        .message
                        .tool_calls
                        .as_ref()
                        .map(|tcs| {
                            tcs.iter()
                                .map(|tc| {
                                    let args_str =
                                        tc.function.arguments.clone().unwrap_or_default();
                                    let arguments =
                                        serde_json::from_str(&args_str).unwrap_or(Value::Null);
                                    FunctionCall {
                                        name: tc.function.name.clone().unwrap_or_default(),
                                        arguments,
                                    }
                                })
                                .collect()
                        })
                        .unwrap_or_default();

                    Ok(Message {
                        role: Role::Assistant,
                        message: choice.message.content.clone(),
                        audio: None,
                        images: None,
                        thinking: choice.message.reasoning_content.clone(),
                        function_calls,
                        function_results: vec![],
                    })
                } else {
                    Err(Box::from(CoreError::Generic(
                        "No choices returned from OpenAI".to_owned(),
                    )))
                }
            }
            #[cfg(feature = "ollama")]
            LLMProvider::Ollama => {
                let mut history: Vec<ChatMessage> = history.iter().map(|m| m.to_ollama()).collect();

                match Ollama::default()
                    .send_chat_messages_with_history(
                        &mut history,
                        ChatMessageRequest::new("model".to_owned(), vec![message.to_ollama()]),
                    )
                    .await
                {
                    Ok(res) => return Ok(Message::from_ollama(res)),
                    Err(err) => Err(Box::from(CoreError::Ollama(err))),
                }
            }
            #[cfg(feature = "anthropic")]
            LLMProvider::Anthropic => Err(Box::from(CoreError::Generic(
                "Anthropic provider not yet implemented".to_owned(),
            ))),
            _ => Err(Box::from(CoreError::Unsupported(
                "Provider not supported for generation".to_owned(),
            ))),
        }
    }

    /// Streams text generation results based on the provided history and message.
    async fn stream(
        &self,
        history: &mut Vec<Message>,
        message: Message,
        config: GenerationConfig,
    ) -> crate::Result<Pin<Box<dyn Stream<Item = crate::Result<Message>> + Send + 'static>>> {
        match self.provider {
            #[cfg(feature = "google")]
            LLMProvider::Google => {
                let client = self.get_gemini_client()?;

                history.push(message);
                let history: Vec<gemini_rust::Message> =
                    history.iter().map(|m| m.to_gemini()).collect();

                let thinking_budget: i32 = match config.thinking {
                    ThinkingMode::None => 0,
                    ThinkingMode::Dynamic => -1,
                    ThinkingMode::Sized(size) => size,
                };

                let function_calling_mode = match self.tools.is_empty() {
                    true => FunctionCallingMode::None,
                    false => FunctionCallingMode::Auto,
                };

                let mut req = client
                    .generate_content()
                    .with_system_instruction(self.system_prompt.clone())
                    .with_messages(history)
                    .with_thinking_budget(0)
                    .with_generation_config(GeminiGenerationConfig {
                        temperature: Some(config.temperature),
                        top_p: Some(config.top_p),
                        top_k: Some(config.top_k),
                        max_output_tokens: Some(config.max_output_tokens),
                        thinking_config: Some(ThinkingConfig {
                            thinking_budget: Some(thinking_budget),
                            ..Default::default()
                        }),
                        candidate_count: Some(config.candidate_count),
                        stop_sequences: config.stop_sequences.clone(),
                        response_mime_type: config.response_mime_type.clone(),
                        response_schema: config.output_schema.clone(),
                        ..Default::default()
                    })
                    .with_function_calling_mode(function_calling_mode);

                for tool in &self.tools {
                    req = req.with_tool(tool.gemini_tool_definition());
                }

                let stream = req.execute_stream().await?;

                let mapped = stream.into_stream().map(|res| match res {
                    Ok(gen_resp) => Ok(Message::from_gemini(gen_resp)),
                    Err(e) => Err(Box::new(CoreError::Gemini(e)) as Box<dyn Error + Send + Sync>),
                });

                Ok(Box::pin(mapped))
            }
            #[cfg(feature = "openai")]
            LLMProvider::OpenAI => {
                let mut client = self.get_openai_client()?;

                // Build messages with system prompt first
                let mut messages: Vec<ChatCompletionMessage> = vec![];
                if !self.system_prompt.is_empty() {
                    messages.push(ChatCompletionMessage {
                        role: openai_api_rs::v1::chat_completion::MessageRole::system,
                        content: openai_api_rs::v1::chat_completion::Content::Text(
                            self.system_prompt.clone(),
                        ),
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                messages.extend(history.iter().map(|m| m.to_openai()));
                messages.push(message.to_openai());

                let req = ChatCompletionStreamRequest::new(self.name.clone(), messages)
                    .max_tokens(config.max_output_tokens as i64)
                    .temperature(config.temperature as f64)
                    .top_p(config.top_p as f64)
                    .n(config.candidate_count as i64)
                    .tools(
                        self.tools
                            .iter()
                            .map(|t| t.openai_tool_definition())
                            .collect(),
                    );

                let stream = client.chat_completion_stream(req).await.map_err(
                    |_| -> Box<dyn Error + Send + Sync> {
                        Box::from(CoreError::OpenAI("OpenAI stream request failed".to_owned()))
                    },
                )?;
                let mapped = stream.map(|res| match res.clone() {
                    ChatCompletionStreamResponse::ToolCall(toolcalls) => {
                        return Ok(Message {
                            role: Role::Assistant,
                            message: None,
                            audio: None,
                            images: None,
                            thinking: None,
                            function_calls: toolcalls
                                .iter()
                                .map(|tc| {
                                    let args_str =
                                        tc.function.arguments.clone().unwrap_or_default();
                                    let arguments =
                                        serde_json::from_str(&args_str).unwrap_or(Value::Null);
                                    FunctionCall {
                                        name: tc.function.name.clone().unwrap_or_default(),
                                        arguments,
                                    }
                                })
                                .collect(),
                            function_results: vec![],
                        });
                    }
                    ChatCompletionStreamResponse::Content(content) => {
                        return Ok(Message {
                            role: Role::Assistant,
                            message: Some(content),
                            audio: None,
                            images: None,
                            thinking: None,
                            function_calls: vec![],
                            function_results: vec![],
                        });
                    }
                    ChatCompletionStreamResponse::Done => {
                        return Ok(Message {
                            role: Role::Assistant,
                            message: None,
                            audio: None,
                            images: None,
                            thinking: None,
                            function_calls: vec![],
                            function_results: vec![],
                        });
                    }
                });
                Ok(Box::pin(mapped))
            }
            #[cfg(feature = "ollama")]
            LLMProvider::Ollama => {
                let _history: Vec<ChatMessage> = history.iter().map(|m| m.to_ollama()).collect();

                match Ollama::default()
                    .send_chat_messages_stream(
                        //&mut history,
                        ChatMessageRequest::new("model".to_owned(), vec![message.to_ollama()]),
                    )
                    .await
                {
                    Ok(res) => {
                        let mapped = res.map(|res| match res {
                            Ok(ollama_msg) => Ok(Message::from_ollama(ollama_msg)),
                            Err(_e) => Err(Box::new(CoreError::Generic(
                                "Ollama stream error".to_owned(),
                            ))
                                as Box<dyn Error + Send + Sync>),
                        });

                        Ok(Box::pin(mapped))
                    }
                    Err(err) => Err(Box::from(CoreError::Ollama(err))),
                }
            }
            #[cfg(feature = "anthropic")]
            LLMProvider::Anthropic => Err(Box::from(CoreError::Generic(
                "Streaming not yet implemented for this provider".to_owned(),
            ))),
            _ => Err(Box::from(CoreError::Unsupported(
                "Provider not supported for streaming".to_owned(),
            ))),
        }
    }

    /// Streams text generation results with audio input.
    async fn stream_with_audio_input(
        &self,
        history: &mut Vec<Message>,
        audio: Vec<u8>,
        config: GenerationConfig,
    ) -> crate::Result<Pin<Box<dyn Stream<Item = crate::Result<Message>> + Send + 'static>>> {
        match self.provider {
            #[cfg(feature = "google")]
            LLMProvider::Google => {
                let client = self.get_gemini_client()?;

                history.push(Message {
                    role: Role::User,
                    message: None,
                    audio: Some(audio),
                    images: None,
                    thinking: None,
                    function_calls: vec![],
                    function_results: vec![],
                });

                let history: Vec<gemini_rust::Message> =
                    history.iter().map(|m| m.to_gemini()).collect();

                let thinking_budget: i32 = match config.thinking {
                    ThinkingMode::None => 0,
                    ThinkingMode::Dynamic => -1,
                    ThinkingMode::Sized(size) => size,
                };

                let function_calling_mode = match self.tools.is_empty() {
                    true => FunctionCallingMode::None,
                    false => FunctionCallingMode::Auto,
                };

                let mut req = client
                    .generate_content()
                    .with_system_instruction(self.system_prompt.clone())
                    .with_messages(history)
                    .with_thinking_budget(0)
                    .with_generation_config(GeminiGenerationConfig {
                        temperature: Some(config.temperature),
                        top_p: Some(config.top_p),
                        top_k: Some(config.top_k),
                        max_output_tokens: Some(config.max_output_tokens),
                        thinking_config: Some(ThinkingConfig {
                            thinking_budget: Some(thinking_budget),
                            ..Default::default()
                        }),
                        candidate_count: Some(config.candidate_count),
                        stop_sequences: config.stop_sequences.clone(),
                        response_mime_type: config.response_mime_type.clone(),
                        response_schema: config.output_schema.clone(),
                        ..Default::default()
                    })
                    .with_function_calling_mode(function_calling_mode);

                for tool in &self.tools {
                    req = req.with_tool(tool.gemini_tool_definition());
                }

                let stream = req.execute_stream().await?;

                let mapped = stream.into_stream().map(|res| match res {
                    Ok(gen_resp) => Ok(Message::from_gemini(gen_resp)),
                    Err(e) => Err(Box::new(CoreError::Gemini(e)) as Box<dyn Error + Send + Sync>),
                });

                Ok(Box::pin(mapped))
            }
            _ => Err(Box::from(CoreError::Unsupported(
                "Streaming with audio input not supported for this provider".to_owned(),
            ))),
        }
    }

    /// Performs text generation with audio input.
    async fn generation_with_audio_input(
        &self,
        history: &mut Vec<Message>,
        audio: Vec<u8>,
        config: GenerationConfig,
    ) -> crate::Result<Message> {
        match self.provider {
            #[cfg(feature = "google")]
            LLMProvider::Google => {
                let client = self.get_gemini_client()?;

                history.push(Message {
                    role: Role::User,
                    message: None,
                    audio: Some(audio),
                    images: None,
                    thinking: None,
                    function_calls: vec![],
                    function_results: vec![],
                });

                let history: Vec<gemini_rust::Message> =
                    history.iter().map(|m| m.to_gemini()).collect();

                let thinking_budget: i32 = match config.thinking {
                    ThinkingMode::None => 0,
                    ThinkingMode::Dynamic => -1,
                    ThinkingMode::Sized(size) => size,
                };

                let function_calling_mode = match self.tools.is_empty() {
                    true => FunctionCallingMode::None,
                    false => FunctionCallingMode::Auto,
                };

                let mut req = client
                    .generate_content()
                    .with_system_instruction(self.system_prompt.clone())
                    .with_messages(history)
                    .with_thinking_budget(0)
                    .with_generation_config(GeminiGenerationConfig {
                        temperature: Some(config.temperature),
                        top_p: Some(config.top_p),
                        top_k: Some(config.top_k),
                        max_output_tokens: Some(config.max_output_tokens),
                        thinking_config: Some(ThinkingConfig {
                            thinking_budget: Some(thinking_budget),
                            ..Default::default()
                        }),
                        candidate_count: Some(config.candidate_count),
                        stop_sequences: config.stop_sequences.clone(),
                        response_mime_type: config.response_mime_type.clone(),
                        response_schema: config.output_schema.clone(),
                        ..Default::default()
                    })
                    .with_function_calling_mode(function_calling_mode);

                for tool in &self.tools {
                    req = req.with_tool(tool.gemini_tool_definition());
                }

                match req.execute().await {
                    Ok(res) => Ok(Message::from_gemini(res)),
                    Err(err) => Err(Box::from(CoreError::Gemini(err))),
                }
            }
            _ => Err(Box::from(CoreError::Unsupported(
                "Generation with audio input not supported for this provider".to_owned(),
            ))),
        }
    }

    /// Generates an embedding for the given text.
    async fn embedding(&self, text: &str, dim: i32) -> crate::Result<Vec<f32>> {
        match self.provider {
            #[cfg(feature = "google")]
            LLMProvider::Google => {
                let client = self.get_gemini_client()?;

                let res = client
                    .embed_content()
                    .with_output_dimensionality(dim)
                    .with_text(text)
                    .execute()
                    .await?;

                Ok(res.embedding.values)
            }
            #[cfg(feature = "openai")]
            LLMProvider::OpenAI => {
                let mut client = self.get_openai_client()?;

                let mut req = openai_api_rs::v1::embedding::EmbeddingRequest::new(
                    self.name.clone(),
                    vec![text.to_owned()],
                );
                req.dimensions = Some(dim);

                let res =
                    client
                        .embedding(req)
                        .await
                        .map_err(|e| -> Box<dyn Error + Send + Sync> {
                            Box::from(CoreError::OpenAI(format!(
                                "OpenAI embedding request failed: {:?}",
                                e
                            )))
                        })?;

                if let Some(data) = res.data.first() {
                    Ok(data.embedding.clone())
                } else {
                    Err(Box::from(CoreError::Generic(
                        "No embedding data returned from OpenAI".to_owned(),
                    )))
                }
            }
            _ => Err(Box::from(CoreError::Unsupported(
                "Embedding not supported for this provider".to_owned(),
            ))),
        }
    }
}
