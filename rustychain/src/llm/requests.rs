#[cfg(feature = "google")]
use gemini_rust::ContentBuilder;
#[cfg(feature = "ollama")]
use ollama_rs::generation::chat::request::ChatMessageRequest;
#[cfg(feature = "openai")]
use openai_api_rs::v1::chat_completion::{
    chat_completion::ChatCompletionRequest, chat_completion_stream::ChatCompletionStreamRequest,
};

#[cfg(feature = "google")]
use crate::{GenerationConfig, Inference};
use crate::{LLM, Role, non_native_function_calling::NonNativeFunctionCallingSchema};

impl LLM {
    #[cfg(feature = "google")]
    pub fn new_google_request(
        &self,
        history: &[Inference],
        inference: &Inference,
        config: &GenerationConfig,
    ) -> crate::Result<ContentBuilder> {
        let client = self.get_gemini_client()?;
        let mut history: Vec<gemini_rust::Message> =
            history.iter().map(|m| m.to_gemini_message()).collect();
        history.push(inference.to_gemini_message());

        let mut req = client
            .generate_content()
            .with_messages(history.clone())
            .with_thinking_budget(config.thinking_mode.to_google())
            .with_generation_config(gemini_rust::GenerationConfig {
                temperature: Some(config.temperature),
                top_p: Some(config.top_p),
                top_k: Some(config.top_k),
                max_output_tokens: Some(config.max_output_tokens),
                thinking_config: Some(gemini_rust::ThinkingConfig {
                    thinking_budget: Some(config.thinking_mode.to_google()),
                    include_thoughts: Some(config.include_thoughts),
                }),
                candidate_count: Some(config.candidate_count),
                ..Default::default()
            });

        // Schema & Tool injection logic
        match (&config.output_schema, config.native_tool_handling) {
            (Some(schema), false) => {
                let wrapped = NonNativeFunctionCallingSchema::new_dynamic_inner(schema.clone());
                req = req
                    .with_response_mime_type("application/json".to_owned())
                    .with_response_schema(serde_json::to_value(wrapped).unwrap_or_default());
            }
            (Some(schema), true) => {
                req = req
                    .with_response_mime_type("application/json".to_owned())
                    .with_response_schema(serde_json::to_value(schema.clone()).unwrap_or_default());
            }
            (None, false) => {
                let default_schema = NonNativeFunctionCallingSchema::new_default();
                req = req
                    .with_response_mime_type("application/json".to_owned())
                    .with_response_schema(serde_json::to_value(default_schema).unwrap_or_default());
            }
            (None, true) => {} // No action needed for native tools without explicit output schema
        }

        if let Some(stop_sequences) = &config.stop_sequences {
            req = req.with_stop_sequences(stop_sequences.clone());
        }

        let mut system_prompt = self.system_prompt.clone();
        if config.native_tool_handling {
            req = req.with_function_calling_mode(config.to_google_tool_calling_mode());
            for tool in &self.tools {
                req = req.with_tool(tool.to_google());
            }
        } else {
            use gemini_rust::FunctionCallingMode;

            req = req.with_function_calling_mode(FunctionCallingMode::None);
            system_prompt.push_str(&format!(
                "\n\nThe following tools are available to you:\n{}\n\n",
                crate::util::tools_to_string(&self.tools)
            ));
        }

        req = req.with_system_prompt(system_prompt);
        Ok(req)
    }

    #[cfg(feature = "openai")]
    pub fn new_openai_request(
        &self,
        history: &[Inference],
        inference: &Inference,
        config: &GenerationConfig,
    ) -> crate::Result<ChatCompletionRequest> {
        use openai_api_rs::v1::chat_completion::ChatCompletionMessage;

        let mut system_prompt = self.system_prompt.clone();
        if !config.native_tool_handling {
            system_prompt.push_str(&format!(
                "\n\nThe following tools are available to you:\n{}\n\n",
                crate::util::tools_to_string(&self.tools)
            ));
        }

        let mut messages: Vec<ChatCompletionMessage> = vec![];
        messages.push(ChatCompletionMessage {
            role: openai_api_rs::v1::chat_completion::MessageRole::system,
            content: openai_api_rs::v1::chat_completion::Content::Text(system_prompt),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });

        for m in history {
            messages.push(m.to_openai_message()?);
        }
        messages.push(inference.to_openai_message()?);

        let mut req = ChatCompletionRequest::new(self.model.clone(), messages)
            .max_tokens(config.max_output_tokens as i64)
            .temperature(config.temperature as f64)
            .top_p(config.top_p as f64)
            .n(config.candidate_count as i64);

        if let Some(reasoning) = config.thinking_mode.to_openai() {
            req = req.reasoning(reasoning);
        }

        if let Some(stop_sequences) = &config.stop_sequences {
            req = req.stop(stop_sequences.clone());
        }

        // Schema & Tool injection logic
        match (&config.output_schema, config.native_tool_handling) {
            (Some(schema), false) => {
                let wrapped = NonNativeFunctionCallingSchema::new_dynamic_inner(schema.clone());
                req = req.response_format(serde_json::to_value(wrapped).unwrap_or_default());
            }
            (Some(schema), true) => {
                req = req.response_format(serde_json::to_value(schema.clone()).unwrap_or_default());
            }
            (None, false) => {
                let default_schema = NonNativeFunctionCallingSchema::new_default();
                req = req.response_format(serde_json::to_value(default_schema).unwrap_or_default());
            }
            (None, true) => {}
        }

        if config.native_tool_handling {
            req = req
                .tools(self.tools.iter().map(|t| t.to_openai()).collect())
                .tool_choice(config.to_openai_tool_calling_mode());
        } else {
            req = req.tool_choice(openai_api_rs::v1::chat_completion::ToolChoiceType::None);
        }

        Ok(req)
    }

    #[cfg(feature = "openai")]
    pub fn new_openai_stream_request(
        &self,
        history: &[Inference],
        inference: &Inference,
        config: &GenerationConfig,
    ) -> crate::Result<ChatCompletionStreamRequest> {
        use openai_api_rs::v1::chat_completion::ChatCompletionMessage;

        let mut system_prompt = self.system_prompt.clone();
        if !config.native_tool_handling {
            system_prompt.push_str(&format!(
                "\n\nThe following tools are available to you:\n{}\n\n",
                crate::util::tools_to_string(&self.tools)
            ));
        }

        let mut messages: Vec<ChatCompletionMessage> = vec![];
        messages.push(ChatCompletionMessage {
            role: openai_api_rs::v1::chat_completion::MessageRole::system,
            content: openai_api_rs::v1::chat_completion::Content::Text(system_prompt),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });

        for m in history {
            messages.push(m.to_openai_message()?);
        }
        messages.push(inference.to_openai_message()?);

        let mut req = ChatCompletionStreamRequest::new(self.model.clone(), messages)
            .max_tokens(config.max_output_tokens as i64)
            .temperature(config.temperature as f64)
            .top_p(config.top_p as f64)
            .n(config.candidate_count as i64);

        if let Some(reasoning) = config.thinking_mode.to_openai() {
            req = req.reasoning(reasoning);
        }

        if let Some(stop_sequences) = &config.stop_sequences {
            req = req.stop(stop_sequences.clone());
        }

        // Schema & Tool injection logic (Stream)
        match (&config.output_schema, config.native_tool_handling) {
            (Some(schema), false) => {
                let wrapped = NonNativeFunctionCallingSchema::new_dynamic_inner(schema.clone());
                req = req.response_format(serde_json::to_value(wrapped).unwrap_or_default());
            }
            (Some(schema), true) => {
                req = req.response_format(serde_json::to_value(schema.clone()).unwrap_or_default());
            }
            (None, false) => {
                let default_schema = NonNativeFunctionCallingSchema::new_default();
                req = req.response_format(serde_json::to_value(default_schema).unwrap_or_default());
            }
            (None, true) => {}
        }

        if config.native_tool_handling {
            req = req
                .tools(self.tools.iter().map(|t| t.to_openai()).collect())
                .tool_choice(config.to_openai_tool_calling_mode());
        } else {
            req = req.tool_choice(openai_api_rs::v1::chat_completion::ToolChoiceType::None);
        }

        Ok(req)
    }

    #[cfg(feature = "ollama")]
    pub fn new_ollama_request(
        &self,
        history: &[Inference],
        inference: &Inference,
        config: &GenerationConfig,
    ) -> crate::Result<ChatMessageRequest> {
        use ollama_rs::{
            generation::{chat::ChatMessage, parameters::JsonStructure},
            models::ModelOptions,
        };

        let mut system_prompt = self.system_prompt.clone();

        // If not using native tool handling, append tool list to system prompt
        if !config.native_tool_handling {
            system_prompt.push_str(&format!(
                "\n\nThe following tools are available to you:\n{}\n\n",
                crate::util::tools_to_string(&self.tools)
            ));
        }

        let mut ollama_messages: Vec<ChatMessage> = vec![];
        ollama_messages.push(ChatMessage::system(system_prompt));
        ollama_messages.extend(history.iter().map(|m| m.to_ollama_message()));
        ollama_messages.push(inference.to_ollama_message());

        let mut req = ChatMessageRequest::new(self.model.clone(), ollama_messages)
            .think(config.thinking_mode.to_ollama())
            .options(
                ModelOptions::default()
                    .top_k(config.top_k as u32)
                    .top_p(config.top_p)
                    .temperature(config.temperature)
                    .stop(config.stop_sequences.clone().unwrap_or_default()),
            );

        // Schema & Tool injection logic
        match (&config.output_schema, config.native_tool_handling) {
            // Case 1: YES schema and NO native tools -> Wrap schema for non-native handling
            (Some(schema), false) => {
                let wrapped_schema =
                    NonNativeFunctionCallingSchema::new_dynamic_inner_draft7(schema.clone());
                req = req.format(
                    ollama_rs::generation::parameters::FormatType::StructuredJson(Box::from(
                        JsonStructure::new_for_schema(wrapped_schema),
                    )),
                );
            }
            // Case 2: YES schema and YES native tools -> Direct injection of both
            (Some(schema), true) => {
                req = req
                    .tools(self.tools.iter().map(|t| t.to_ollama()).collect())
                    .format(
                        ollama_rs::generation::parameters::FormatType::StructuredJson(Box::from(
                            JsonStructure::new_for_schema(
                                crate::non_native_function_calling::clean_schema_for_ollama(
                                    schema.clone(),
                                ),
                            ),
                        )),
                    );
            }
            // Case 3: NO schema and NO native tools -> Default schema wrapping
            (None, false) => {
                let default_schema = NonNativeFunctionCallingSchema::new_default_draft7();
                req = req.format(
                    ollama_rs::generation::parameters::FormatType::StructuredJson(Box::from(
                        JsonStructure::new_for_schema(default_schema),
                    )),
                );
            }
            // Case 4: NO schema and YES native tools -> Just inject tools
            (None, true) => {
                req = req.tools(self.tools.iter().map(|t| t.to_ollama()).collect());
            }
        }

        Ok(req)
    }

    #[cfg(feature = "openrouter")]
    pub fn new_openrouter_request(
        &self,
        history: &[Inference],
        inference: &Inference,
        config: &GenerationConfig,
    ) -> crate::Result<openrouter_rs::api::chat::ChatCompletionRequest> {
        let mut system_prompt = self.system_prompt.clone();
        if !config.native_tool_handling {
            system_prompt.push_str(&format!(
                "\n\nThe following tools are available to you:\n{}\n\n",
                crate::util::tools_to_string(&self.tools)
            ));
        }

        let mut messages: Vec<openrouter_rs::api::chat::Message> = vec![];
        messages.push(openrouter_rs::api::chat::Message {
            role: Role::to_openrouter(&crate::Role::System),
            content: system_prompt,
        });

        for m in history {
            messages.push(m.to_openrouter_message()?);
        }
        messages.push(inference.to_openrouter_message()?);

        let mut req = openrouter_rs::api::chat::ChatCompletionRequest::builder();
        let mut req = req
            .model(self.model.clone())
            .messages(messages)
            .temperature(config.temperature.into())
            .top_p(config.top_p.into())
            .max_tokens(config.max_output_tokens as u32);

        if let Some(thinking_mode) = config.thinking_mode.to_openrouter() {
            req = req.reasoning(thinking_mode);
        }

        //if let Some(stop_sequences) = &config.stop_sequences {
        //  req = req.stop(stop_sequences.clone());
        //}

        /*if config.native_tool_handling {
            req = req. lib not support yet tools
        }*/

        match (&config.output_schema, config.native_tool_handling) {
            (Some(schema), false) => {
                let wrapped = NonNativeFunctionCallingSchema::new_dynamic_inner(schema.clone());
                req = req.response_format(openrouter_rs::types::ResponseFormat::JsonSchema {
                    type_: openrouter_rs::types::ResponseFormatType::JsonSchema,
                    json_schema: openrouter_rs::types::JsonSchemaConfig {
                        name: "schema".to_string(),
                        strict: true,
                        schema: wrapped.to_value(),
                    },
                });
            }
            (Some(schema), true) => {
                req = req.response_format(openrouter_rs::types::ResponseFormat::JsonSchema {
                    type_: openrouter_rs::types::ResponseFormatType::JsonSchema,
                    json_schema: openrouter_rs::types::JsonSchemaConfig {
                        name: "schema".to_string(),
                        strict: true,
                        schema: schema.clone().to_value(),
                    },
                });
            }
            (None, false) => {
                let default_schema = NonNativeFunctionCallingSchema::new_default();
                req = req.response_format(openrouter_rs::types::ResponseFormat::JsonSchema {
                    type_: openrouter_rs::types::ResponseFormatType::JsonSchema,
                    json_schema: openrouter_rs::types::JsonSchemaConfig {
                        name: "schema".to_string(),
                        strict: true,
                        schema: default_schema.to_value(),
                    },
                });
            }
            (None, true) => {}
        }

        req.build().map_err(|e| {
            crate::Error::Internal(format!("OpenRouter request build error: {}", e).into())
        })
    }
}
