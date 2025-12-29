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
            name: None,
            tool_call_id: None,
            tool_calls: None,
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

        /*if let Some(stop_sequences) = &config.stop_sequences {
          req = req.
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

        if config.native_tool_handling {
            req = req
                .tools(
                    self.tools
                        .iter()
                        .map(|t| t.to_openrouter())
                        .collect::<Vec<_>>(),
                )
                .tool_choice(config.to_openrouter_tool_calling_mode());
        }

        req.build().map_err(|e| {
            crate::Error::Internal(format!("OpenRouter request build error: {}", e).into())
        })
    }
}
