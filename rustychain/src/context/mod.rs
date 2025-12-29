use std::sync::Arc;

use crate::{Inference, prelude::InferenceContent};

/// # Context
///
/// the context is ur most valuable assets when dealing with LLMs
/// a larger context windows differentiates production ready agents from shitty prototypes
/// but sometimes, even with a large context window, the relevant information is buried deep inside
/// so, there is some techniques to improve the context management
/// - summarization: periodically summarize the conversation history to condense information, this is useful
/// but usually relay in other model to do it, this add latency and cost
/// - retrieval-augmented generation: store past interactions in a vector database, and retrieve relevant
/// documents based on similarity to the current query, the problem is that need a good prompting and agent understanding
/// of how and when to use the retrieved documents
/// - delete the turns too far in the past: this is the simplest approach, but can lead to loss of important context
/// - hierarchical context management: organize the conversation into topics or threads, allowing the model to focus
/// on the most relevant parts of the conversation, this is complex to implement but can be very effective
/// - selective context inclusion: only include the most relevant parts of the conversation history based on
/// the current query, this can be done by using heuristics or another model to identify relevant information
/// - selective context summarization: instead of summarizing the entire conversation history,
/// selectively summarize only the most relevant parts based on the current query, this can be done by using other models
/// or more strategically (idk how write that word, im not a native english speaker) choosing which parts to summarize
/// the most effective ones, are to shrink tools results, that are usually very verbose, and can be implemented in conjuntion
/// do it progressively, based on how far the function results are in the past, so important recent results are not summarized too early
/// - and other that idk, im not a phd
///
/// todo:
///
/// implement context strategies, as part of a context manager, to apply them, also do it hierarchically, by layers
/// for example, the first layer is the raw conversation history, the second layer is the summarized history,
/// the third layer is the retrieved documents, etc
/// this context manager can be used by the LLM during generation, to build the final context to send to the model
/// also implement context manager into agents
pub struct ContextManager {}

impl ContextManager {
    pub async fn trim_by_n_turns(
        &self,
        history: &mut Vec<Inference>,
        n: usize,
    ) -> crate::Result<()> {
        if history.len() > n {
            history.drain(0..history.len() - n);
        }
        Ok(())
    }

    pub async fn summary(
        &self,
        history: &mut Vec<Inference>,
        llm: Arc<crate::LLM>,
    ) -> crate::Result<()> {
        let prompt = format!(
            "Summarize the following conversation between a user and an AI assistant in a concise manner, \
            focusing on the key points and important information exchanged:\n\n{}",
            history
                .iter()
                .map(|inf| format!("{:?}: {:?}", inf.content.role, inf.content.text))
                .collect::<Vec<String>>()
                .join("\n")
        );

        let summary = llm.inference(prompt.as_str()).generate().await?;

        history.clear();
        history.push(Inference {
            content: InferenceContent {
                role: crate::Role::Assistant,
                text: summary.content.text,
                audio: None,
                images: None,
            },
            ..Default::default()
        });

        Ok(())
    }
}
