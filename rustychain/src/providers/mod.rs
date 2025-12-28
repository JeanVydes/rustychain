use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum LLMProvider {
    Google,
    OpenAI,
    Anthropic,
    Ollama,
    Bedrock,
    OpenRouter,
}
