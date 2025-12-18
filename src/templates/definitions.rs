use serde::Serialize;

/// Associated context trait to ensure a struct is valid for templating
pub trait TemplateContext: Serialize {}

pub trait PromptTemplate {
    /// The specific struct associated with this template
    type Context: TemplateContext;

    /// The raw string containing placeholders like {{variable}}
    fn raw(&self) -> &str;

    /// Compiles the template and validates that all placeholders were replaced
    fn compile(&self, context: &Self::Context) -> crate::Result<String> {
        let value = serde_json::to_value(context)?;

        let mut result = self.raw().to_string();

        if let serde_json::Value::Object(map) = value {
            for (key, val) in map {
                let placeholder = format!("{{{{{}}}}}", key);

                // Convert JSON value to a clean string
                let replacement = match val {
                    serde_json::Value::String(s) => s,
                    serde_json::Value::Null => "".to_string(),
                    _ => val.to_string().trim_matches('"').to_string(),
                };

                result = result.replace(&placeholder, &replacement);
            }
        }

        // Safety Check: Ensure no {{placeholder}} remains in the output
        if result.contains("{{") && result.contains("}}") {
            return Err(crate::Error::Input(
                "Template rendering incomplete: some placeholders were not found in the context struct".to_string()
            ));
        }

        Ok(result)
    }
}
