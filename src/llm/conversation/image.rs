use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Image {
    pub url: String,
    pub text: Option<String>,
}

impl Image {
    pub fn new(url: String, text: Option<String>) -> Self {
        Self { url, text }
    }

    #[cfg(feature = "openai")]
    pub fn from_openai(image: openai_api_rs::v1::chat_completion::ImageUrl) -> Self {
        let url = match image.image_url {
            Some(u) => u.url,
            None => "".to_string(),
        };

        Self {
            url,
            text: image.text,
        }
    }

    #[cfg(feature = "openai")]
    pub fn to_openai(&self) -> openai_api_rs::v1::chat_completion::ImageUrl {
        openai_api_rs::v1::chat_completion::ImageUrl {
            r#type: openai_api_rs::v1::chat_completion::ContentType::image_url,
            image_url: Some(openai_api_rs::v1::chat_completion::ImageUrlType {
                url: self.url.clone(),
            }),
            text: self.text.clone(),
        }
    }

    #[cfg(feature = "ollama")]
    pub fn from_ollama(image: ollama_rs::generation::images::Image) -> Self {
        Self {
            url: image.to_base64().to_string(),
            text: None,
        }
    }

    #[cfg(feature = "ollama")]
    pub fn to_ollama(&self) -> ollama_rs::generation::images::Image {
        ollama_rs::generation::images::Image::from_base64(&self.url)
    }

    #[cfg(feature = "google")]
    pub fn from_gemini(image: gemini_rust::Part) -> crate::Result<Self> {
        if let gemini_rust::Part::InlineData { inline_data } = image {
            match inline_data.mime_type.as_str() {
                "image/png" | "image/jpeg" | "image/jpg" | "image/gif" => Ok(Self {
                    url: format!("data:{};base64,{}", inline_data.mime_type, inline_data.data),
                    text: None,
                }),
                _ => Err(crate::Error::Generic(
                    "Unsupported image MIME type".to_string(),
                )),
            }
        } else {
            Err(crate::Error::Generic(
                "Provided part is not an InlineData part".to_string(),
            ))
        }
    }

    #[cfg(feature = "google")]
    pub fn to_gemini(&self) -> gemini_rust::Part {
        use gemini_rust::Blob;

        let base64_data = if self.url.starts_with("data:") {
            // Extract base64 part from data URL
            self.url
                .split_once(",")
                .map(|x| x.1)
                .unwrap_or("")
                .to_string()
        } else {
            self.url.clone()
        };
        gemini_rust::Part::InlineData {
            inline_data: Blob::new("image/png", base64_data),
        }
    }
}
