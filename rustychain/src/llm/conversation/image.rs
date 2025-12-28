use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_basic_construction() {
        let img = Image::new("https://example.com/i.png".into(), Some("alt text".into()));
        assert_eq!(img.url, "https://example.com/i.png");
        assert_eq!(img.text.unwrap(), "alt text");
    }

    #[cfg(feature = "openai")]
    #[test]
    fn test_openai_image_conversion() {
        use openai_api_rs::v1::chat_completion::{ImageUrl, ImageUrlType};

        let openai_url = ImageUrl {
            r#type: openai_api_rs::v1::chat_completion::ContentType::image_url,
            image_url: Some(ImageUrlType {
                url: "https://openai.com/image.png".into(),
            }),
            text: Some("openai text".into()),
        };

        let internal = Image::from_openai(openai_url);
        assert_eq!(internal.url.clone(), "https://openai.com/image.png");
        assert_eq!(internal.text.clone().unwrap(), "openai text");

        let back_to_openai = internal.to_openai();
        assert_eq!(back_to_openai.image_url.unwrap().url, internal.url);
    }

    #[cfg(feature = "google")]
    #[test]
    fn test_gemini_image_conversion_success() {
        use gemini_rust::{Blob, Part};

        let base64_payload = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
        let part = Part::InlineData {
            inline_data: Blob {
                mime_type: "image/png".into(),
                data: base64_payload.into(),
            },
        };

        let internal = Image::from_gemini(part).expect("Should parse valid InlineData");
        assert!(internal.url.contains("data:image/png;base64,"));
        assert!(internal.url.contains(base64_payload));

        let back_to_gemini = internal.to_gemini();
        if let Part::InlineData { inline_data } = back_to_gemini {
            assert_eq!(inline_data.data, base64_payload);
            assert_eq!(inline_data.mime_type, "image/png");
        } else {
            panic!("Expected InlineData part");
        }
    }

    #[cfg(feature = "google")]
    #[test]
    fn test_gemini_image_conversion_invalid_mime() {
        use gemini_rust::{Blob, Part};

        let part = Part::InlineData {
            inline_data: Blob {
                mime_type: "application/pdf".into(),
                data: "abc".into(),
            },
        };

        let result = Image::from_gemini(part);
        assert!(result.is_err());
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn test_ollama_image_roundtrip() {
        let base64_data = "SGVsbG8=";
        let internal = Image::new(base64_data.into(), None);

        let ollama_img = internal.to_ollama();
        let back_to_internal = Image::from_ollama(ollama_img);

        assert_eq!(back_to_internal.url, internal.url);
    }

    #[test]
    fn test_image_serialization() {
        let img = Image::new("url".into(), None);
        let json = serde_json::to_string(&img).unwrap();
        assert_eq!(json, "{\"url\":\"url\",\"text\":null}");
    }
}
