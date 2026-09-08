use anyhow::{Context, Result};
use std::path::PathBuf;

/// Output format for the document.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputFormat {
    PlainText,
    Markdown,
}

impl OutputFormat {
    pub fn file_extension(&self) -> &str {
        match self {
            OutputFormat::PlainText => "txt",
            OutputFormat::Markdown => "md",
        }
    }

    pub fn display_name(&self) -> &str {
        match self {
            OutputFormat::PlainText => "Plain Text",
            OutputFormat::Markdown => "Markdown",
        }
    }
}

/// Application configuration loaded from environment variables.
#[derive(Debug, Clone)]
pub struct Config {
    // ASR (Speech-to-Text) API
    pub asr_api_url: String,
    pub asr_api_key: String,
    pub asr_model: String,
    pub asr_language: String,

    // LLM API
    pub llm_api_key: String,
    pub llm_base_url: String,
    pub llm_model: String,

    // Audio settings
    pub silence_threshold: f32,
    pub silence_duration_secs: f32,
    pub max_record_secs: f32,

    // Output settings
    pub output_format: OutputFormat,
    pub typing_speed_cps: u32,
    pub save_directory: PathBuf,
}

impl Config {
    /// Load configuration from environment variables (and .env file if present).
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        let output_format = match std::env::var("OUTPUT_FORMAT")
            .unwrap_or_else(|_| "markdown".into())
            .to_lowercase()
            .as_str()
        {
            "plain" | "plaintext" | "text" | "txt" => OutputFormat::PlainText,
            _ => OutputFormat::Markdown,
        };

        let save_directory =
            PathBuf::from(std::env::var("SAVE_DIRECTORY").unwrap_or_else(|_| "./output".into()));
        std::fs::create_dir_all(&save_directory)
            .with_context(|| format!("Failed to create save directory: {:?}", save_directory))?;

        Ok(Self {
            // ASR
            asr_api_url: std::env::var("ASR_API_URL")
                .unwrap_or_else(|_| "http://localhost:8080/v1/audio/transcriptions".into()),
            asr_api_key: std::env::var("ASR_API_KEY").unwrap_or_default(),
            asr_model: std::env::var("ASR_MODEL").unwrap_or_else(|_| "whisper-1".into()),
            asr_language: std::env::var("ASR_LANGUAGE").unwrap_or_else(|_| "auto".into()),

            // LLM
            llm_api_key: std::env::var("LLM_API_KEY")
                .context("LLM_API_KEY environment variable is required")?,
            llm_base_url: std::env::var("LLM_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".into()),
            llm_model: std::env::var("LLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into()),

            // Audio
            silence_threshold: std::env::var("SILENCE_THRESHOLD")
                .unwrap_or_else(|_| "0.02".into())
                .parse()
                .unwrap_or(0.02),
            silence_duration_secs: std::env::var("SILENCE_DURATION")
                .unwrap_or_else(|_| "1.5".into())
                .parse()
                .unwrap_or(1.5),
            max_record_secs: std::env::var("MAX_RECORD_SECONDS")
                .unwrap_or_else(|_| "30".into())
                .parse()
                .unwrap_or(30.0),

            // Output
            output_format,
            typing_speed_cps: std::env::var("TYPING_SPEED")
                .unwrap_or_else(|_| "0".into())
                .parse()
                .unwrap_or(0),
            save_directory,
        })
    }
}
