use crate::config::{AsrBackend, Config};
use anyhow::{Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, info};

/// Backend used for speech-to-text transcription.
enum TranscriberBackend {
    Remote {
        client: reqwest::Client,
        api_url: String,
        api_key: String,
        model: String,
        language: String,
    },
    Local {
        model: Arc<oxiwhisper::WhisperModel>,
        model_name: String,
        language: String,
    },
}

/// Transcriber supporting both local Whisper (via oxiwhisper) and remote ASR APIs.
pub struct Transcriber {
    backend: TranscriberBackend,
}

impl Transcriber {
    /// Create a new transcriber with the given configuration.
    ///
    /// If `config.asr_backend` is `Local`, loads or downloads the lightweight Whisper model.
    /// If `config.asr_backend` is `Remote`, initializes the HTTP client for the ASR API.
    pub async fn new(config: &Config) -> Result<Self> {
        match config.asr_backend {
            AsrBackend::Remote => {
                let client = reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(30))
                    .build()
                    .context("Failed to create HTTP client for ASR")?;

                info!(
                    "Transcriber initialized (Remote API): {} (model: {}, language: {})",
                    config.asr_api_url, config.asr_model, config.asr_language
                );

                Ok(Self {
                    backend: TranscriberBackend::Remote {
                        client,
                        api_url: config.asr_api_url.clone(),
                        api_key: config.asr_api_key.clone(),
                        model: config.asr_model.clone(),
                        language: config.asr_language.clone(),
                    },
                })
            }
            AsrBackend::Local => {
                info!(
                    "Initializing local Whisper ASR: model='{}', language='{}'",
                    config.whisper_model, config.asr_language
                );

                let model_path = ensure_model_file(
                    &config.whisper_model,
                    config.local_whisper_model_path.as_ref(),
                )
                .await?;

                info!("Loading local Whisper model from {:?}...", model_path);
                let model = tokio::task::spawn_blocking(move || {
                    oxiwhisper::WhisperModel::from_file(&model_path)
                })
                .await
                .context("Task join error while loading Whisper model")?
                .context("Failed to load local Whisper model")?;

                info!("Local Whisper model loaded successfully.");

                Ok(Self {
                    backend: TranscriberBackend::Local {
                        model: Arc::new(model),
                        model_name: config.whisper_model.clone(),
                        language: config.asr_language.clone(),
                    },
                })
            }
        }
    }

    /// Return a human-readable description of the active ASR backend.
    pub fn backend_name(&self) -> String {
        match &self.backend {
            TranscriberBackend::Remote { model, .. } => format!("API ({})", model),
            TranscriberBackend::Local { model_name, .. } => {
                format!("Local Whisper ({})", model_name)
            }
        }
    }

    /// Transcribe a speech segment to text.
    pub async fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<String> {
        match &self.backend {
            TranscriberBackend::Remote {
                client,
                api_url,
                api_key,
                model,
                language,
            } => {
                debug!(
                    "Transcribing (Remote API) {} samples at {}Hz ({:.1}s of audio)",
                    samples.len(),
                    sample_rate,
                    samples.len() as f32 / sample_rate as f32
                );

                let wav_data = encode_wav(samples, sample_rate)
                    .context("Failed to encode audio as WAV")?;

                let file_part = reqwest::multipart::Part::bytes(wav_data)
                    .file_name("audio.wav")
                    .mime_str("audio/wav")?;

                let mut form = reqwest::multipart::Form::new()
                    .part("file", file_part)
                    .text("model", model.clone());

                if language != "auto" && !language.is_empty() {
                    form = form.text("language", language.clone());
                }

                let mut request = client.post(api_url).multipart(form);
                if !api_key.is_empty() {
                    request = request.header("Authorization", format!("Bearer {}", api_key));
                }

                let response = request
                    .send()
                    .await
                    .context("Failed to send request to ASR API")?;

                let status = response.status();
                if !status.is_success() {
                    let error_body = response
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".into());
                    anyhow::bail!("ASR API returned error {}: {}", status, error_body);
                }

                let response_text = response.text().await?;
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&response_text) {
                    if let Some(text) = json.get("text").and_then(|t| t.as_str()) {
                        let transcript = text.trim().to_string();
                        debug!("Transcript: \"{}\"", transcript);
                        return Ok(transcript);
                    }
                }

                let transcript = response_text.trim().to_string();
                debug!("Transcript (plain): \"{}\"", transcript);
                Ok(transcript)
            }
            TranscriberBackend::Local { model, language, .. } => {
                debug!(
                    "Transcribing (Local Whisper) {} samples at {}Hz ({:.1}s of audio)",
                    samples.len(),
                    sample_rate,
                    samples.len() as f32 / sample_rate as f32
                );

                let samples_16k = resample_to_16k(samples, sample_rate);
                let model = model.clone();
                let lang_opt = if language == "auto" || language.is_empty() {
                    None
                } else {
                    Some(language.clone())
                };

                let text = tokio::task::spawn_blocking(move || {
                    let opts = oxiwhisper::TranscribeOptions {
                        language: lang_opt.as_deref(),
                        suppress_blank: true,
                        no_speech_threshold: 0.6,
                        ..Default::default()
                    };
                    model.transcribe(&samples_16k, &opts)
                })
                .await
                .context("Local transcription task join failed")?
                .context("Local Whisper transcription failed")?;

                let trimmed = text.trim();
                // If Whisper hallucinated pure punctuation on low energy / silence, return empty
                if trimmed.chars().all(|c| c.is_ascii_punctuation() || c.is_whitespace()) {
                    return Ok(String::new());
                }

                debug!("Local Whisper transcript: \"{}\"", trimmed);
                Ok(trimmed.to_string())
            }
        }
    }
}

/// Locate an existing model file, or automatically download it from Hugging Face.
async fn ensure_model_file(
    model_name: &str,
    custom_path: Option<&PathBuf>,
) -> Result<PathBuf> {
    if let Some(path) = custom_path {
        if path.exists() {
            return Ok(path.clone());
        } else {
            anyhow::bail!("Configured LOCAL_WHISPER_MODEL_PATH does not exist: {:?}", path);
        }
    }

    let normalized = model_name.trim().to_lowercase();
    let filename = if normalized.ends_with(".bin") {
        normalized
    } else {
        format!("ggml-{}.bin", normalized)
    };

    // Candidate search paths
    let candidates = [
        PathBuf::from("models").join(&filename),
        PathBuf::from("../models").join(&filename),
        dirs_cache_path(&filename),
    ];

    for path in &candidates {
        if path.exists() {
            info!("Found local Whisper model at {:?}", path);
            return Ok(path.clone());
        }
    }

    // Not found — download to models/ directory
    let target_path = PathBuf::from("models").join(&filename);
    if let Some(parent) = target_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let url = format!(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
        filename
    );

    info!("Downloading local Whisper model from {} to {:?}...", url, target_path);
    println!("\nDownloading local Whisper model ({}) from Hugging Face... Please wait.", filename);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;

    let response = client
        .get(&url)
        .send()
        .await
        .context(format!("Failed to download Whisper model from {}", url))?;

    if !response.status().is_success() {
        anyhow::bail!(
            "Failed to download Whisper model: HTTP {} from {}",
            response.status(),
            url
        );
    }

    let bytes = response
        .bytes()
        .await
        .context("Failed to read model bytes from download response")?;

    std::fs::write(&target_path, &bytes)
        .context(format!("Failed to save downloaded model to {:?}", target_path))?;

    info!(
        "Downloaded Whisper model to {:?} ({} bytes)",
        target_path,
        bytes.len()
    );
    println!("Model downloaded successfully.\n");

    Ok(target_path)
}

/// Get the user cache directory path for a model file.
fn dirs_cache_path(filename: &str) -> PathBuf {
    if let Ok(home) = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")) {
        PathBuf::from(home)
            .join(".cache")
            .join("typist")
            .join("models")
            .join(filename)
    } else {
        PathBuf::from("models").join(filename)
    }
}

/// Resample audio samples to 16,000 Hz using linear interpolation.
pub fn resample_to_16k(samples: &[f32], from_rate: u32) -> Vec<f32> {
    if from_rate == 16000 {
        return samples.to_vec();
    }
    if from_rate == 0 || samples.is_empty() {
        return Vec::new();
    }
    let ratio = 16000.0 / from_rate as f64;
    let target_len = (samples.len() as f64 * ratio).round() as usize;
    let mut resampled = Vec::with_capacity(target_len);
    for i in 0..target_len {
        let src_idx = i as f64 / ratio;
        let idx0 = src_idx.floor() as usize;
        let idx1 = (idx0 + 1).min(samples.len() - 1);
        let frac = (src_idx - idx0 as f64) as f32;
        let val = samples[idx0] * (1.0 - frac) + samples[idx1] * frac;
        resampled.push(val);
    }
    resampled
}

/// Encode f32 PCM samples as a WAV file in memory.
fn encode_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut cursor = Cursor::new(Vec::with_capacity(samples.len() * 2 + 44));
    {
        let mut writer = WavWriter::new(&mut cursor, spec)?;
        for &sample in samples {
            let clamped = sample.clamp(-1.0, 1.0);
            let sample_i16 = (clamped * 32767.0) as i16;
            writer.write_sample(sample_i16)?;
        }
        writer.finalize()?;
    }

    Ok(cursor.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_encode_wav() {
        let sample_rate = 16000;
        let duration_secs = 0.1;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
            })
            .collect();

        let wav_data = encode_wav(&samples, sample_rate).unwrap();

        assert_eq!(wav_data.len(), 44 + num_samples * 2);
        assert_eq!(&wav_data[0..4], b"RIFF");
        assert_eq!(&wav_data[8..12], b"WAVE");
    }

    #[test]
    fn test_resample_to_16k() {
        let samples = vec![1.0; 48000];
        let resampled = resample_to_16k(&samples, 48000);
        assert_eq!(resampled.len(), 16000);
        for &val in &resampled {
            assert!((val - 1.0).abs() < 1e-4);
        }

        let samples_16k = vec![0.5; 16000];
        let identity = resample_to_16k(&samples_16k, 16000);
        assert_eq!(identity.len(), 16000);
        assert_eq!(identity, samples_16k);
    }

    #[test]
    fn test_oxiwhisper_inference() {
        let model_path = Path::new("models/ggml-tiny.bin");
        if !model_path.exists() {
            return;
        }

        let model = oxiwhisper::WhisperModel::from_file(model_path).expect("Failed to load WhisperModel");
        let samples = vec![0.0f32; 16000];
        let opts = oxiwhisper::TranscribeOptions::default();
        let result = model.transcribe(&samples, &opts);
        assert!(result.is_ok());
    }
}
