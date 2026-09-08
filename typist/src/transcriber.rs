use crate::config::Config;
use anyhow::{Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use std::io::Cursor;
use tracing::{debug, info};

/// HTTP client for calling the ASR (Automatic Speech Recognition) API.
///
/// Supports any OpenAI Whisper-compatible API endpoint.
/// Encodes audio as WAV and sends it via multipart form upload.
pub struct Transcriber {
    client: reqwest::Client,
    api_url: String,
    api_key: String,
    model: String,
    language: String,
}

impl Transcriber {
    /// Create a new transcriber with the given configuration.
    pub fn new(config: &Config) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        info!(
            "Transcriber initialized: {} (model: {}, language: {})",
            config.asr_api_url, config.asr_model, config.asr_language
        );

        Self {
            client,
            api_url: config.asr_api_url.clone(),
            api_key: config.asr_api_key.clone(),
            model: config.asr_model.clone(),
            language: config.asr_language.clone(),
        }
    }

    /// Transcribe a speech segment to text.
    ///
    /// Encodes the f32 PCM samples as a WAV file and sends them to the ASR API.
    /// Returns the transcribed text.
    pub async fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<String> {
        debug!(
            "Transcribing {} samples at {}Hz ({:.1}s of audio)",
            samples.len(),
            sample_rate,
            samples.len() as f32 / sample_rate as f32
        );

        // Encode audio as WAV
        let wav_data = encode_wav(samples, sample_rate)
            .context("Failed to encode audio as WAV")?;

        debug!("Encoded WAV: {} bytes", wav_data.len());

        // Build multipart form
        let file_part = reqwest::multipart::Part::bytes(wav_data)
            .file_name("audio.wav")
            .mime_str("audio/wav")?;

        let mut form = reqwest::multipart::Form::new()
            .part("file", file_part)
            .text("model", self.model.clone());

        // Only add language if it's not "auto"
        if self.language != "auto" && !self.language.is_empty() {
            form = form.text("language", self.language.clone());
        }

        // Send request
        let mut request = self.client.post(&self.api_url).multipart(form);

        // Add authorization header if API key is provided
        if !self.api_key.is_empty() {
            request = request.header("Authorization", format!("Bearer {}", self.api_key));
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

        // Parse response — try JSON first (OpenAI format), then plain text
        let response_text = response.text().await?;

        // Try parsing as JSON (OpenAI Whisper format: {"text": "..."})
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&response_text) {
            if let Some(text) = json.get("text").and_then(|t| t.as_str()) {
                let transcript = text.trim().to_string();
                debug!("Transcript: \"{}\"", transcript);
                return Ok(transcript);
            }
        }

        // Fall back to treating the response as plain text
        let transcript = response_text.trim().to_string();
        debug!("Transcript (plain): \"{}\"", transcript);
        Ok(transcript)
    }
}

/// Encode f32 PCM samples as a WAV file in memory.
///
/// Converts f32 samples (-1.0 to 1.0) to 16-bit PCM and writes a WAV header.
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

    #[test]
    fn test_encode_wav() {
        // Generate a simple sine wave
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

        // WAV header is 44 bytes, plus 2 bytes per sample (16-bit)
        assert_eq!(wav_data.len(), 44 + num_samples * 2);

        // Check RIFF header
        assert_eq!(&wav_data[0..4], b"RIFF");
        assert_eq!(&wav_data[8..12], b"WAVE");
    }
}
