use crate::config::Config;
use crate::events::AppEvent;
use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{SampleFormat, WavReader};
use minimp3::{Decoder, Error as Mp3Error};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// The source from which Typist receives audio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioSource {
    Microphone,
    /// Loopback capture of the default speaker output on Windows.
    SystemAudio,
    File(PathBuf),
}

impl AudioSource {
    pub fn display_name(&self) -> String {
        match self {
            Self::Microphone => "Microphone".into(),
            Self::SystemAudio => "System audio".into(),
            Self::File(path) => format!(
                "File: {}",
                path.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("audio file")
            ),
        }
    }

    pub fn is_live(&self) -> bool {
        !matches!(self, Self::File(_))
    }
}

/// Decoded mono audio loaded from a WAV or MP3 file.
pub struct AudioFile {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Load a WAV or MP3 file and downmix it to mono.
pub fn load_audio_file(path: &Path) -> Result<AudioFile> {
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("mp3"))
    {
        return load_mp3(path);
    }
    load_wav(path)
}

fn load_mp3(path: &Path) -> Result<AudioFile> {
    let file = File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mut decoder = Decoder::new(file);
    let mut samples = Vec::new();
    let mut sample_rate = None;
    loop {
        let frame = match decoder.next_frame() {
            Ok(frame) => frame,
            Err(Mp3Error::Eof) => break,
            Err(e) => return Err(e).context("Failed to decode MP3 audio"),
        };
        let rate = frame.sample_rate as u32;
        if sample_rate.is_some_and(|previous| previous != rate) {
            anyhow::bail!("MP3 sample rate changes within {}", path.display());
        }
        sample_rate = Some(rate);
        let pcm: Vec<f32> = frame
            .data
            .iter()
            .map(|&sample| sample as f32 / 32768.0)
            .collect();
        samples.extend(to_mono_f32(&pcm, frame.channels));
    }
    let sample_rate = sample_rate.context("MP3 file contains no decodable audio")?;
    info!(
        "Loaded MP3 file {}: {}Hz, {:.1}s",
        path.display(),
        sample_rate,
        samples.len() as f32 / sample_rate as f32
    );
    Ok(AudioFile {
        samples,
        sample_rate,
    })
}

fn load_wav(path: &Path) -> Result<AudioFile> {
    let mut reader = WavReader::open(path)
        .with_context(|| format!("Failed to open audio file: {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels == 0 || spec.sample_rate == 0 {
        anyhow::bail!("Audio file has an invalid WAV format: {}", path.display());
    }

    let interleaved = match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .map(|sample| sample.context("Failed to decode float WAV sample"))
            .collect::<Result<Vec<_>>>()?,
        SampleFormat::Int => {
            let scale = 2_f32.powi(spec.bits_per_sample.saturating_sub(1) as i32);
            reader
                .samples::<i32>()
                .map(|sample| {
                    sample
                        .map(|value| (value as f32 / scale).clamp(-1.0, 1.0))
                        .context("Failed to decode PCM WAV sample")
                })
                .collect::<Result<Vec<_>>>()?
        }
    };

    let samples = to_mono_f32(&interleaved, channels);
    if samples.is_empty() {
        anyhow::bail!("Audio file contains no samples: {}", path.display());
    }

    info!(
        "Loaded audio file {}: {}Hz, {} channel(s), {:.1}s",
        path.display(),
        spec.sample_rate,
        channels,
        samples.len() as f32 / spec.sample_rate as f32
    );

    Ok(AudioFile {
        samples,
        sample_rate: spec.sample_rate,
    })
}

/// Live audio capture using cpal with integrated Voice Activity Detection (VAD).
///
/// Captures microphone or system output, detects speech boundaries using RMS-based VAD,
/// and sends complete speech segments to the processing pipeline.
pub struct AudioCapture {
    /// The cpal stream — kept alive to maintain audio capture.
    _stream: cpal::Stream,
    /// Flag to pause/resume listening.
    is_listening: Arc<AtomicBool>,
    is_active: Arc<AtomicBool>,
    /// The sample rate the audio was captured at.
    pub sample_rate: u32,
}

impl AudioCapture {
    /// Create and start a new audio capture instance.
    ///
    /// Opens the requested input or playback device, starts streaming audio,
    /// and spawns a VAD processor task that detects speech segments.
    pub fn start(
        source: &AudioSource,
        config: &Config,
        event_tx: mpsc::UnboundedSender<AppEvent>,
    ) -> Result<Self> {
        let host = cpal::default_host();
        let device = match source {
            AudioSource::Microphone => {
                host.default_input_device().context("No microphone found")?
            }
            AudioSource::SystemAudio => {
                #[cfg(target_os = "windows")]
                {
                    host.default_output_device()
                        .context("No speaker output found")?
                }
                #[cfg(not(target_os = "windows"))]
                {
                    anyhow::bail!("System audio capture is currently supported on Windows");
                }
            }
            AudioSource::File(_) => anyhow::bail!("File is not a live audio source"),
        };

        let device_name = device.name().unwrap_or_else(|_| "Unknown".into());
        info!("Using audio input device: {}", device_name);

        let supported_config = if matches!(source, AudioSource::SystemAudio) {
            device
                .default_output_config()
                .context("Failed to get speaker output config")?
        } else {
            device
                .default_input_config()
                .context("Failed to get microphone input config")?
        };

        let sample_rate = supported_config.sample_rate().0;
        let channels = supported_config.channels() as usize;
        let sample_format = supported_config.sample_format();

        info!(
            "Audio config: {}Hz, {} channel(s), {:?}",
            sample_rate, channels, sample_format
        );

        let is_listening = Arc::new(AtomicBool::new(true));
        let is_active = Arc::new(AtomicBool::new(true));
        let is_listening_clone = is_listening.clone();

        // Channel to send audio chunks from cpal callback to VAD processor.
        // Using tokio's UnboundedSender which has a sync send() method,
        // safe to call from cpal's non-async audio callback thread.
        let (chunk_tx, chunk_rx) = mpsc::unbounded_channel::<Vec<f32>>();

        let stream_config: cpal::StreamConfig = supported_config.into();

        let error_callback = |err: cpal::StreamError| {
            error!("Audio stream error: {}", err);
        };

        let stream = match sample_format {
            cpal::SampleFormat::F32 => {
                let chunk_tx = chunk_tx.clone();
                let is_listening = is_listening_clone.clone();
                device.build_input_stream(
                    &stream_config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        if !is_listening.load(Ordering::Relaxed) {
                            return;
                        }
                        let mono = to_mono_f32(data, channels);
                        let _ = chunk_tx.send(mono);
                    },
                    error_callback,
                    None,
                )?
            }
            cpal::SampleFormat::I16 => {
                let chunk_tx = chunk_tx.clone();
                let is_listening = is_listening_clone.clone();
                device.build_input_stream(
                    &stream_config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        if !is_listening.load(Ordering::Relaxed) {
                            return;
                        }
                        let float_data: Vec<f32> =
                            data.iter().map(|&s| s as f32 / 32768.0).collect();
                        let mono = to_mono_f32(&float_data, channels);
                        let _ = chunk_tx.send(mono);
                    },
                    error_callback,
                    None,
                )?
            }
            cpal::SampleFormat::U16 => {
                let chunk_tx = chunk_tx.clone();
                let is_listening = is_listening_clone.clone();
                device.build_input_stream(
                    &stream_config,
                    move |data: &[u16], _: &cpal::InputCallbackInfo| {
                        if !is_listening.load(Ordering::Relaxed) {
                            return;
                        }
                        let float_data: Vec<f32> =
                            data.iter().map(|&s| (s as f32 / 32768.0) - 1.0).collect();
                        let mono = to_mono_f32(&float_data, channels);
                        let _ = chunk_tx.send(mono);
                    },
                    error_callback,
                    None,
                )?
            }
            format => {
                anyhow::bail!("Unsupported audio sample format: {:?}", format);
            }
        };

        stream.play().context("Failed to start audio stream")?;
        info!("Audio capture started");

        // Spawn the VAD processor as a tokio task
        let vad_config = VadConfig {
            silence_threshold: config.silence_threshold,
            silence_duration_samples: (config.silence_duration_secs * sample_rate as f32) as usize,
            max_samples: (config.max_record_secs * sample_rate as f32) as usize,
            min_speech_samples: (0.1 * sample_rate as f32) as usize, // min 100ms of speech
        };

        tokio::spawn(vad_processor(
            chunk_rx,
            event_tx,
            vad_config,
            sample_rate,
            is_active.clone(),
        ));

        Ok(Self {
            _stream: stream,
            is_listening,
            is_active,
            sample_rate,
        })
    }

    /// Check if audio capture is currently listening.
    #[allow(dead_code)]
    pub fn is_listening(&self) -> bool {
        self.is_listening.load(Ordering::Relaxed)
    }

    /// Toggle listening on/off.
    pub fn toggle_listening(&self) -> bool {
        let was_listening = self.is_listening.load(Ordering::Relaxed);
        self.is_listening.store(!was_listening, Ordering::Relaxed);
        let now_listening = !was_listening;
        if now_listening {
            info!("Audio capture resumed");
        } else {
            info!("Audio capture paused");
        }
        now_listening
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        self.is_active.store(false, Ordering::Relaxed);
    }
}

/// Convert multi-channel audio to mono by averaging channels.
fn to_mono_f32(data: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return data.to_vec();
    }
    data.chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{WavSpec, WavWriter};

    #[test]
    fn loads_and_downmixes_stereo_wav() {
        let path = std::env::temp_dir().join(format!(
            "typist-audio-test-{}-{}.wav",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let spec = WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(&path, spec).unwrap();
        writer.write_sample(16_384_i16).unwrap();
        writer.write_sample(-16_384_i16).unwrap();
        writer.write_sample(8_192_i16).unwrap();
        writer.write_sample(8_192_i16).unwrap();
        writer.finalize().unwrap();

        let audio = load_audio_file(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.samples.len(), 2);
        assert!(audio.samples[0].abs() < 0.0001);
        assert!((audio.samples[1] - 0.25).abs() < 0.0001);
    }
}

/// Compute the Root Mean Square (RMS) energy of an audio buffer.
fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_squares / samples.len() as f32).sqrt()
}

/// Configuration for the Voice Activity Detection processor.
struct VadConfig {
    /// RMS threshold above which we consider audio as speech.
    silence_threshold: f32,
    /// Number of consecutive silent samples needed to end a speech segment.
    silence_duration_samples: usize,
    /// Maximum number of samples in a single speech segment.
    max_samples: usize,
    /// Minimum number of speech samples to consider it valid (rejects noise bursts).
    min_speech_samples: usize,
}

/// VAD processor task that runs asynchronously.
///
/// Reads audio chunks from the channel, detects speech boundaries,
/// and sends complete speech segments as events.
async fn vad_processor(
    mut chunk_rx: mpsc::UnboundedReceiver<Vec<f32>>,
    event_tx: mpsc::UnboundedSender<AppEvent>,
    config: VadConfig,
    sample_rate: u32,
    is_active: Arc<AtomicBool>,
) {
    let mut speech_buffer: Vec<f32> = Vec::with_capacity(sample_rate as usize * 5);
    let mut is_speaking = false;
    let mut silence_sample_count: usize = 0;
    let mut speech_sample_count: usize = 0;

    debug!("VAD processor started");

    while let Some(chunk) = chunk_rx.recv().await {
        if !is_active.load(Ordering::Relaxed) {
            break;
        }
        let rms = compute_rms(&chunk);

        // Send audio level for VU meter (throttle to avoid flooding)
        let _ = event_tx.send(AppEvent::AudioLevelUpdate(rms));

        if rms > config.silence_threshold {
            // Speech detected
            if !is_speaking {
                is_speaking = true;
                speech_sample_count = 0;
                debug!("Speech started (RMS: {:.4})", rms);
                let _ = event_tx.send(AppEvent::SpeechStarted);
            }
            silence_sample_count = 0;
            speech_sample_count += chunk.len();
            speech_buffer.extend_from_slice(&chunk);
        } else if is_speaking {
            // Was speaking, now silent — accumulate silence
            silence_sample_count += chunk.len();
            speech_buffer.extend_from_slice(&chunk);

            if silence_sample_count >= config.silence_duration_samples {
                // Speech ended — send the segment if it's long enough
                if speech_sample_count >= config.min_speech_samples {
                    let segment = std::mem::take(&mut speech_buffer);
                    debug!(
                        "Speech segment ready: {} samples ({:.1}s)",
                        segment.len(),
                        segment.len() as f32 / sample_rate as f32
                    );
                    let _ = event_tx.send(AppEvent::SpeechSegmentReady {
                        samples: segment,
                        sample_rate,
                    });
                } else {
                    debug!("Speech segment too short, discarding");
                    speech_buffer.clear();
                }
                is_speaking = false;
                silence_sample_count = 0;
                speech_sample_count = 0;
            }

            // Check max recording length
            if speech_buffer.len() >= config.max_samples {
                warn!("Max recording length reached, sending segment");
                let segment = std::mem::take(&mut speech_buffer);
                let _ = event_tx.send(AppEvent::SpeechSegmentReady {
                    samples: segment,
                    sample_rate,
                });
                is_speaking = false;
                speech_sample_count = 0;
            }
        }
        // If not speaking and below threshold, discard ambient noise
    }

    debug!("VAD processor stopped");
}
