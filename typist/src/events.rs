use serde::{Deserialize, Serialize};

/// All events flowing through the application.
#[derive(Debug, Clone)]
pub enum AppEvent {
    /// Real-time audio RMS level for VU meter display.
    AudioLevelUpdate(f32),

    /// Voice activity detected — user started speaking.
    SpeechStarted,

    /// A complete speech segment is ready for transcription.
    /// Contains f32 PCM samples and the sample rate they were captured at.
    SpeechSegmentReady {
        samples: Vec<f32>,
        sample_rate: u32,
    },

    /// ASR transcription is in progress.
    Transcribing,

    /// Raw transcript received from ASR API.
    TranscriptReady {
        text: String,
    },

    /// LLM intent classification is in progress.
    ClassifyingIntent,

    /// LLM has classified the intent and returned an action.
    LlmResponse(LlmAction),

    /// Status message for display in the UI.
    #[allow(dead_code)]
    StatusMessage(String),

    /// An error occurred in the pipeline.
    Error(String),
}

/// The LLM's classified intent for a user utterance.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "intent", rename_all = "snake_case")]
pub enum LlmAction {
    /// User is dictating content to be typed into the document.
    Dictate {
        text: String,
    },

    /// User is giving an editing command.
    Command {
        action: EditCommand,
        /// Optional text parameter for the command (e.g., heading text).
        #[serde(default)]
        text: Option<String>,
    },

    /// User is having a conversation (not dictating).
    Conversation {
        response: String,
    },
}

/// Editing commands that can be applied to the document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum EditCommand {
    /// Start a new paragraph.
    NewParagraph,
    /// Delete the last sentence from the current paragraph.
    DeleteLastSentence,
    /// Delete the last paragraph entirely.
    DeleteLastParagraph,
    /// Undo the last document change.
    Undo,
    /// Redo the last undone change.
    Redo,
    /// Clear the entire document.
    ClearAll,
    /// Insert a heading (text provided separately).
    InsertHeading,
    /// Insert a bullet point (text provided separately).
    InsertBullet,
}
