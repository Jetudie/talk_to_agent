use anyhow::{Context, Result};
use tracing::{debug, info};

/// Handles text output via keyboard simulation and clipboard.
///
/// Implements the "clipboard paste" pattern:
/// 1. Copy text to clipboard via `arboard`
/// 2. Simulate Ctrl+V via `enigo` to paste into the focused window
///
/// This is far more reliable than character-by-character typing and
/// handles Unicode/CJK characters correctly.
pub struct Typer {
    /// Characters per second for simulated typing (0 = instant paste).
    #[allow(dead_code)]
    typing_speed: u32,
}

impl Typer {
    /// Create a new Typer instance.
    pub fn new(typing_speed: u32) -> Self {
        Self { typing_speed }
    }

    /// Copy text to the system clipboard.
    pub fn copy_to_clipboard(&self, text: &str) -> Result<()> {
        let mut clipboard =
            arboard::Clipboard::new().context("Failed to access system clipboard")?;
        clipboard
            .set_text(text.to_string())
            .context("Failed to set clipboard text")?;
        info!("Copied {} chars to clipboard", text.len());
        Ok(())
    }

    /// Type out text into the currently focused window.
    ///
    /// Uses the clipboard-paste pattern:
    /// 1. Saves current clipboard content
    /// 2. Copies new text to clipboard
    /// 3. Simulates Ctrl+V
    /// 4. Restores original clipboard content
    pub fn type_out(&self, text: &str) -> Result<()> {
        use enigo::{Direction, Enigo, Key, Keyboard, Settings};

        debug!("Type-out: {} chars", text.len());

        // Save current clipboard content
        let mut clipboard =
            arboard::Clipboard::new().context("Failed to access system clipboard")?;
        let original_clipboard = clipboard.get_text().ok();

        // Copy our text to clipboard
        clipboard
            .set_text(text.to_string())
            .context("Failed to set clipboard text")?;

        // Small delay to ensure clipboard is ready
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Simulate Ctrl+V to paste
        let mut enigo = Enigo::new(&Settings::default())
            .map_err(|e| anyhow::anyhow!("Failed to initialize enigo: {:?}", e))?;

        enigo
            .key(Key::Control, Direction::Press)
            .map_err(|e| anyhow::anyhow!("Failed to press Ctrl: {:?}", e))?;
        enigo
            .key(Key::Unicode('v'), Direction::Click)
            .map_err(|e| anyhow::anyhow!("Failed to press V: {:?}", e))?;
        enigo
            .key(Key::Control, Direction::Release)
            .map_err(|e| anyhow::anyhow!("Failed to release Ctrl: {:?}", e))?;

        // Small delay for paste to complete
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Restore original clipboard content
        if let Some(original) = original_clipboard {
            let _ = clipboard.set_text(original);
        }

        info!("Type-out complete: {} chars", text.len());
        Ok(())
    }
}
