use crate::config::OutputFormat;
use crate::events::{EditCommand, LlmAction};
use anyhow::Result;
use std::path::Path;

/// A snapshot of the document state for undo/redo.
#[derive(Debug, Clone)]
struct Snapshot {
    paragraphs: Vec<String>,
}

/// The document buffer that accumulates typed text.
///
/// Maintains a list of paragraphs with full undo/redo support.
/// Can render as plain text or Markdown depending on the configured format.
#[derive(Debug)]
pub struct Document {
    /// The document content as a list of paragraphs.
    paragraphs: Vec<String>,
    /// Undo history (previous states).
    undo_stack: Vec<Snapshot>,
    /// Redo stack (states undone).
    redo_stack: Vec<Snapshot>,
    /// Output format (plain text or Markdown).
    pub format: OutputFormat,
    /// Maximum undo history size.
    max_history: usize,
}

impl Document {
    /// Create a new empty document with the given output format.
    pub fn new(format: OutputFormat) -> Self {
        Self {
            paragraphs: vec![String::new()],
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            format,
            max_history: 100,
        }
    }

    /// Save the current state to the undo stack before making changes.
    fn save_snapshot(&mut self) {
        self.undo_stack.push(Snapshot {
            paragraphs: self.paragraphs.clone(),
        });
        // Trim undo history if too large
        if self.undo_stack.len() > self.max_history {
            self.undo_stack.remove(0);
        }
        // Clear redo stack since we're making a new change
        self.redo_stack.clear();
    }

    /// Apply an LLM action to the document.
    pub fn apply_action(&mut self, action: &LlmAction) {
        match action {
            LlmAction::Dictate { text } => {
                self.append_text(text);
            }
            LlmAction::Command {
                action: cmd,
                text: param,
            } => match cmd {
                EditCommand::NewParagraph => self.new_paragraph(),
                EditCommand::DeleteLastSentence => self.delete_last_sentence(),
                EditCommand::DeleteLastParagraph => self.delete_last_paragraph(),
                EditCommand::Undo => {
                    self.undo();
                }
                EditCommand::Redo => {
                    self.redo();
                }
                EditCommand::ClearAll => self.clear(),
                EditCommand::InsertHeading => {
                    if let Some(heading_text) = param {
                        self.insert_heading(heading_text);
                    }
                }
                EditCommand::InsertBullet => {
                    if let Some(bullet_text) = param {
                        self.insert_bullet(bullet_text);
                    }
                }
            },
            LlmAction::Conversation { .. } => {
                // Conversations don't modify the document
            }
        }
    }

    /// Append text to the current (last) paragraph.
    pub fn append_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.save_snapshot();

        if let Some(last) = self.paragraphs.last_mut() {
            if !last.is_empty() {
                last.push(' ');
            }
            last.push_str(text);
        } else {
            self.paragraphs.push(text.to_string());
        }
    }

    /// Start a new paragraph.
    pub fn new_paragraph(&mut self) {
        self.save_snapshot();
        self.paragraphs.push(String::new());
    }

    /// Delete the last sentence from the current paragraph.
    pub fn delete_last_sentence(&mut self) {
        self.save_snapshot();

        if let Some(last) = self.paragraphs.last_mut() {
            if last.is_empty() {
                return;
            }
            // Find the last sentence boundary (. ! ?)
            let trimmed = last.trim_end();
            if let Some(pos) = trimmed[..trimmed.len().saturating_sub(1)]
                .rfind(['.', '!', '?', '。', '！', '？'])
            {
                // Keep up to and including the sentence-ending punctuation
                last.truncate(pos + 1);
                // Trim trailing whitespace
                let trimmed_len = last.trim_end().len();
                last.truncate(trimmed_len);
            } else {
                // No sentence boundary found — clear the whole paragraph
                last.clear();
            }
        }
    }

    /// Delete the last paragraph.
    pub fn delete_last_paragraph(&mut self) {
        if self.paragraphs.len() <= 1 {
            // Don't delete the only paragraph, just clear it
            self.save_snapshot();
            if let Some(last) = self.paragraphs.last_mut() {
                last.clear();
            }
            return;
        }
        self.save_snapshot();
        self.paragraphs.pop();
    }

    /// Insert a heading. In Markdown mode, prefixes with `# `.
    pub fn insert_heading(&mut self, text: &str) {
        self.save_snapshot();
        let heading = match self.format {
            OutputFormat::Markdown => format!("# {}", text),
            OutputFormat::PlainText => text.to_uppercase(),
        };
        self.paragraphs.push(heading);
        self.paragraphs.push(String::new());
    }

    /// Insert a bullet point. In Markdown mode, prefixes with `- `.
    pub fn insert_bullet(&mut self, text: &str) {
        self.save_snapshot();
        let bullet = match self.format {
            OutputFormat::Markdown => format!("- {}", text),
            OutputFormat::PlainText => format!("• {}", text),
        };
        if let Some(last) = self.paragraphs.last_mut() {
            if last.is_empty() {
                *last = bullet;
            } else {
                self.paragraphs.push(bullet);
            }
        }
    }

    /// Undo the last change. Returns true if successful.
    pub fn undo(&mut self) -> bool {
        if let Some(snapshot) = self.undo_stack.pop() {
            self.redo_stack.push(Snapshot {
                paragraphs: self.paragraphs.clone(),
            });
            self.paragraphs = snapshot.paragraphs;
            true
        } else {
            false
        }
    }

    /// Redo the last undone change. Returns true if successful.
    pub fn redo(&mut self) -> bool {
        if let Some(snapshot) = self.redo_stack.pop() {
            self.undo_stack.push(Snapshot {
                paragraphs: self.paragraphs.clone(),
            });
            self.paragraphs = snapshot.paragraphs;
            true
        } else {
            false
        }
    }

    /// Clear the entire document.
    pub fn clear(&mut self) {
        self.save_snapshot();
        self.paragraphs = vec![String::new()];
    }

    /// Render the document as a single string.
    pub fn render(&self) -> String {
        self.paragraphs
            .iter()
            .filter(|p| !p.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Render with paragraph numbers for display (used by the UI).
    #[allow(dead_code)]
    pub fn render_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for para in &self.paragraphs {
            if para.is_empty() {
                lines.push(String::new());
            } else {
                // Word-wrap will be handled by the UI widget
                lines.push(para.clone());
            }
        }
        lines
    }

    /// Get the total word count.
    pub fn word_count(&self) -> usize {
        self.paragraphs
            .iter()
            .map(|p| {
                p.split(|c: char| c.is_whitespace())
                    .filter(|w| !w.is_empty())
                    .count()
            })
            .sum()
    }

    /// Get the total character count (excluding whitespace between paragraphs).
    pub fn char_count(&self) -> usize {
        self.paragraphs.iter().map(|p| p.chars().count()).sum()
    }

    /// Get the number of paragraphs (non-empty).
    pub fn paragraph_count(&self) -> usize {
        self.paragraphs.iter().filter(|p| !p.is_empty()).count()
    }

    /// Check if the document is empty.
    pub fn is_empty(&self) -> bool {
        self.paragraphs.iter().all(|p| p.is_empty())
    }

    /// Save the document to a file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let content = self.render();
        std::fs::write(path, &content)?;
        Ok(())
    }

    /// Toggle the output format between plain text and Markdown.
    pub fn toggle_format(&mut self) {
        self.format = match self.format {
            OutputFormat::PlainText => OutputFormat::Markdown,
            OutputFormat::Markdown => OutputFormat::PlainText,
        };
    }

    /// Get a brief summary of the document for LLM context.
    pub fn context_summary(&self) -> String {
        let rendered = self.render();
        if rendered.is_empty() {
            return "[Document is empty]".to_string();
        }
        let word_count = self.word_count();
        let para_count = self.paragraph_count();
        // Show the last ~500 chars for context
        let recent = if rendered.len() > 500 {
            format!("...{}", &rendered[rendered.len() - 500..])
        } else {
            rendered.clone()
        };
        format!(
            "[Document: {} paragraphs, {} words]\n\
             --- Recent content ---\n\
             {}",
            para_count, word_count, recent
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_and_render() {
        let mut doc = Document::new(OutputFormat::PlainText);
        doc.append_text("Hello world.");
        doc.append_text("How are you?");
        assert_eq!(doc.render(), "Hello world. How are you?");
    }

    #[test]
    fn test_new_paragraph() {
        let mut doc = Document::new(OutputFormat::PlainText);
        doc.append_text("First paragraph.");
        doc.new_paragraph();
        doc.append_text("Second paragraph.");
        assert_eq!(doc.render(), "First paragraph.\n\nSecond paragraph.");
    }

    #[test]
    fn test_undo_redo() {
        let mut doc = Document::new(OutputFormat::PlainText);
        doc.append_text("Hello.");
        doc.append_text("World.");
        assert_eq!(doc.render(), "Hello. World.");

        assert!(doc.undo());
        assert_eq!(doc.render(), "Hello.");

        assert!(doc.redo());
        assert_eq!(doc.render(), "Hello. World.");
    }

    #[test]
    fn test_delete_last_sentence() {
        let mut doc = Document::new(OutputFormat::PlainText);
        doc.append_text("First sentence. Second sentence.");
        doc.delete_last_sentence();
        assert_eq!(doc.render(), "First sentence.");
    }

    #[test]
    fn test_delete_last_paragraph() {
        let mut doc = Document::new(OutputFormat::PlainText);
        doc.append_text("First.");
        doc.new_paragraph();
        doc.append_text("Second.");
        assert_eq!(doc.paragraph_count(), 2);

        doc.delete_last_paragraph();
        assert_eq!(doc.render(), "First.");
    }

    #[test]
    fn test_insert_heading_markdown() {
        let mut doc = Document::new(OutputFormat::Markdown);
        doc.insert_heading("Introduction");
        doc.append_text("Some content here.");
        assert!(doc.render().contains("# Introduction"));
    }

    #[test]
    fn test_insert_bullet_markdown() {
        let mut doc = Document::new(OutputFormat::Markdown);
        doc.insert_bullet("Item one");
        doc.insert_bullet("Item two");
        let rendered = doc.render();
        assert!(rendered.contains("- Item one"));
        assert!(rendered.contains("- Item two"));
    }

    #[test]
    fn test_clear() {
        let mut doc = Document::new(OutputFormat::PlainText);
        doc.append_text("Some content.");
        doc.clear();
        assert!(doc.is_empty());
        // Should be undoable
        assert!(doc.undo());
        assert_eq!(doc.render(), "Some content.");
    }

    #[test]
    fn test_word_count() {
        let mut doc = Document::new(OutputFormat::PlainText);
        doc.append_text("Hello world foo bar.");
        assert_eq!(doc.word_count(), 4);
    }
}
