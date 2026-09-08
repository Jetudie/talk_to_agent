use crate::document::Document;
use crate::events::LlmAction;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap},
    Frame,
};

/// A single entry in the transcript log.
#[derive(Debug, Clone)]
pub struct TranscriptEntry {
    pub timestamp: String,
    pub raw_text: String,
    pub intent: String,
    pub detail: String,
}

/// The main application state.
pub struct App {
    /// The document being typed.
    pub document: Document,
    /// Whether the agent is currently listening for speech.
    pub is_listening: bool,
    /// Whether the agent is currently processing (transcribing/classifying).
    pub is_processing: bool,
    /// Current status message displayed in the status panel.
    pub status: String,
    /// Log of recent transcript entries.
    pub transcript_log: Vec<TranscriptEntry>,
    /// Current audio input level (0.0 to 1.0).
    pub audio_level: f32,
    /// Whether the application should quit.
    pub should_quit: bool,
    /// Most recent conversation response from the LLM.
    pub conversation_response: Option<String>,
    /// Document scroll offset.
    pub doc_scroll: u16,
    /// Transcript scroll offset.
    pub log_scroll: u16,
    /// Flash message (temporary notification).
    pub flash_message: Option<(String, std::time::Instant)>,
}

impl App {
    /// Create a new App with the given document.
    pub fn new(document: Document) -> Self {
        Self {
            document,
            is_listening: true,
            is_processing: false,
            status: "Ready — Listening...".into(),
            transcript_log: Vec::new(),
            audio_level: 0.0,
            should_quit: false,
            conversation_response: None,
            doc_scroll: 0,
            log_scroll: 0,
            flash_message: None,
        }
    }

    /// Add a transcript entry to the log.
    pub fn add_transcript_entry(
        &mut self,
        raw_text: &str,
        intent: &str,
        detail: &str,
    ) {
        let timestamp = chrono::Local::now().format("%H:%M:%S").to_string();
        self.transcript_log.push(TranscriptEntry {
            timestamp,
            raw_text: raw_text.to_string(),
            intent: intent.to_string(),
            detail: detail.to_string(),
        });
        // Keep log manageable
        if self.transcript_log.len() > 100 {
            self.transcript_log.remove(0);
        }
        // Auto-scroll to bottom
        self.log_scroll = self.transcript_log.len().saturating_sub(1) as u16;
    }

    /// Apply an LLM action to the app state.
    pub fn apply_action(&mut self, action: LlmAction) {
        match &action {
            LlmAction::Dictate { text } => {
                self.update_last_transcript("dictate", text);
                self.document.apply_action(&action);
            }
            LlmAction::Command { action: cmd, text } => {
                let detail = if let Some(t) = text {
                    format!("{:?}: {}", cmd, t)
                } else {
                    format!("{:?}", cmd)
                };
                self.update_last_transcript("command", &detail);
                self.document.apply_action(&action);
            }
            LlmAction::Conversation { response } => {
                self.update_last_transcript("chat", response);
                self.conversation_response = Some(response.clone());
            }
        }
    }

    /// Update the last transcript entry with the classified intent.
    fn update_last_transcript(&mut self, intent: &str, detail: &str) {
        if let Some(last) = self.transcript_log.last_mut() {
            last.intent = intent.to_string();
            last.detail = detail.to_string();
        }
    }

    /// Set a flash message that auto-clears after a few seconds.
    pub fn set_flash(&mut self, message: String) {
        self.flash_message = Some((message, std::time::Instant::now()));
    }

    /// Clear expired flash messages.
    pub fn tick(&mut self) {
        if let Some((_, created)) = &self.flash_message {
            if created.elapsed() > std::time::Duration::from_secs(3) {
                self.flash_message = None;
            }
        }
        // Clear old conversation responses
        if self.conversation_response.is_some() {
            // Keep conversation response visible for 10 seconds
        }
    }
}

/// Render the entire UI.
pub fn render(frame: &mut Frame, app: &App) {
    let size = frame.area();

    // Main vertical layout: Document | Bottom panel | Help bar
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(8),      // Document area
            Constraint::Length(12),   // Bottom panel (transcript + status)
            Constraint::Length(3),    // Help bar
        ])
        .split(size);

    render_document(frame, app, main_chunks[0]);
    render_bottom_panel(frame, app, main_chunks[1]);
    render_help_bar(frame, app, main_chunks[2]);
}

/// Render the document view panel.
fn render_document(frame: &mut Frame, app: &App, area: Rect) {
    let format_label = app.document.format.display_name();
    let word_count = app.document.word_count();
    let para_count = app.document.paragraph_count();

    let title = format!(
        " 📄 Document [{}] — {} words, {} paragraphs ",
        format_label, word_count, para_count
    );

    let doc_text = if app.document.is_empty() {
        "Start speaking to begin typing...\n\nYour dictated text will appear here.".to_string()
    } else {
        app.document.render()
    };

    let style = if app.document.is_empty() {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default().fg(Color::White)
    };

    // Show conversation response if present
    let display_text = if let Some(response) = &app.conversation_response {
        format!("{}\n\n💬 Agent: {}", doc_text, response)
    } else {
        doc_text
    };

    let paragraph = Paragraph::new(display_text)
        .style(style)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(title)
                .title_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .wrap(Wrap { trim: false })
        .scroll((app.doc_scroll, 0));

    frame.render_widget(paragraph, area);
}

/// Render the bottom panel (transcript log + status).
fn render_bottom_panel(frame: &mut Frame, app: &App, area: Rect) {
    let bottom_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(65), // Transcript log
            Constraint::Percentage(35), // Status panel
        ])
        .split(area);

    render_transcript_log(frame, app, bottom_chunks[0]);
    render_status_panel(frame, app, bottom_chunks[1]);
}

/// Render the transcript log panel.
fn render_transcript_log(frame: &mut Frame, app: &App, area: Rect) {
    let items: Vec<ListItem> = app
        .transcript_log
        .iter()
        .rev()
        .take(20)
        .rev()
        .map(|entry| {
            let intent_color = match entry.intent.as_str() {
                "dictate" => Color::Green,
                "command" => Color::Yellow,
                "chat" => Color::Blue,
                _ => Color::DarkGray,
            };

            let intent_label = if entry.intent.is_empty() {
                "...".to_string()
            } else {
                entry.intent.clone()
            };

            let line = Line::from(vec![
                Span::styled(
                    format!("[{}] ", entry.timestamp),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    format!("[{}] ", intent_label),
                    Style::default()
                        .fg(intent_color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    truncate_str(&entry.raw_text, 30),
                    Style::default().fg(Color::White),
                ),
                if !entry.detail.is_empty() {
                    Span::styled(
                        format!(" → {}", truncate_str(&entry.detail, 35)),
                        Style::default().fg(Color::Gray),
                    )
                } else {
                    Span::raw("")
                },
            ]);

            ListItem::new(line)
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" 📝 Transcript Log ")
            .title_style(Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))
            .border_style(Style::default().fg(Color::Magenta)),
    );

    frame.render_widget(list, area);
}

/// Render the status panel.
fn render_status_panel(frame: &mut Frame, app: &App, area: Rect) {
    let status_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Status line
            Constraint::Length(3), // Audio level
            Constraint::Min(2),   // Stats
        ])
        .split(area);

    // Status line
    let status_color = if app.is_processing {
        Color::Yellow
    } else if app.is_listening {
        Color::Green
    } else {
        Color::Red
    };

    let status_icon = if app.is_processing {
        "⏳"
    } else if app.is_listening {
        "🎤"
    } else {
        "⏸"
    };

    // Check for flash message
    let status_text = if let Some((msg, _)) = &app.flash_message {
        msg.clone()
    } else {
        app.status.clone()
    };

    let status = Paragraph::new(format!("{} {}", status_icon, status_text))
        .style(Style::default().fg(status_color))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Status ")
                .title_style(Style::default().fg(Color::Yellow))
                .border_style(Style::default().fg(Color::Yellow)),
        );

    frame.render_widget(status, status_chunks[0]);

    // Audio level gauge
    let level = (app.audio_level * 50.0).clamp(0.0, 1.0); // Scale for visibility
    let level_color = if level > 0.7 {
        Color::Red
    } else if level > 0.3 {
        Color::Yellow
    } else {
        Color::Green
    };

    let gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Level ")
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .gauge_style(Style::default().fg(level_color))
        .ratio(level as f64);

    frame.render_widget(gauge, status_chunks[1]);

    // Stats
    let stats = Paragraph::new(format!(
        "Words: {}\nChars: {}\nParas: {}",
        app.document.word_count(),
        app.document.char_count(),
        app.document.paragraph_count(),
    ))
    .style(Style::default().fg(Color::Gray))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Stats ")
            .border_style(Style::default().fg(Color::DarkGray)),
    );

    frame.render_widget(stats, status_chunks[2]);
}

/// Render the help bar at the bottom.
fn render_help_bar(frame: &mut Frame, app: &App, area: Rect) {
    let help_spans = vec![
        Span::styled(" Space", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(": Pause  ", Style::default().fg(Color::Gray)),
        Span::styled("Ctrl+S", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(": Save  ", Style::default().fg(Color::Gray)),
        Span::styled("Ctrl+Y", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(": Copy  ", Style::default().fg(Color::Gray)),
        Span::styled("Ctrl+E", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(": Type-out  ", Style::default().fg(Color::Gray)),
        Span::styled("Ctrl+Z", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(": Undo  ", Style::default().fg(Color::Gray)),
        Span::styled("Ctrl+R", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(": Redo  ", Style::default().fg(Color::Gray)),
        Span::styled("Tab", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(": Format  ", Style::default().fg(Color::Gray)),
        Span::styled("q/Esc", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(": Quit", Style::default().fg(Color::Gray)),
    ];

    let format_indicator = format!(" [{}] ", app.document.format.display_name());

    let help = Paragraph::new(Line::from(help_spans)).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Typist Agent ")
            .title_style(
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )
            .title(
                ratatui::widgets::block::Title::from(format_indicator)
                    .alignment(ratatui::layout::Alignment::Right),
            )
            .border_style(Style::default().fg(Color::DarkGray)),
    );

    frame.render_widget(help, area);
}

/// Truncate a string to a maximum length, appending "..." if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_len.saturating_sub(3)).collect();
        format!("{}...", truncated)
    }
}
