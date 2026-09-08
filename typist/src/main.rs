mod audio;
mod config;
mod document;
mod events;
mod llm;
mod transcriber;
mod typer;
mod ui;

use anyhow::Result;
use config::Config;
use crossterm::{
    event::{Event, KeyCode, KeyEvent, KeyModifiers},
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
};
use document::Document;
use events::{AppEvent, LlmAction};
use futures::StreamExt;
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging to file (not stdout, since we use TUI)
    let log_file = std::fs::File::create("typist.log").ok();
    if let Some(file) = log_file {
        tracing_subscriber::fmt()
            .with_writer(std::sync::Mutex::new(file))
            .with_env_filter(
                tracing_subscriber::EnvFilter::from_default_env()
                    .add_directive("typist=debug".parse().unwrap()),
            )
            .init();
    }

    info!("=== Typist Agent starting ===");

    // Load configuration
    let config = Config::from_env()?;
    info!("Configuration loaded: {:?}", config);

    // Create the document
    let document = Document::new(config.output_format.clone());

    // Create the app state
    let mut app = ui::App::new(document);

    // Create event channel
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<AppEvent>();

    // Start audio capture
    let audio = audio::AudioCapture::start(&config, event_tx.clone())?;
    info!("Audio capture started (sample rate: {}Hz)", audio.sample_rate);

    // Create shared services
    let transcriber = Arc::new(transcriber::Transcriber::new(&config));
    let llm_client = Arc::new(llm::LlmClient::new(&config));
    let typer = typer::Typer::new(config.typing_speed_cps);

    // Setup terminal
    terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    // Create crossterm event stream for async key input
    let mut event_stream = crossterm::event::EventStream::new();

    info!("TUI initialized, entering main loop");

    // Tick timer for periodic UI updates
    let mut tick_interval = tokio::time::interval(std::time::Duration::from_millis(100));

    // Main event loop
    let result: Result<()> = loop {
        // Render the UI
        terminal.draw(|frame| ui::render(frame, &app))?;

        tokio::select! {
            // App events from the audio/transcription/LLM pipeline
            Some(event) = event_rx.recv() => {
                handle_app_event(
                    &mut app,
                    event,
                    &event_tx,
                    &transcriber,
                    &llm_client,
                );
            }

            // Terminal key events
            Some(Ok(event)) = event_stream.next() => {
                match event {
                    Event::Key(key) => {
                        let action = handle_key_event(
                            &mut app,
                            key,
                            &typer,
                            &config,
                            &audio,
                            &mut terminal,
                        );
                        match action {
                            KeyAction::Quit => break Ok(()),
                            KeyAction::Continue => {}
                        }
                    }
                    Event::Resize(_, _) => {
                        // Terminal resized — will re-render on next loop
                    }
                    _ => {}
                }
            }

            // Periodic tick for UI updates (flash messages, etc.)
            _ = tick_interval.tick() => {
                app.tick();
            }
        }

        if app.should_quit {
            break Ok(());
        }
    };

    // Cleanup terminal
    terminal::disable_raw_mode()?;
    crossterm::execute!(io::stdout(), LeaveAlternateScreen)?;

    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }

    info!("=== Typist Agent shutting down ===");
    println!("Typist Agent exited. Document saved to clipboard if copied.");

    result
}

/// Handle an application event from the processing pipeline.
fn handle_app_event(
    app: &mut ui::App,
    event: AppEvent,
    event_tx: &mpsc::UnboundedSender<AppEvent>,
    transcriber: &Arc<transcriber::Transcriber>,
    llm_client: &Arc<llm::LlmClient>,
) {
    match event {
        AppEvent::AudioLevelUpdate(level) => {
            app.audio_level = level;
        }

        AppEvent::SpeechStarted => {
            app.status = "🗣️ Listening — Speaking detected...".into();
        }

        AppEvent::SpeechSegmentReady {
            samples,
            sample_rate,
        } => {
            app.status = "⏳ Transcribing...".into();
            app.is_processing = true;

            let tx = event_tx.clone();
            let t = transcriber.clone();

            tokio::spawn(async move {
                let _ = tx.send(AppEvent::Transcribing);
                match t.transcribe(&samples, sample_rate).await {
                    Ok(text) => {
                        let _ = tx.send(AppEvent::TranscriptReady { text });
                    }
                    Err(e) => {
                        error!("Transcription failed: {}", e);
                        let _ = tx.send(AppEvent::Error(format!("ASR error: {}", e)));
                    }
                }
            });
        }

        AppEvent::Transcribing => {
            app.status = "⏳ Transcribing...".into();
        }

        AppEvent::TranscriptReady { text } => {
            if text.trim().is_empty() {
                app.is_processing = false;
                app.status = "🎤 Listening...".into();
                return;
            }

            info!("Transcript: \"{}\"", text);
            app.add_transcript_entry(&text, "", "");
            app.status = "🧠 Classifying intent...".into();

            let tx = event_tx.clone();
            let l = llm_client.clone();
            let doc_context = app.document.context_summary();
            let raw_text = text.clone();

            tokio::spawn(async move {
                let _ = tx.send(AppEvent::ClassifyingIntent);
                match l.classify_intent(&raw_text, &doc_context).await {
                    Ok(action) => {
                        let _ = tx.send(AppEvent::LlmResponse(action));
                    }
                    Err(e) => {
                        error!("LLM classification failed: {}", e);
                        // Fall back to treating as dictation
                        let _ = tx.send(AppEvent::LlmResponse(LlmAction::Dictate {
                            text: raw_text,
                        }));
                    }
                }
            });
        }

        AppEvent::ClassifyingIntent => {
            app.status = "🧠 Classifying intent...".into();
        }

        AppEvent::LlmResponse(action) => {
            info!("LLM action: {:?}", action);
            app.apply_action(action);
            app.is_processing = false;
            app.status = "🎤 Listening...".into();
        }

        AppEvent::StatusMessage(msg) => {
            app.status = msg;
        }

        AppEvent::Error(msg) => {
            warn!("Error: {}", msg);
            app.set_flash(format!("❌ {}", msg));
            app.is_processing = false;
            app.status = "🎤 Listening...".into();
        }
    }
}

/// Result of handling a key event.
enum KeyAction {
    Quit,
    Continue,
}

/// Handle a keyboard event.
fn handle_key_event(
    app: &mut ui::App,
    key: KeyEvent,
    typer: &typer::Typer,
    config: &Config,
    audio: &audio::AudioCapture,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> KeyAction {
    // Handle key combinations
    match (key.modifiers, key.code) {
        // Quit
        (KeyModifiers::NONE, KeyCode::Char('q'))
        | (KeyModifiers::NONE, KeyCode::Esc) => {
            return KeyAction::Quit;
        }

        // Save document to file (Ctrl+S)
        (KeyModifiers::CONTROL, KeyCode::Char('s')) => {
            let ext = app.document.format.file_extension();
            let filename = format!(
                "typist_{}.{}",
                chrono::Local::now().format("%Y-%m-%d_%H%M%S"),
                ext
            );
            let path = config.save_directory.join(&filename);
            match app.document.save(&path) {
                Ok(()) => {
                    app.set_flash(format!("✅ Saved to {}", filename));
                    info!("Document saved to {:?}", path);
                }
                Err(e) => {
                    app.set_flash(format!("❌ Save failed: {}", e));
                    error!("Save failed: {}", e);
                }
            }
        }

        // Copy to clipboard (Ctrl+Y)
        (KeyModifiers::CONTROL, KeyCode::Char('y')) => {
            let text = app.document.render();
            if text.is_empty() {
                app.set_flash("📋 Document is empty".into());
            } else {
                match typer.copy_to_clipboard(&text) {
                    Ok(()) => {
                        app.set_flash(format!("📋 Copied {} chars to clipboard", text.len()));
                    }
                    Err(e) => {
                        app.set_flash(format!("❌ Clipboard error: {}", e));
                    }
                }
            }
        }

        // Type-out to focused window (Ctrl+E)
        (KeyModifiers::CONTROL, KeyCode::Char('e')) => {
            let text = app.document.render();
            if text.is_empty() {
                app.set_flash("Document is empty".into());
            } else {
                // Exit TUI temporarily
                let _ = terminal::disable_raw_mode();
                let _ = crossterm::execute!(io::stdout(), LeaveAlternateScreen);

                println!("\n🖊️  Type-out mode activated!");
                println!("Switch to your target window now.");
                println!("Pasting in 3 seconds...\n");

                std::thread::sleep(std::time::Duration::from_secs(3));

                match typer.type_out(&text) {
                    Ok(()) => {
                        println!("✅ Done! Press Enter to return to Typist...");
                    }
                    Err(e) => {
                        println!("❌ Type-out failed: {}", e);
                        println!("Press Enter to return to Typist...");
                    }
                }

                // Wait for user to press Enter
                let mut input = String::new();
                let _ = std::io::stdin().read_line(&mut input);

                // Re-enter TUI
                let _ = crossterm::execute!(io::stdout(), EnterAlternateScreen);
                let _ = terminal::enable_raw_mode();
                let _ = terminal.clear();
            }
        }

        // Undo (Ctrl+Z)
        (KeyModifiers::CONTROL, KeyCode::Char('z')) => {
            if app.document.undo() {
                app.set_flash("↩️ Undo".into());
            } else {
                app.set_flash("Nothing to undo".into());
            }
        }

        // Redo (Ctrl+R)
        (KeyModifiers::CONTROL, KeyCode::Char('r')) => {
            if app.document.redo() {
                app.set_flash("↪️ Redo".into());
            } else {
                app.set_flash("Nothing to redo".into());
            }
        }

        // Toggle listening (Space)
        (KeyModifiers::NONE, KeyCode::Char(' ')) => {
            let now_listening = audio.toggle_listening();
            app.is_listening = now_listening;
            if now_listening {
                app.status = "🎤 Listening...".into();
                app.set_flash("▶️ Resumed listening".into());
            } else {
                app.status = "⏸ Paused".into();
                app.set_flash("⏸ Paused listening".into());
            }
        }

        // Toggle output format (Tab)
        (KeyModifiers::NONE, KeyCode::Tab) => {
            app.document.toggle_format();
            app.set_flash(format!(
                "📝 Format: {}",
                app.document.format.display_name()
            ));
        }

        // Scroll document up
        (KeyModifiers::NONE, KeyCode::Up) => {
            app.doc_scroll = app.doc_scroll.saturating_sub(1);
        }

        // Scroll document down
        (KeyModifiers::NONE, KeyCode::Down) => {
            app.doc_scroll = app.doc_scroll.saturating_add(1);
        }

        _ => {}
    }

    KeyAction::Continue
}
