mod audio;
mod config;
mod document;
mod events;
mod llm;
mod transcriber;
mod typer;
mod ui;

use anyhow::{Context, Result};
use audio::AudioSource;
use config::{Config, OutputSource};
use crossterm::{
    event::{Event, KeyCode, KeyEvent, KeyModifiers},
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
};
use document::Document;
use events::{AppEvent, LlmAction};
use futures::StreamExt;
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
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
    let mut config = Config::from_env()?;
    info!("Configuration loaded: {:?}", config);

    // Select the audio source before entering raw terminal mode.
    let (audio_source, output_file_override) = select_audio_source()?;
    if let Some(path) = output_file_override {
        config.output_file = Some(path);
    }
    info!("Selected audio source: {:?}", audio_source);

    // Create shared services
    let transcriber = Arc::new(transcriber::Transcriber::new(&config).await?);
    let llm_client = Arc::new(llm::LlmClient::new(&config));
    let typer = typer::Typer::new(config.typing_speed_cps);

    let asr_info = transcriber.backend_name();
    let llm_info = if config.has_llm_api() {
        config.llm_model.clone()
    } else {
        "Direct (Offline)".into()
    };

    // Create the document
    let document = Document::new(config.output_format.clone());

    // Create the app state
    let mut app = ui::App::new(
        document,
        config.output_source,
        asr_info,
        llm_info,
        audio_source.display_name(),
        audio_source.is_live(),
    );

    // Create event channel
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<AppEvent>();

    // Start live capture or decode the selected file.
    let (mut audio_capture, audio_file) = match &audio_source {
        AudioSource::Microphone | AudioSource::SystemAudio => {
            let capture = audio::AudioCapture::start(&audio_source, &config, event_tx.clone())?;
            info!(
                "Audio capture started (sample rate: {}Hz)",
                capture.sample_rate
            );
            (Some(capture), None)
        }
        AudioSource::File(path) => (None, Some(audio::load_audio_file(path)?)),
    };

    // Setup terminal
    terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, EnterAlternateScreen)?;
    crossterm::execute!(stdout, crossterm::event::EnableBracketedPaste)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    // Create crossterm event stream for async key input
    let mut event_stream = crossterm::event::EventStream::new();
    let source_epoch = Arc::new(AtomicU64::new(0));

    info!("TUI initialized, entering main loop");

    // File input is submitted once through the same pipeline used by live audio.
    if let Some(file) = audio_file {
        app.status = "⏳ Transcribing audio file...".into();
        event_tx.send(AppEvent::SpeechSegmentReady {
            samples: file.samples,
            sample_rate: file.sample_rate,
        })?;
    }

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
                    &config,
                    &source_epoch,
                );
            }

            // Terminal key events
            Some(Ok(event)) = event_stream.next() => {
                match event {
                    Event::Key(key) => {
                        let had_prompt = app.source_prompt.is_some();
                        if let Some(source) = handle_source_prompt(&mut app, key) {
                            match prepare_source(&source, &config, &event_tx) {
                                Ok((new_capture, new_file)) => {
                                    source_epoch.fetch_add(1, Ordering::Relaxed);
                                    audio_capture = new_capture;
                                    while event_rx.try_recv().is_ok() {}
                                    app.audio_source_info = source.display_name();
                                    app.continuous_audio = source.is_live();
                                    app.is_listening = source.is_live();
                                    app.is_processing = false;
                                    app.audio_level = 0.0;
                                    app.status = app.ready_status();
                                    if let Some(file) = new_file {
                                        let _ = event_tx.send(AppEvent::SpeechSegmentReady {
                                            samples: file.samples,
                                            sample_rate: file.sample_rate,
                                        });
                                    }
                                }
                                Err(e) => app.set_flash(format!("Audio source error: {e}")),
                            }
                            continue;
                        }
                        if had_prompt {
                            continue;
                        }
                        let action = handle_key_event(
                            &mut app,
                            key,
                            &typer,
                            &config,
                            audio_capture.as_ref(),
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
                    Event::Paste(text) => {
                        if let Some(ui::SourcePrompt::FilePath(path)) = app.source_prompt.as_mut() {
                            path.push_str(&text);
                        }
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
    crossterm::execute!(io::stdout(), crossterm::event::DisableBracketedPaste)?;
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
    config: &Config,
    source_epoch: &Arc<AtomicU64>,
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
            let epoch = source_epoch.clone();
            let started_at = epoch.load(Ordering::Relaxed);

            tokio::spawn(async move {
                if epoch.load(Ordering::Relaxed) != started_at {
                    return;
                }
                let _ = tx.send(AppEvent::Transcribing);
                match t.transcribe(&samples, sample_rate).await {
                    Ok(text) => {
                        if epoch.load(Ordering::Relaxed) == started_at {
                            let _ = tx.send(AppEvent::TranscriptReady { text });
                        }
                    }
                    Err(e) => {
                        error!("Transcription failed: {}", e);
                        if epoch.load(Ordering::Relaxed) == started_at {
                            let _ = tx.send(AppEvent::Error(format!("ASR error: {}", e)));
                        }
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
                app.status = app.ready_status();
                return;
            }

            info!("Transcript: \"{}\"", text);
            app.add_transcript_entry(&text, "", "");

            if app.output_source == OutputSource::Asr {
                app.apply_action(LlmAction::Dictate { text });
                app.is_processing = false;
                app.status = app.ready_status();
                save_output_file(app, config);
                return;
            }

            // If no LLM API is configured, use local rule-based intent parsing (dictation / simple commands)
            if !config.has_llm_api() {
                let action = parse_local_intent(&text);
                info!("Local offline action: {:?}", action);
                app.apply_action(action);
                app.is_processing = false;
                app.status = app.ready_status();
                save_output_file(app, config);
                return;
            }

            app.status = "🧠 Classifying intent...".into();

            let tx = event_tx.clone();
            let l = llm_client.clone();
            let doc_context = app.document.context_summary();
            let raw_text = text.clone();
            let epoch = source_epoch.clone();
            let started_at = epoch.load(Ordering::Relaxed);

            tokio::spawn(async move {
                if epoch.load(Ordering::Relaxed) != started_at {
                    return;
                }
                let _ = tx.send(AppEvent::ClassifyingIntent);
                match l.classify_intent(&raw_text, &doc_context).await {
                    Ok(action) => {
                        if epoch.load(Ordering::Relaxed) == started_at {
                            let _ = tx.send(AppEvent::LlmResponse(action));
                        }
                    }
                    Err(e) => {
                        error!("LLM classification failed: {}", e);
                        // Fall back to treating as dictation
                        if epoch.load(Ordering::Relaxed) == started_at {
                            let _ = tx
                                .send(AppEvent::LlmResponse(LlmAction::Dictate { text: raw_text }));
                        }
                    }
                }
            });
        }

        AppEvent::ClassifyingIntent => {
            app.status = "🧠 Classifying intent...".into();
        }

        AppEvent::LlmResponse(action) => {
            info!("LLM action: {:?}", action);
            let previous = app.document.render();
            app.apply_action(action);
            app.is_processing = false;
            app.status = app.ready_status();
            if app.document.render() != previous {
                save_output_file(app, config);
            }
        }

        AppEvent::StatusMessage(msg) => {
            app.status = msg;
        }

        AppEvent::Error(msg) => {
            warn!("Error: {}", msg);
            app.set_flash(format!("❌ {}", msg));
            app.is_processing = false;
            app.status = app.ready_status();
        }
    }
}

/// Keep the configured output file current after document edits.
fn save_output_file(app: &mut ui::App, config: &Config) {
    let Some(path) = &config.output_file else {
        return;
    };
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        if let Err(e) = std::fs::create_dir_all(parent) {
            app.set_flash(format!("❌ Output file error: {e}"));
            error!("Failed to create output directory {:?}: {}", parent, e);
            return;
        }
    }
    if let Err(e) = app.document.save(path) {
        app.set_flash(format!("❌ Output file error: {e}"));
        error!("Failed to save output file {:?}: {}", path, e);
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
    audio: Option<&audio::AudioCapture>,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> KeyAction {
    // Handle key combinations
    match (key.modifiers, key.code) {
        // Quit
        (KeyModifiers::NONE, KeyCode::Char('q')) | (KeyModifiers::NONE, KeyCode::Esc) => {
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
                save_output_file(app, config);
            } else {
                app.set_flash("Nothing to undo".into());
            }
        }

        // Redo (Ctrl+R)
        (KeyModifiers::CONTROL, KeyCode::Char('r')) => {
            if app.document.redo() {
                app.set_flash("↪️ Redo".into());
                save_output_file(app, config);
            } else {
                app.set_flash("Nothing to redo".into());
            }
        }

        // Toggle listening (Space)
        (KeyModifiers::NONE, KeyCode::Char(' ')) => {
            let Some(audio) = audio else {
                app.set_flash("Pause is only available for live audio".into());
                return KeyAction::Continue;
            };
            let now_listening = audio.toggle_listening();
            app.is_listening = now_listening;
            if now_listening {
                app.status = app.ready_status();
                app.set_flash("▶️ Resumed listening".into());
            } else {
                app.status = "⏸ Paused".into();
                app.set_flash("⏸ Paused listening".into());
            }
        }

        // Toggle output format (Tab)
        (KeyModifiers::NONE, KeyCode::Tab) => {
            app.document.toggle_format();
            app.set_flash(format!("📝 Format: {}", app.document.format.display_name()));
        }

        // Switch the result used for future utterances.
        (KeyModifiers::NONE, KeyCode::Char('m')) => {
            app.output_source = app.output_source.toggle();
            app.set_flash(format!("Output: {}", app.output_source.display_name()));
        }

        (KeyModifiers::NONE, KeyCode::Char('a')) => {
            app.source_prompt = Some(ui::SourcePrompt::Choice);
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

fn handle_source_prompt(app: &mut ui::App, key: KeyEvent) -> Option<AudioSource> {
    use ui::SourcePrompt;
    let prompt = app.source_prompt.as_mut()?;
    match prompt {
        SourcePrompt::Choice => match key.code {
            KeyCode::Char('1') => app.source_prompt = Some(SourcePrompt::FilePath(String::new())),
            KeyCode::Char('2') => {
                app.source_prompt = None;
                return Some(AudioSource::Microphone);
            }
            KeyCode::Char('3') => {
                app.source_prompt = None;
                return Some(AudioSource::SystemAudio);
            }
            KeyCode::Esc => app.source_prompt = None,
            _ => {}
        },
        SourcePrompt::FilePath(path) => match key.code {
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => path.push(c),
            KeyCode::Backspace => {
                path.pop();
            }
            KeyCode::Esc => app.source_prompt = None,
            KeyCode::Enter => {
                let entered = PathBuf::from(path.trim().trim_matches('"'));
                match validate_audio_path(entered) {
                    Ok(source) => {
                        app.source_prompt = None;
                        return Some(source);
                    }
                    Err(e) => app.set_flash(format!("Audio file error: {e}")),
                }
            }
            _ => {}
        },
    }
    None
}

fn prepare_source(
    source: &AudioSource,
    config: &Config,
    event_tx: &mpsc::UnboundedSender<AppEvent>,
) -> Result<(Option<audio::AudioCapture>, Option<audio::AudioFile>)> {
    match source {
        AudioSource::Microphone | AudioSource::SystemAudio => Ok((
            Some(audio::AudioCapture::start(
                source,
                config,
                event_tx.clone(),
            )?),
            None,
        )),
        AudioSource::File(path) => Ok((None, Some(audio::load_audio_file(path)?))),
    }
}

/// Select a live source or an audio file. Command-line flags make this
/// usable in scripts; without one, Typist presents a small startup menu.
fn select_audio_source() -> Result<(AudioSource, Option<PathBuf>)> {
    let mut args = std::env::args().skip(1);
    let mut source = None;
    let mut output_file = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--microphone" | "--computer-audio" | "--computer" => {
                source = Some(AudioSource::Microphone)
            }
            "--system-audio" => source = Some(AudioSource::SystemAudio),
            "--audio-file" | "--file" => {
                let path = args
                    .next()
                    .map(PathBuf::from)
                    .context("--audio-file requires a path to a WAV or MP3 file")?;
                source = Some(validate_audio_path(path)?);
            }
            "--output-file" => {
                output_file = Some(PathBuf::from(
                    args.next().context("--output-file requires a path")?,
                ));
            }
            "--help" | "-h" => {
                println!("Usage: typist [--audio-file <path.wav|path.mp3> | --microphone | --system-audio] [--output-file <path>]");
                std::process::exit(0);
            }
            _ => anyhow::bail!(
                "Unknown argument '{}'. Use --help for available options.",
                arg
            ),
        }
    }
    if let Some(source) = source {
        return Ok((source, output_file));
    }

    println!("Select audio source:");
    println!("  1. From file (WAV or MP3)");
    println!("  2. Microphone");
    println!("  3. System audio (what you hear)");
    print!("Choice [2]: ");
    io::stdout().flush()?;

    let mut choice = String::new();
    io::stdin().read_line(&mut choice)?;
    match choice.trim() {
        "" | "2" => Ok((AudioSource::Microphone, output_file)),
        "3" => Ok((AudioSource::SystemAudio, output_file)),
        "1" => {
            print!("WAV or MP3 file path: ");
            io::stdout().flush()?;
            let mut path = String::new();
            io::stdin().read_line(&mut path)?;
            let path = path.trim().trim_matches('"');
            if path.is_empty() {
                anyhow::bail!("No audio file was selected");
            }
            Ok((validate_audio_path(PathBuf::from(path))?, output_file))
        }
        other => anyhow::bail!("Invalid audio source '{}'. Choose 1, 2, or 3.", other),
    }
}

fn validate_audio_path(path: PathBuf) -> Result<AudioSource> {
    if !path.is_file() {
        anyhow::bail!("Audio file does not exist: {}", path.display());
    }
    let supported = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("wav") || extension.eq_ignore_ascii_case("mp3")
        });
    if !supported {
        anyhow::bail!("Unsupported audio file. Typist accepts WAV and MP3 files.");
    }
    Ok(AudioSource::File(path))
}

/// Parse simple editing commands or dictation locally when no LLM API is configured.
fn parse_local_intent(text: &str) -> LlmAction {
    let lower = text.trim().to_lowercase();
    let trimmed = lower.trim_matches(|c: char| {
        c.is_ascii_punctuation() || c == '。' || c == '，' || c == '！' || c == '？' || c == '、'
    });

    match trimmed {
        "new paragraph" | "new line" | "換行" | "换行" | "另起一段" | "下一段" => {
            LlmAction::Command {
                action: events::EditCommand::NewParagraph,
                text: None,
            }
        }
        "delete last sentence"
        | "delete sentence"
        | "刪除最後一句"
        | "删除最后一句"
        | "刪掉最後一句"
        | "删掉最后一句" => LlmAction::Command {
            action: events::EditCommand::DeleteLastSentence,
            text: None,
        },
        "delete last paragraph"
        | "delete paragraph"
        | "刪除最後一段"
        | "删除最后一段"
        | "刪掉最後一段"
        | "删掉最后一段" => LlmAction::Command {
            action: events::EditCommand::DeleteLastParagraph,
            text: None,
        },
        "undo" | "撤銷" | "撤销" | "復原" | "复原" => LlmAction::Command {
            action: events::EditCommand::Undo,
            text: None,
        },
        "redo" | "重做" => LlmAction::Command {
            action: events::EditCommand::Redo,
            text: None,
        },
        "clear all" | "clear document" | "清空" | "全部清空" | "清空文件" => {
            LlmAction::Command {
                action: events::EditCommand::ClearAll,
                text: None,
            }
        }
        _ => {
            if let Some(rest) = trimmed
                .strip_prefix("heading ")
                .or_else(|| trimmed.strip_prefix("標題 "))
                .or_else(|| trimmed.strip_prefix("标题 "))
            {
                LlmAction::Command {
                    action: events::EditCommand::InsertHeading,
                    text: Some(rest.trim().to_string()),
                }
            } else if let Some(rest) = trimmed
                .strip_prefix("bullet ")
                .or_else(|| trimmed.strip_prefix("項目 "))
                .or_else(|| trimmed.strip_prefix("列表 "))
            {
                LlmAction::Command {
                    action: events::EditCommand::InsertBullet,
                    text: Some(rest.trim().to_string()),
                }
            } else {
                LlmAction::Dictate {
                    text: text.trim().to_string(),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_local_intent_commands() {
        match parse_local_intent("new paragraph") {
            LlmAction::Command { action, .. } => {
                assert_eq!(action, events::EditCommand::NewParagraph);
            }
            _ => panic!("Expected NewParagraph"),
        }

        match parse_local_intent("換行。") {
            LlmAction::Command { action, .. } => {
                assert_eq!(action, events::EditCommand::NewParagraph);
            }
            _ => panic!("Expected NewParagraph"),
        }

        match parse_local_intent("undo") {
            LlmAction::Command { action, .. } => {
                assert_eq!(action, events::EditCommand::Undo);
            }
            _ => panic!("Expected Undo"),
        }

        match parse_local_intent("撤銷！") {
            LlmAction::Command { action, .. } => {
                assert_eq!(action, events::EditCommand::Undo);
            }
            _ => panic!("Expected Undo"),
        }
    }

    #[test]
    fn test_parse_local_intent_dictate() {
        match parse_local_intent("Hello world, this is a test.") {
            LlmAction::Dictate { text } => {
                assert_eq!(text, "Hello world, this is a test.");
            }
            _ => panic!("Expected Dictate"),
        }
    }
}
