# Typist — AI Professional Typer Agent

A Rust-based AI agent that listens to your voice and acts as a **professional typist**. Unlike simple speech-to-text, this agent intelligently distinguishes between:

- **Dictation** — content you want typed into the document
- **Commands** — editing instructions (new paragraph, delete, undo, etc.)
- **Conversation** — things you say that should NOT be typed (questions, thinking aloud)

The agent uses an LLM to understand your intent and produces clean, well-formatted written output.

## Features

- Real-time microphone capture with Voice Activity Detection (VAD)
- Configurable ASR (Speech-to-Text) API endpoint
- LLM-powered intent classification (dictation vs command vs conversation)
- English and Mandarin (中文) support
- Terminal UI with live document view, transcript log, and audio level meter
- Full undo/redo support
- Plain text and Markdown output formats
- Export via clipboard copy or keyboard simulation (type-out)
- Save to .txt or .md files
- **Zero API setup required**: Runs fully offline with built-in lightweight local Whisper model, with seamless fallback if no API is configured.

## Prerequisites

- Rust toolchain (install via [rustup](https://rustup.rs/))
- A working microphone
- *(Optional)* An ASR API endpoint (OpenAI Whisper-compatible) if not using local Whisper
- *(Optional)* An LLM API key (OpenAI or compatible) for intent classification; if unset, typist runs in direct dictation mode with local voice commands

## Setup

1. **Clone and build**:
   ```bash
   cd typist
   cargo build --release
   ```

2. **Run immediately (Zero Config / Local Whisper)**:
   ```bash
   cargo run --release
   ```
   If no `.env` or API keys are configured, typist will automatically run using local lightweight Whisper (`tiny`).

3. **Optional Configuration**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` if you wish to configure a remote ASR endpoint or an LLM for enhanced formatting.

## Configuration

Edit the `.env` file:

| Variable | Description | Default |
|---|---|---|
| `ASR_BACKEND` | `local` (runs local Whisper) or `remote` (calls ASR API) | Auto (`local` if no API set) |
| `WHISPER_MODEL` | Local Whisper model size (`tiny`, `base`, `small`) | `tiny` |
| `LOCAL_WHISPER_MODEL_PATH` | Path to custom GGML model file | Auto-detected / downloaded |
| `ASR_API_URL` | ASR API endpoint URL | `http://localhost:8080/v1/audio/transcriptions` |
| `ASR_API_KEY` | ASR API key | (empty) |
| `ASR_MODEL` | Remote ASR model name | `whisper-1` |
| `ASR_LANGUAGE` | Language hint (`en`, `zh`, `auto`) | `auto` |
| `LLM_API_KEY` | LLM API key | (optional, empty = offline dictation) |
| `LLM_BASE_URL` | LLM API base URL | `https://api.openai.com/v1` |
| `LLM_MODEL` | LLM model name | `gpt-4o-mini` |
| `SILENCE_THRESHOLD` | Speech detection sensitivity | `0.02` |
| `SILENCE_DURATION` | Seconds of silence to end recording | `1.5` |
| `OUTPUT_FORMAT` | Default format: `plain` or `markdown` | `markdown` |
| `SAVE_DIRECTORY` | Directory for saved documents | `./output` |

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Space` | Pause/resume listening |
| `Ctrl+S` | Save document to file |
| `Ctrl+Y` | Copy document to clipboard |
| `Ctrl+E` | Type-out (paste into focused window) |
| `Ctrl+Z` | Undo last change |
| `Ctrl+R` | Redo last undone change |
| `Tab` | Toggle plain text / Markdown format |
| `↑/↓` | Scroll document |
| `q` / `Esc` | Quit |

## How It Works

1. **Audio Capture**: `cpal` streams microphone input in real-time
2. **VAD**: Detects when you start and stop speaking using RMS energy
3. **Transcription**: Sends the speech segment to your ASR API
4. **Intent Classification**: LLM analyzes the transcript and classifies it:
   - *Dictate*: Cleans up filler words, fixes grammar, formats as written text
   - *Command*: Extracts editing action (new paragraph, delete, undo, etc.)
   - *Conversation*: Responds briefly without modifying the document
5. **Document Update**: Applies the classified action to the document buffer
6. **Display**: Real-time TUI shows the document, transcript log, and status

## Examples

**Dictation:**
> "Um, so the project deadline is uh next Friday I think"
> → Types: "The project deadline is next Friday."

**Command:**
> "Delete the last sentence"
> → Removes the last sentence from the document

**Conversation:**
> "What have I written so far?"
> → Shows a brief summary without modifying the document

**Mandarin:**
> "第一点，我们需要完成市场调研"
> → Types: "第一点，我们需要完成市场调研。"

## Architecture

```
Microphone → cpal → VAD → ASR API → LLM Intent Classifier
                                          ↓
                              ┌───────────┼───────────┐
                              ↓           ↓           ↓
                          Dictate     Command    Conversation
                              ↓           ↓           ↓
                          Document ← Edit Action   Response
                              ↓
                     TUI / Clipboard / Type-out
```

## License

MIT
