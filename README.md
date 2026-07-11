# Local Voice Agent

A Python-based AI agent that you can talk to directly. It listens to your voice, transcribes it locally, processes the command through a chosen Language Model (LLM) backend, and speaks the response back to you.

## Features

- **Local Speech-to-Text (STT)**: Uses `faster-whisper` for fast, offline transcription.
- **Voice Activity Detection (VAD)**: Automatically detects when you start and stop speaking.
- **Flexible LLM Backends**:
  - **Ollama**: Run models entirely locally and free (e.g., `llama3`, `phi3`).
  - **OpenAI Compatible**: Connect to OpenAI or any compatible API (e.g., LM Studio, vLLM).
  - **Opencode**: Forward your voice commands directly to a running `opencode serve` instance to give the agent full file and environment access.
- **Local Text-to-Speech (TTS)**: Uses `pyttsx3` to synthesize speech offline.
- **Persistent Memory**: File-based memory system that survives across sessions, with systematic session handover.

## Prerequisites

- Python 3.8+
- A working microphone
- (Optional) [Ollama](https://ollama.com/) installed if using the `ollama` backend.
- (Optional) `opencode` CLI installed and running if using the `opencode` backend.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   > Note: `pyaudio` handles microphone access. On Windows, it usually installs without issues. On macOS/Linux, you might need to install `portaudio` first (e.g., `brew install portaudio` or `sudo apt install portaudio19-dev`).

2. **Configure the Environment**:
   Copy `.env.example` to a new file named `.env`:
   ```bash
   cp .env.example .env
   ```
   *(On Windows Command Prompt, use `copy .env.example .env`)*

## Configuration (`.env`)

Open the `.env` file and customize the settings based on your needs:

### LLM Backend
Set `LLM_BACKEND` to one of the following:
- `ollama`: Requires Ollama to be running locally. Set `OLLAMA_MODEL` to your desired model.
- `openai`: Requires `OPENAI_API_KEY` to be set. You can also override `OPENAI_BASE_URL` to point to a local server.
- `opencode`: Requires an `opencode` server to be running.
  - Start the server using: `opencode serve --port 4096`
  - Ensure `OPENCODE_SERVER_URL` in your `.env` points to `http://127.0.0.1:4096`.
  - Optionally set `OPENCODE_MODEL` and `OPENCODE_AGENT` to use a specific model/agent with pre-configured system prompts.

### Whisper STT
- **`WHISPER_MODEL_SIZE`**: `tiny`, `base`, `small`, `medium`, or `large-v3`. (Default: `base`).
- **`WHISPER_DEVICE`**: `auto` (uses GPU if available), `cpu`, or `cuda`.
- **`WHISPER_LANGUAGE`**: e.g., `en`, `zh`. Leave blank for auto-detect.

### Audio Recording
- **`SILENCE_THRESHOLD`**: RMS energy threshold (Default: `500`). Decrease if it's struggling to pick up quiet speech; increase if background noise is triggering it.
- **`SILENCE_DURATION`**: Seconds of silence required to consider you "finished" speaking (Default: `1.5`).

### Memory
- **`MEMORY_DIR`**: Directory for persistent memory files (Default: `memory`).

## Usage

Run the main script:
```bash
python main.py
```

1. Wait for the agent to announce: *"I'm ready. You can start talking."*
2. Speak your prompt or question clearly.
3. The agent will transcribe your audio, process the response through the configured backend, and speak it aloud.
4. To stop the agent, say **"exit"**, **"quit"**, or **"goodbye"**. You can also press `Ctrl+C` in the terminal.

## Memory System

When using the `opencode` backend, the agent has a persistent memory system that stores information in files so it can remember things across sessions.

### Directory Structure

```
memory/
├── context/                ← Always injected into every request
│   ├── summary.md          ← Rolling summary of key facts, user preferences, recurring topics
│   ├── notes.md            ← Quick notes ("remember that...", "make a note...")
│   └── last_session.md     ← Detailed handover from the most recent session
└── tasks/                  ← Task-oriented files
    ├── todo.md             ← Active to-do list (always injected)
    └── done.md             ← Completed tasks log (injected on-demand)
```

### How It Works

**Context files** (`context/`) are always included in every request so the agent has continuity:
- `summary.md` — Long-term facts and preferences learned about you.
- `notes.md` — Things you explicitly asked the agent to remember.
- `last_session.md` — A detailed handover from the previous session so the agent knows exactly where you left off.

**Task files** (`tasks/`) track your to-do items:
- `todo.md` — Active tasks. Always included so the agent knows what's pending.
- `done.md` — Completed tasks with timestamps. Only included when you ask about completed/finished tasks (to save tokens).

### Session Handover

When you end a session (by saying "goodbye" or pressing Ctrl+C), the agent automatically:
1. Writes a detailed handover to `last_session.md` covering what was discussed, decisions made, and open threads.
2. Updates `summary.md` with any new long-term information learned.
3. Ensures `todo.md` accurately reflects the current task state.

When you start a new session, all this context is injected into the first message, giving the agent seamless continuity.

### Resetting Memory

To start fresh, simply delete the `memory/` directory:
```bash
rm -rf memory/
```
It will be re-created with empty seed files on the next run.

## Project Files

| File | Description |
|---|---|
| `main.py` | Core voice agent script |
| `.env.example` | Template for environment configuration |
| `AGENTS.md` | Instructions for the opencode agent (memory rules, response style) |
| `requirements.txt` | Python dependencies |
| `memory/` | Persistent memory directory (auto-created) |
