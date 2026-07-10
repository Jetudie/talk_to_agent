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

### Whisper STT
- **`WHISPER_MODEL_SIZE`**: `tiny`, `base`, `small`, `medium`, or `large-v3`. (Default: `base`).
- **`WHISPER_DEVICE`**: `auto` (uses GPU if available), `cpu`, or `cuda`.
- **`WHISPER_LANGUAGE`**: e.g., `en`, `zh`. Leave blank for auto-detect.

### Audio Recording
- **`SILENCE_THRESHOLD`**: RMS energy threshold (Default: `500`). Decrease if it's struggling to pick up quiet speech; increase if background noise is triggering it.
- **`SILENCE_DURATION`**: Seconds of silence required to consider you "finished" speaking (Default: `1.5`).

## Usage

Run the main script:
```bash
python main.py
```

1. Wait for the agent to announce: *"I'm ready. You can start talking."*
2. Speak your prompt or question clearly.
3. The agent will transcribe your audio, process the response through the configured backend, and speak it aloud.
4. To stop the agent, say **"exit"**, **"quit"**, or **"goodbye"**. You can also press `Ctrl+C` in the terminal.
