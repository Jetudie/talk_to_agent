import logging
import os
import re
import struct
import subprocess
import wave
import tempfile
import numpy as np
import pyaudio
import pyttsx3
from faster_whisper import WhisperModel
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configuration
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
OPENCODE_MODEL = os.getenv("OPENCODE_MODEL", None)
OPENCODE_AGENT = os.getenv("OPENCODE_AGENT", None)
OPENCODE_SERVER_URL = os.getenv("OPENCODE_SERVER_URL", "http://127.0.0.1:4096")

# Memory configuration
MEMORY_DIR = os.getenv("MEMORY_DIR", "memory")

# Whisper STT configuration
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # "cpu", "cuda", or "auto"
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", None)  # e.g. "en", "zh", None for auto-detect

# Audio recording parameters
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024  # frames per buffer
FORMAT = pyaudio.paInt16
SILENCE_THRESHOLD = int(os.getenv("SILENCE_THRESHOLD", "500"))  # RMS energy threshold
SILENCE_DURATION = float(os.getenv("SILENCE_DURATION", "1.5"))  # seconds of silence to stop recording
MAX_RECORD_SECONDS = float(os.getenv("MAX_RECORD_SECONDS", "30"))  # max recording length
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))  # max user+assistant messages to keep

# Initialize TTS
engine = pyttsx3.init()
# Optional: tweak speech rate or voice
# rate = engine.getProperty('rate')
# engine.setProperty('rate', rate - 20)

def speak(text: str) -> None:
    print(f"Agent: {text}")
    engine.say(text)
    engine.runAndWait()

# Initialize Whisper model
logger.info("Loading Faster-Whisper model '%s' on device '%s'...", WHISPER_MODEL_SIZE, WHISPER_DEVICE)
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE)
logger.info("Whisper model loaded.")

# Initialize LLM Client
if LLM_BACKEND == "ollama":
    try:
        import ollama
        logger.info("Initialized Ollama backend with model '%s'.", OLLAMA_MODEL)
    except ImportError:
        logger.error("'ollama' package not installed. Please run: pip install ollama")
        exit(1)
elif LLM_BACKEND == "openai":
    try:
        from openai import OpenAI
        client_kwargs = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            client_kwargs["base_url"] = OPENAI_BASE_URL
        openai_client = OpenAI(**client_kwargs)
        logger.info("Initialized OpenAI backend with model '%s'.", OPENAI_MODEL)
    except ImportError:
        logger.error("'openai' package not installed. Please run: pip install openai")
        exit(1)
elif LLM_BACKEND == "opencode":
    logger.info("Initialized Opencode backend. Ensure the 'opencode' CLI is installed and configured.")
else:
    logger.error("Unknown LLM_BACKEND: %s", LLM_BACKEND)
    exit(1)

def get_rms(data: bytes) -> float:
    """Calculate the root mean square (RMS) energy of an audio chunk."""
    count = len(data) // 2  # 16-bit = 2 bytes per sample
    shorts = struct.unpack(f"<{count}h", data)
    sum_squares = sum(s * s for s in shorts)
    return (sum_squares / count) ** 0.5 if count > 0 else 0

def record_speech(pa: pyaudio.PyAudio) -> bytes | None:
    """Record audio from the microphone until silence is detected. Returns raw PCM bytes or None."""
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    frames = []
    silent_chunks = 0
    has_speech = False
    chunks_per_second = SAMPLE_RATE / CHUNK_SIZE
    max_chunks = int(MAX_RECORD_SECONDS * chunks_per_second)
    silence_chunks_needed = int(SILENCE_DURATION * chunks_per_second)

    try:
        for _ in range(max_chunks):
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            rms = get_rms(data)

            if rms >= SILENCE_THRESHOLD:
                has_speech = True
                silent_chunks = 0
                frames.append(data)
            elif has_speech:
                # Speech was detected before, now counting silence
                silent_chunks += 1
                frames.append(data)
                if silent_chunks >= silence_chunks_needed:
                    break
            # else: still waiting for speech to start, discard ambient noise
    finally:
        stream.stop_stream()
        stream.close()

    if not has_speech:
        return None

    return b"".join(frames)

def transcribe(audio_bytes: bytes) -> str:
    """Transcribe raw PCM audio bytes using Faster-Whisper. Returns the transcribed text."""
    # Convert raw PCM bytes to a float32 numpy array (what faster-whisper expects)
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    kwargs = {}
    if WHISPER_LANGUAGE:
        kwargs["language"] = WHISPER_LANGUAGE

    segments, info = whisper_model.transcribe(audio_float32, beam_size=5, **kwargs)
    text = " ".join(segment.text for segment in segments).strip()
    return text

# --- Memory System Helpers ---

def read_memory_file(filepath: str) -> str:
    """Read a memory file and return its contents, or empty string if it doesn't exist."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""
    except Exception as e:
        logger.warning("Could not read %s: %s", filepath, e)
        return ""

def ensure_memory_dir() -> None:
    """Create the memory directory structure and seed files if they don't exist."""
    dirs = [
        os.path.join(MEMORY_DIR, "context"),
        os.path.join(MEMORY_DIR, "tasks"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    seed_files = {
        os.path.join(MEMORY_DIR, "context", "summary.md"): "# Rolling Summary\n\n*This file is maintained by the voice agent. It contains a concise summary of key facts, user preferences, and recurring topics learned across sessions.*\n",
        os.path.join(MEMORY_DIR, "context", "notes.md"): "# Notes\n\n*This file is maintained by the voice agent. It stores quick notes and facts the user asked the agent to remember.*\n",
        os.path.join(MEMORY_DIR, "context", "last_session.md"): "# Last Session\n\nNo previous session recorded.\n",
        os.path.join(MEMORY_DIR, "tasks", "todo.md"): "# To-Do List\n\n*This file is maintained by the voice agent. It contains active tasks.*\n",
        os.path.join(MEMORY_DIR, "tasks", "done.md"): "# Completed Tasks\n\n*This file is maintained by the voice agent. It logs completed tasks with timestamps.*\n",
    }
    for filepath, content in seed_files.items():
        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

def build_memory_context(user_message: str) -> str:
    """Build a memory context block to prepend to the user's message for opencode."""
    # Always-injected files
    last_session = read_memory_file(os.path.join(MEMORY_DIR, "context", "last_session.md"))
    summary = read_memory_file(os.path.join(MEMORY_DIR, "context", "summary.md"))
    notes = read_memory_file(os.path.join(MEMORY_DIR, "context", "notes.md"))
    todo = read_memory_file(os.path.join(MEMORY_DIR, "tasks", "todo.md"))

    # On-demand: include done.md only when relevant keywords appear
    done_keywords = ["done", "completed", "finished", "history", "past tasks", "what did i finish", "accomplishments"]
    include_done = any(kw in user_message.lower() for kw in done_keywords)

    context_parts = [
        "[MEMORY CONTEXT — Read this carefully before responding]",
        "",
        "## Last Session Handover",
        last_session,
        "",
        "## Rolling Summary",
        summary,
        "",
        "## Notes",
        notes,
        "",
        "## Active To-Do",
        todo,
    ]

    if include_done:
        done = read_memory_file(os.path.join(MEMORY_DIR, "tasks", "done.md"))
        context_parts.extend(["", "## Completed Tasks", done])

    context_parts.extend(["", "[END MEMORY CONTEXT]"])

    return "\n".join(context_parts)

def build_handover_message() -> str:
    """Build the shutdown handover message to send to opencode."""
    return (
        "[SESSION ENDING — HANDOVER REQUIRED]\n"
        "The voice session is ending. Please perform the following handover steps:\n"
        "1. Write a detailed handover to memory/context/last_session.md covering:\n"
        "   - Date/time of this session\n"
        "   - Topics discussed and key decisions made\n"
        "   - Any tasks that were added, completed, or are still in progress\n"
        "   - Any open questions or unfinished threads the user may want to continue\n"
        "   - User's apparent priorities if notable\n"
        "2. Update memory/context/summary.md if any new long-term facts or preferences were learned.\n"
        "3. Ensure memory/tasks/todo.md accurately reflects the current state of all tasks.\n"
        "Respond with a brief confirmation of what you saved."
    )

def run_opencode(message: str) -> str:
    """Run a message through opencode CLI and return the response."""
    cmd = ["opencode", "run", message]
    if OPENCODE_SERVER_URL:
        cmd.extend(["--attach", OPENCODE_SERVER_URL])
    if OPENCODE_MODEL:
        cmd.extend(["--model", OPENCODE_MODEL])
    if OPENCODE_AGENT:
        cmd.extend(["--agent", OPENCODE_AGENT])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip()

# --- LLM Query ---

def trim_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Trim conversation history to MAX_HISTORY_MESSAGES, preserving the system prompt."""
    # messages[0] is the system prompt; the rest are user/assistant pairs
    non_system = messages[1:]
    if len(non_system) > MAX_HISTORY_MESSAGES:
        messages = [messages[0]] + non_system[-MAX_HISTORY_MESSAGES:]
        logger.debug("Trimmed conversation history to %d messages.", MAX_HISTORY_MESSAGES)
    return messages

def query_llm(messages: list[dict[str, str]]) -> str:
    """Sends the conversation history to the chosen LLM backend and returns the response string."""
    try:
        if LLM_BACKEND == "ollama":
            response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
            return response['message']['content']
        elif LLM_BACKEND == "openai":
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages
            )
            return response.choices[0].message.content
        elif LLM_BACKEND == "opencode":
            latest_message = messages[-1]["content"]
            # Build memory context and prepend it to the user's message
            memory_context = build_memory_context(latest_message)
            full_message = f"{memory_context}\n\nUser says: {latest_message}"
            try:
                return run_opencode(full_message)
            except subprocess.CalledProcessError as e:
                return f"Opencode failed: {e.stderr}"
    except Exception as e:
        return f"I encountered an error connecting to my brain. Details: {e}"

def perform_handover() -> None:
    """Send the handover message to opencode so it can save session state."""
    if LLM_BACKEND != "opencode":
        return
    try:
        logger.info("Performing session handover...")
        handover_msg = build_handover_message()
        response = run_opencode(handover_msg)
        logger.info("Handover complete: %s", response)
    except Exception as e:
        logger.warning("Handover failed: %s", e)

def main() -> None:
    # Ensure memory directory exists with seed files
    ensure_memory_dir()

    pa = pyaudio.PyAudio()
    
    # Context window to keep track of conversation
    messages = [
        {"role": "system", "content": "You are a helpful and concise voice assistant. Since you are speaking, keep your answers relatively short and conversational. Do not use markdown like asterisks or code blocks if possible, as it will be read aloud."}
    ]
    
    speak("I'm ready. You can start talking.")
    
    try:
        while True:
            try:
                logger.debug("Listening...")
                audio_bytes = record_speech(pa)
                
                if audio_bytes is None:
                    continue  # No speech detected, keep listening
                
                logger.debug("Transcribing...")
                text = transcribe(audio_bytes)
                
                if not text or not re.search(r"[a-zA-Z0-9]", text):
                    logger.debug("Empty or noise-only transcription, skipping.")
                    continue
                    
                print(f"You: {text}")
                
                # Check for an exit command
                if text.lower().strip() in ["exit", "quit", "stop listening", "goodbye",
                                             "exit.", "quit.", "goodbye."]:
                    speak("Saving session and shutting down. Goodbye!")
                    perform_handover()
                    break
                
                # Add user input to history
                messages.append({"role": "user", "content": text})
                
                # Query LLM
                response_text = query_llm(messages)
                
                # Add assistant response to history
                messages.append({"role": "assistant", "content": response_text})
                
                # Trim conversation history to prevent unbounded growth
                messages = trim_messages(messages)
                
                # Speak response
                speak(response_text)
                
            except KeyboardInterrupt:
                logger.info("Stopping...")
                speak("Saving session and shutting down. Goodbye!")
                perform_handover()
                break
            except Exception as e:
                logger.error("An unexpected error occurred: %s", e, exc_info=True)
    finally:
        pa.terminate()

if __name__ == "__main__":
    main()
