import glob
import logging
import os
import re
import struct
import subprocess
import wave
import tempfile
from datetime import datetime
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
MAX_SESSION_ARCHIVES = int(os.getenv("MAX_SESSION_ARCHIVES", "10"))  # max archived sessions to keep on disk
SESSION_CONTEXT_COUNT = int(os.getenv("SESSION_CONTEXT_COUNT", "3"))  # recent sessions to inject into context

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

# Module-level references (initialized in init())
whisper_model = None
openai_client = None

def speak(text: str) -> None:
    print(f"Agent: {text}")
    try:
        # Initialize TTS
        engine = pyttsx3.init()
        # Optional: tweak speech rate or voice
        # rate = engine.getProperty('rate')
        # engine.setProperty('rate', rate - 20)

        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logger.warning("TTS failed to speak text: %s", e)

def init() -> None:
    """Initialize TTS engine, Whisper model, and LLM client."""
    global whisper_model, openai_client

    # Initialize Whisper model
    logger.info("Loading Faster-Whisper model '%s' on device '%s'...", WHISPER_MODEL_SIZE, WHISPER_DEVICE)
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE)
    logger.info("Whisper model loaded.")

    # Initialize LLM Client
    if LLM_BACKEND == "ollama":
        try:
            import ollama as _ollama
            globals()["ollama"] = _ollama
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
        os.path.join(MEMORY_DIR, "context", "sessions"),
        os.path.join(MEMORY_DIR, "episodes"),
        os.path.join(MEMORY_DIR, "tasks"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    seed_files = {
        os.path.join(MEMORY_DIR, "context", "summary.md"): "# Rolling Summary\n\n*Active summary of recent and important facts about the user. Keep concise — rotate older facts to summary_archive.md.*\n",
        os.path.join(MEMORY_DIR, "context", "summary_archive.md"): "# Summary Archive\n\n*Long-term archive of older facts, preferences, and context rotated out of the active summary.*\n",
        os.path.join(MEMORY_DIR, "context", "notes.md"): "# Notes\n\n*This file is maintained by the voice agent. It stores quick notes and facts the user asked the agent to remember.*\n",
        os.path.join(MEMORY_DIR, "context", "last_session.md"): "# Last Session\n\nNo previous session recorded.\n",
        os.path.join(MEMORY_DIR, "tasks", "todo.md"): "# To-Do List\n\n*This file is maintained by the voice agent. It contains active tasks.*\n",
        os.path.join(MEMORY_DIR, "tasks", "done.md"): "# Completed Tasks\n\n*This file is maintained by the voice agent. It logs completed tasks with timestamps.*\n",
    }
    for filepath, content in seed_files.items():
        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

def archive_last_session() -> None:
    """Archive the current last_session.md before it gets overwritten.

    Saves to memory/context/sessions/<timestamp>.md and prunes old archives
    beyond MAX_SESSION_ARCHIVES.
    """
    last_session_path = os.path.join(MEMORY_DIR, "context", "last_session.md")
    content = read_memory_file(last_session_path)
    if not content or "No previous session recorded" in content:
        return  # nothing worth archiving

    sessions_dir = os.path.join(MEMORY_DIR, "context", "sessions")
    os.makedirs(sessions_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    archive_path = os.path.join(sessions_dir, f"{timestamp}.md")
    try:
        with open(archive_path, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        logger.info("Archived last session to %s", archive_path)
    except Exception as e:
        logger.warning("Failed to archive last session: %s", e)
        return

    # Prune old archives beyond MAX_SESSION_ARCHIVES
    archives = sorted(glob.glob(os.path.join(sessions_dir, "*.md")))
    if len(archives) > MAX_SESSION_ARCHIVES:
        for old in archives[:len(archives) - MAX_SESSION_ARCHIVES]:
            try:
                os.remove(old)
                logger.debug("Pruned old session archive: %s", old)
            except Exception as e:
                logger.warning("Failed to prune archive %s: %s", old, e)

def get_recent_sessions(count: int = SESSION_CONTEXT_COUNT) -> str:
    """Read the most recent archived sessions and return them as context."""
    sessions_dir = os.path.join(MEMORY_DIR, "context", "sessions")
    if not os.path.isdir(sessions_dir):
        return ""
    archives = sorted(glob.glob(os.path.join(sessions_dir, "*.md")))
    recent = archives[-count:] if len(archives) >= count else archives
    if not recent:
        return ""

    parts = []
    for path in reversed(recent):  # newest first
        name = os.path.splitext(os.path.basename(path))[0]
        content = read_memory_file(path)
        if content:
            parts.append(f"### Session {name}\n{content}")
    return "\n\n".join(parts)

def log_episode(user_text: str, assistant_text: str) -> None:
    """Append a user/assistant exchange to today's episode log.

    Episodes are stored as daily files in memory/episodes/YYYY-MM-DD.md.
    Each entry includes a timestamp for precise recall.
    """
    episodes_dir = os.path.join(MEMORY_DIR, "episodes")
    os.makedirs(episodes_dir, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    episode_path = os.path.join(episodes_dir, f"{today}.md")
    timestamp = datetime.now().strftime("%H:%M:%S")

    entry = (
        f"\n---\n"
        f"**[{timestamp}]**\n\n"
        f"User: {user_text}\n\n"
        f"Assistant: {assistant_text}\n"
    )

    try:
        # Create file with header if it doesn't exist
        if not os.path.exists(episode_path):
            with open(episode_path, "w", encoding="utf-8") as f:
                f.write(f"# Episode Log — {today}\n")
        with open(episode_path, "a", encoding="utf-8") as f:
            f.write(entry)
        logger.debug("Logged episode to %s", episode_path)
    except Exception as e:
        logger.warning("Failed to log episode: %s", e)

# Stop words to filter out when extracting search keywords
_STOP_WORDS = frozenset([
    "i", "me", "my", "we", "you", "your", "it", "its", "he", "she", "they",
    "the", "a", "an", "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "can", "may", "might", "shall", "to", "of", "in", "for", "on", "with", "at",
    "by", "from", "as", "into", "about", "that", "this", "what", "which", "who",
    "when", "where", "how", "not", "no", "but", "or", "and", "if", "so", "than",
    "too", "very", "just", "also", "up", "out", "there", "here", "all", "some",
    "any", "each", "more", "most", "other", "then", "now", "only", "even", "still",
    "tell", "said", "say", "know", "think", "want", "like", "get", "go", "make",
    "see", "come", "take", "give", "good", "new", "well", "way", "thing", "much",
    "right", "great", "old", "big", "little", "long", "time", "day", "back",
])

MAX_RETRIEVAL_SNIPPETS = int(os.getenv("MAX_RETRIEVAL_SNIPPETS", "5"))  # max relevant snippets to inject

def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text by removing stop words and short tokens."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 2]

def search_episodes(query: str, max_results: int = MAX_RETRIEVAL_SNIPPETS) -> str:
    """Search episode logs and session archives for content relevant to the query.

    Uses keyword matching with scoring: each passage (separated by --- in episodes,
    or full file for sessions) is scored by the number of matching keywords.
    Returns the top-N most relevant snippets formatted for injection into context.
    """
    keywords = _extract_keywords(query)
    if not keywords:
        return ""

    scored_snippets: list[tuple[int, str, str]] = []  # (score, source_label, snippet)

    # Search episode logs
    episodes_dir = os.path.join(MEMORY_DIR, "episodes")
    if os.path.isdir(episodes_dir):
        for ep_file in sorted(glob.glob(os.path.join(episodes_dir, "*.md")), reverse=True):
            date_label = os.path.splitext(os.path.basename(ep_file))[0]
            content = read_memory_file(ep_file)
            if not content:
                continue
            # Split into individual exchanges (separated by ---)
            passages = [p.strip() for p in content.split("---") if p.strip()]
            for passage in passages:
                passage_lower = passage.lower()
                score = sum(1 for kw in keywords if kw in passage_lower)
                if score > 0:
                    scored_snippets.append((score, f"Episode {date_label}", passage))

    # Search session archives
    sessions_dir = os.path.join(MEMORY_DIR, "context", "sessions")
    if os.path.isdir(sessions_dir):
        for sess_file in sorted(glob.glob(os.path.join(sessions_dir, "*.md")), reverse=True):
            sess_label = os.path.splitext(os.path.basename(sess_file))[0]
            content = read_memory_file(sess_file)
            if not content:
                continue
            content_lower = content.lower()
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                # Truncate long session summaries to keep context manageable
                truncated = content[:500] + "..." if len(content) > 500 else content
                scored_snippets.append((score, f"Session {sess_label}", truncated))

    if not scored_snippets:
        return ""

    # Sort by score descending and take top results
    scored_snippets.sort(key=lambda x: x[0], reverse=True)
    top = scored_snippets[:max_results]

    parts = []
    for score, source, snippet in top:
        parts.append(f"[From: {source} | relevance: {score}]\n{snippet}")
    return "\n\n".join(parts)

def build_memory_context(user_message: str) -> str:
    """Build a memory context block from persistent memory files."""
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
        "## Rolling Summary (active)",
        summary,
        "",
        "## Summary Archive (long-term)",
        read_memory_file(os.path.join(MEMORY_DIR, "context", "summary_archive.md")),
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

    # Include recent session archives for multi-session recall
    recent_sessions = get_recent_sessions()
    if recent_sessions:
        context_parts.extend(["", "## Previous Sessions (recent)", recent_sessions])

    # Semantic retrieval: search episodes and old sessions for relevant content
    if user_message:
        relevant = search_episodes(user_message)
        if relevant:
            context_parts.extend(["", "## Relevant Past Conversations", relevant])

    context_parts.extend(["", "[END MEMORY CONTEXT]"])

    return "\n".join(context_parts)

# Regex to match <memory file="..." mode="...">content</memory> blocks
_MEMORY_TAG_RE = re.compile(
    r'<memory\s+file="([^"]+)"\s+mode="(overwrite|append)">\s*\n?(.*?)\n?\s*</memory>',
    re.DOTALL,
)

def parse_memory_updates(response: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Parse <memory> tags from LLM response.

    Returns:
        A tuple of (cleaned_response, updates) where updates is a list of
        (relative_file_path, mode, content) tuples.
    """
    updates: list[tuple[str, str, str]] = []
    for match in _MEMORY_TAG_RE.finditer(response):
        file_path, mode, content = match.group(1), match.group(2), match.group(3)
        updates.append((file_path, mode, content.strip()))

    # Strip all <memory> tags from the response for TTS
    cleaned = _MEMORY_TAG_RE.sub("", response).strip()
    return cleaned, updates

def apply_memory_updates(updates: list[tuple[str, str, str]]) -> None:
    """Write parsed memory updates to disk.

    Each update is a (relative_file_path, mode, content) tuple.
    Only writes to files within MEMORY_DIR (rejects path traversal).
    """
    for file_rel, mode, content in updates:
        # Resolve and validate the target path is within MEMORY_DIR
        target = os.path.normpath(os.path.join(MEMORY_DIR, file_rel))
        if not target.startswith(os.path.normpath(MEMORY_DIR) + os.sep) and \
           target != os.path.normpath(MEMORY_DIR):
            logger.warning("Rejected memory write outside MEMORY_DIR: %s", file_rel)
            continue

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(target), exist_ok=True)

        try:
            if mode == "append":
                with open(target, "a", encoding="utf-8") as f:
                    f.write("\n" + content + "\n")
            else:  # overwrite
                with open(target, "w", encoding="utf-8") as f:
                    f.write(content + "\n")
            logger.info("Memory updated [%s]: %s", mode, file_rel)
        except Exception as e:
            logger.warning("Failed to write memory file %s: %s", file_rel, e)

def build_system_prompt(user_message: str = "") -> str:
    """Build the system prompt with memory context included."""
    base_prompt = (
        "You are a helpful and concise voice assistant. "
        "Since you are speaking, keep your answers relatively short and conversational. "
        "Do not use markdown like asterisks or code blocks if possible, as it will be read aloud."
    )
    memory_context = build_memory_context(user_message)
    prompt = f"{base_prompt}\n\n{memory_context}"

    # For ollama/openai, add instructions on how to emit memory updates via XML tags.
    # Opencode has direct filesystem access via AGENTS.md so it doesn't need this.
    if LLM_BACKEND in ("ollama", "openai"):
        prompt += (
            "\n\n[MEMORY UPDATE INSTRUCTIONS]\n"
            "You can persist information across sessions by including <memory> tags in your response. "
            "These tags are silently processed and will NOT be spoken aloud. "
            "Use them whenever you need to save notes, update tasks, or record facts about the user.\n\n"
            "Tag format:\n"
            '<memory file="<path>" mode="overwrite|append">\ncontent here\n</memory>\n\n'
            "Available files (path is relative to the memory directory):\n"
            '- context/summary.md — Active rolling summary of recent key facts (keep under 50 lines). Use mode="overwrite".\n'
            '- context/summary_archive.md — Long-term archive for older facts rotated out of summary.md. Use mode="append" to add, or mode="overwrite" to reorganize.\n'
            '- context/notes.md — Quick notes the user asked you to remember. Use mode="append" to add entries.\n'
            '- tasks/todo.md — Active to-do items. Use mode="overwrite" with the full updated list.\n'
            '- tasks/done.md — Completed tasks log. Use mode="append" to add entries with timestamps.\n'
            '- context/last_session.md — Session handover (used at session end only). Use mode="overwrite".\n\n'
            "Rules:\n"
            "- When the user says 'remember that...' or 'make a note...', append to context/notes.md.\n"
            "- When the user adds a task, overwrite tasks/todo.md with the full updated list.\n"
            "- When a task is completed, remove it from tasks/todo.md and append it to tasks/done.md with a timestamp.\n"
            "- When updating summary.md, keep it under 50 lines. If it's getting long, move older or less relevant "
            "facts to context/summary_archive.md (append) before overwriting summary.md with the condensed version.\n"
            "- Always include the <memory> tags AFTER your spoken response.\n"
            "- You may include multiple <memory> tags in a single response.\n"
            "[END MEMORY UPDATE INSTRUCTIONS]"
        )

    return prompt

def build_handover_message() -> str:
    """Build the shutdown handover message.

    For opencode: instructs the agent to write files directly.
    For ollama/openai: instructs the LLM to emit <memory> tags.
    """
    if LLM_BACKEND == "opencode":
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
    else:
        return (
            "[SESSION ENDING — HANDOVER REQUIRED]\n"
            "The voice session is ending. Please perform the following handover steps using <memory> tags:\n"
            '1. Write a detailed session handover using <memory file="context/last_session.md" mode="overwrite"> covering: '
            "date/time, topics discussed, key decisions, tasks added/completed/in-progress, "
            "open questions, and user priorities.\n"
            "2. If you learned any new long-term facts or preferences, update the rolling summary using "
            '<memory file="context/summary.md" mode="overwrite">.\n'
            '3. Ensure the to-do list is accurate using <memory file="tasks/todo.md" mode="overwrite">.\n'
            "Respond with a brief spoken confirmation of what you saved, followed by your <memory> tags."
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
        check=True,
        encoding="utf-8",
        shell=True
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
            # Refresh system prompt with latest memory context
            messages = [{"role": "system", "content": build_system_prompt(messages[-1]["content"])}] + messages[1:]
            response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
            return response['message']['content']
        elif LLM_BACKEND == "openai":
            # Refresh system prompt with latest memory context
            messages = [{"role": "system", "content": build_system_prompt(messages[-1]["content"])}] + messages[1:]
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages
            )
            return response.choices[0].message.content
        elif LLM_BACKEND == "opencode":
            latest_message = messages[-1]["content"]

            # Include conversation history (exclude system prompt at index 0)
            history_parts = []
            for msg in messages[1:]:  # skip system prompt
                role = "You" if msg["role"] == "user" else "Assistant"
                history_parts.append(f"{role}: {msg['content']}")

            history_block = ""
            if history_parts:
                history_block = "\n\n## Recent Conversation\n" + "\n".join(history_parts) + "\n"

            # Build memory context
            memory_context = build_memory_context(latest_message)

            full_message = (
                f"User's message: {latest_message}\n"
                f"{history_block}"
                f"\n{memory_context}"
            )

            try:
                return run_opencode(full_message)
            except subprocess.CalledProcessError as e:
                return f"Opencode failed: {e.stderr}"
    except Exception as e:
        return f"I encountered an error connecting to my brain. Details: {e}"

def perform_handover(messages: list[dict[str, str]] | None = None) -> None:
    """Perform session handover to persist state across sessions.

    For opencode: sends handover message via run_opencode (direct filesystem access).
    For ollama/openai: sends handover prompt via query_llm and parses <memory> tags.
    """
    try:
        logger.info("Performing session handover...")
        # Archive the current session before it gets overwritten
        archive_last_session()
        handover_msg = build_handover_message()

        if LLM_BACKEND == "opencode":
            response = run_opencode(handover_msg)
            logger.info("Handover complete: %s", response)
        else:
            # Build a temporary messages list with the handover prompt
            handover_messages = [
                {"role": "system", "content": build_system_prompt()}
            ]
            # Include recent conversation context if available
            if messages and len(messages) > 1:
                handover_messages.extend(messages[1:])  # skip original system prompt
            handover_messages.append({"role": "user", "content": handover_msg})

            response = query_llm(handover_messages)
            cleaned, updates = parse_memory_updates(response)
            if updates:
                apply_memory_updates(updates)
                logger.info("Handover complete: %d memory files updated.", len(updates))
            else:
                logger.warning("Handover produced no memory updates. Response: %s", cleaned)
    except Exception as e:
        logger.warning("Handover failed: %s", e)

def main() -> None:
    # Initialize all components (TTS, Whisper, LLM client)
    init()

    # Ensure memory directory exists with seed files
    ensure_memory_dir()

    pa = pyaudio.PyAudio()
    
    # Context window to keep track of conversation
    messages = [
        {"role": "system", "content": build_system_prompt()}
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
                    perform_handover(messages)
                    break
                
                # Add user input to history
                messages.append({"role": "user", "content": text})
                
                # Query LLM
                response_text = query_llm(messages)
                
                # Parse and apply any memory updates from the response
                cleaned_response, memory_updates = parse_memory_updates(response_text)
                if memory_updates:
                    apply_memory_updates(memory_updates)
                
                # Add cleaned response (without memory tags) to history
                messages.append({"role": "assistant", "content": cleaned_response})
                
                # Trim conversation history to prevent unbounded growth
                messages = trim_messages(messages)
                
                # Speak cleaned response
                speak(cleaned_response)
                
                # Log the exchange to episodic memory
                log_episode(text, cleaned_response)
                
            except KeyboardInterrupt:
                logger.info("Stopping...")
                speak("Saving session and shutting down. Goodbye!")
                perform_handover(messages)
                break
            except Exception as e:
                logger.error("An unexpected error occurred: %s", e, exc_info=True)
    finally:
        pa.terminate()

if __name__ == "__main__":
    main()
