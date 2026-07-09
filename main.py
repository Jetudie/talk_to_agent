import os
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Configuration
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)

# Initialize TTS
engine = pyttsx3.init()
# Optional: tweak speech rate or voice
# rate = engine.getProperty('rate')
# engine.setProperty('rate', rate - 20)

def speak(text):
    print(f"Agent: {text}")
    engine.say(text)
    engine.runAndWait()

# Initialize LLM Client
if LLM_BACKEND == "ollama":
    try:
        import ollama
        print(f"Initialized Ollama backend with model '{OLLAMA_MODEL}'.")
    except ImportError:
        print("Error: 'ollama' package not installed. Please run: pip install ollama")
        exit(1)
elif LLM_BACKEND == "openai":
    try:
        from openai import OpenAI
        client_kwargs = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            client_kwargs["base_url"] = OPENAI_BASE_URL
        openai_client = OpenAI(**client_kwargs)
        print(f"Initialized OpenAI backend with model '{OPENAI_MODEL}'.")
    except ImportError:
        print("Error: 'openai' package not installed. Please run: pip install openai")
        exit(1)
elif LLM_BACKEND == "opencode":
    print("Initialized Opencode backend. Ensure the 'opencode' CLI is installed and configured.")
else:
    print(f"Unknown LLM_BACKEND: {LLM_BACKEND}")
    exit(1)

def query_llm(messages):
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
            import subprocess
            # We just pass the latest message to opencode
            latest_message = messages[-1]["content"]
            result = subprocess.run(
                ["opencode", "run", latest_message], 
                capture_output=True, 
                text=True, 
                check=True
            )
            # You might want to clean up ANSI escape codes if opencode returns styled text,
            # but for now we just return the raw stdout.
            return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Opencode failed: {e.stderr}"
    except Exception as e:
        return f"I encountered an error connecting to my brain. Details: {e}"

def main():
    recognizer = sr.Recognizer()
    
    # Context window to keep track of conversation
    messages = [
        {"role": "system", "content": "You are a helpful and concise voice assistant. Since you are speaking, keep your answers relatively short and conversational. Do not use markdown like asterisks or code blocks if possible, as it will be read aloud."}
    ]
    
    speak("I am starting up. Please wait a moment while I adjust to the background noise.")
    
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        speak("I'm ready. You can start talking.")
        
        while True:
            try:
                print("\nListening...")
                # listen() blocks until speech is detected and finished
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=15)
                
                print("Transcribing...")
                # We use Google's free Web Speech API for ease. 
                # Can be replaced with recognizer.recognize_whisper() for offline.
                text = recognizer.recognize_google(audio)
                print(f"You: {text}")
                
                # Check for an exit command
                if text.lower() in ["exit", "quit", "stop listening", "goodbye"]:
                    speak("Goodbye!")
                    break
                
                # Add user input to history
                messages.append({"role": "user", "content": text})
                
                # Query LLM
                response_text = query_llm(messages)
                
                # Add assistant response to history
                messages.append({"role": "assistant", "content": response_text})
                
                # Speak response
                speak(response_text)
                
            except sr.WaitTimeoutError:
                pass # Nobody spoke within timeout
            except sr.UnknownValueError:
                print("Could not understand audio.")
            except sr.RequestError as e:
                print(f"Could not request results from STT service; {e}")
                speak("I'm having trouble with my speech recognition service.")
            except KeyboardInterrupt:
                print("\nStopping...")
                speak("Goodbye!")
                break
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
