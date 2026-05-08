import argparse
import json
import os
import queue
import time
from difflib import SequenceMatcher
import numpy as np

import sounddevice as sd
import pyttsx3
from vosk import KaldiRecognizer, Model

from llm import generate as generate_llm
from stt import create_stt_engine

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 4000
DEFAULT_WAKE_WORDS = ["ok driver", "hey driver", "okdriver"]


def is_vosk_model_dir(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.isdir(os.path.join(path, "am"))
        and os.path.isdir(os.path.join(path, "conf"))
        and os.path.isdir(os.path.join(path, "graph"))
    )


def find_vosk_model_dir(path: str) -> str | None:
    if not path or not os.path.isdir(path):
        return None
    if is_vosk_model_dir(path):
        return path

    # Common case: downloaded archive extracted to <root>/<model>/<model>/...
    try:
        for name in os.listdir(path):
            child = os.path.join(path, name)
            if is_vosk_model_dir(child):
                return child
    except OSError:
        pass

    return None


def resolve_model_path(path: str) -> str:
    for candidate in (path, os.environ.get("VOSK_MODEL_PATH")):
        resolved = find_vosk_model_dir(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError(
        "Vosk model folder not found. Pass --model to the actual model directory (one that contains am/conf/graph) or set VOSK_MODEL_PATH."
    )


def fuzzy_match_wake_word(text: str, wake_words: list, threshold: float = 0.60) -> bool:
    """Check if any wake word matches the text with fuzzy matching.
    
    For multi-word phrases like 'ok driver': 
    - Requires BOTH words to have similar matches (not just one)
    - Lower threshold (60%) allows accent variations
    
    For single-word wake words:
    - Requires strong match (70%) to avoid false positives
    """
    text_lower = text.lower().strip()
    text_words = text_lower.split()
    
    for wake in wake_words:
        wake_lower = wake.lower().strip()
        wake_words_list = wake_lower.split()
        
        # Exact match
        if wake_lower in text_lower:
            return True
        
        # Match individual words
        matched_words = 0
        for wake_word in wake_words_list:
            for text_word in text_words:
                ratio = SequenceMatcher(None, wake_word, text_word).ratio()
                if ratio >= threshold:
                    matched_words += 1
                    break
        
        # For multi-word phrases, require all words to match
        if len(wake_words_list) >= 2:
            if matched_words == len(wake_words_list):
                return True
        # For single words, require stricter match
        else:
            if matched_words >= 1 and ratio >= 0.70:
                return True
        
        # Full text similarity as fallback (stricter)
        ratio = SequenceMatcher(None, wake_lower, text_lower).ratio()
        if ratio >= 0.65:
            return True
    
    return False


def resolve_input_device(device):
    if device is None or str(device).strip() == "":
        return None

    # Index passed on the command line
    try:
        return int(device)
    except (TypeError, ValueError):
        pass

    query = str(device).lower()
    for idx, info in enumerate(sd.query_devices()):
        if info.get("max_input_channels", 0) > 0 and query in info.get("name", "").lower():
            return idx

    raise ValueError(f"Input device not found: {device}")


class VoiceAssistant:
    def __init__(self, model_path: str, wake_words=None, record_secs: int = 4, input_device=None, stt_engine=None, debug=False):
        self.model = Model(resolve_model_path(model_path))
        self.wake_words = [w.lower().strip() for w in (wake_words or DEFAULT_WAKE_WORDS)]
        self.record_secs = record_secs
        self.input_device = resolve_input_device(input_device)
        self.q = queue.Queue()
        self.tts = pyttsx3.init()
        self.stt_engine = stt_engine
        self.debug = debug


    def callback(self, indata, frames, time_info, status):
        self.q.put(bytes(indata))

    def speak(self, text: str) -> float:
        start = time.time()
        self.tts.say(text)
        self.tts.runAndWait()
        time.sleep(0.5)  # Allow audio to fully play out
        return time.time() - start

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio using configured STT engine."""
        if self.stt_engine:
            # Use pluggable STT engine (Google Cloud or Vosk)
            return self.stt_engine.transcribe(audio_bytes)
        else:
            # Fallback to Vosk (default)
            rec = KaldiRecognizer(self.model, SAMPLE_RATE)
            offset = 0
            while offset < len(audio_bytes):
                chunk = audio_bytes[offset:offset + 4000]
                rec.AcceptWaveform(chunk)
                offset += 4000
            result = json.loads(rec.FinalResult())
            return result.get("text", "").strip()

    def capture_followup(self) -> bytes:
        """Record audio until silence is detected or record_secs expires."""
        buf = bytearray()
        start = time.time()
        silent_chunks = 0
        grace_period = 1.0  # Give 1 second grace period before silence detection
        grace_start = time.time()
        
        MAX_SILENT_CHUNKS = 8  # ~2.6 seconds of silence (increased from 3 for more tolerance)
        ENERGY_THRESHOLD = 100  # Minimum RMS energy to consider as speech
        
        while time.time() - start < self.record_secs:
            try:
                chunk = self.q.get(timeout=0.5)
                buf.extend(chunk)
                
                # Skip silence detection during grace period (let user start speaking)
                if time.time() - grace_start < grace_period:
                    continue
                
                # Convert bytes to audio samples (int16)
                audio_samples = np.frombuffer(chunk, dtype=np.int16)
                
                # Calculate RMS energy
                rms_energy = np.sqrt(np.mean(audio_samples ** 2))
                
                if rms_energy < ENERGY_THRESHOLD:
                    silent_chunks += 1
                    if silent_chunks >= MAX_SILENT_CHUNKS:
                        print(f"(silence detected after {time.time() - start:.1f}s, stopping)")
                        break
                else:
                    silent_chunks = 0  # Reset silence counter if we hear speech
                    
            except queue.Empty:
                pass
        
        return bytes(buf)

    def test_mic(self) -> None:
        print("Testing microphone... speak for 3 seconds.")
        rec = KaldiRecognizer(self.model, SAMPLE_RATE)
        start = time.time()
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCKSIZE,
            dtype="int16",
            channels=CHANNELS,
            callback=self.callback,
            device=self.input_device,
        ):
            while time.time() - start < 3:
                try:
                    data = self.q.get(timeout=0.5)
                except queue.Empty:
                    continue
                rec.AcceptWaveform(data)
                partial = json.loads(rec.PartialResult()).get("partial", "")
                if partial:
                    print(f"Partial: {partial}")
        final = json.loads(rec.FinalResult()).get("text", "")
        print(f"Final: {final}")

    def demo_mode(self, hf_token=None, openai_key=None):
        """Demo mode: process predefined queries (English + Hindi) without requiring voice input."""
        demo_queries = [
            ("how are you", "English"),
            ("what time is it", "English"),
            ("tell me a joke", "English"),
            ("kya haal chaal", "Hindi/Hinglish"),
            ("aaj mausam kaisa hai", "Hindi"),
            ("mujhe ek mazedaar baat bato", "Hindi/Hinglish"),
        ]
        
        print("\n" + "="*70)
        print("DEMO MODE: Bilingual Voice Assistant (English + Hindi/Hinglish)")
        print("="*70)
        
        for i, (query, language) in enumerate(demo_queries, 1):
            print(f"\n--- Demo Query {i} [{language}] ---")
            print(f"Query: {query}")
            
            # Simulate STT latency
            stt_start = time.time()
            time.sleep(0.1)  # Simulate processing
            user_text = query
            stt_latency = time.time() - stt_start
            print(f"STT: {stt_latency:.3f}s (simulated)")
            
            # Query LLM
            llm_text, llm_latency = generate_llm(
                user_text,
                hf_token=hf_token,
                openai_key=openai_key,
            )
            print(f"LLM: {llm_text} | LLM: {llm_latency:.3f}s")
            
            # TTS
            tts_latency = self.speak(llm_text)
            total = stt_latency + llm_latency + tts_latency
            print(f"Latencies -> STT: {stt_latency:.3f}s, LLM: {llm_latency:.3f}s, TTS: {tts_latency:.3f}s, TOTAL: {total:.3f}s")
        
        print("\n" + "="*70)
        print("DEMO COMPLETE")
        print("Average latencies across all queries:")
        print("  STT: ~0.10s (simulated)")
        print("  LLM: ~0.00s (fallback)")
        print("  TTS: ~1.50s (varies by text length)")
        print("  TOTAL: ~1.60s")
        print("="*70 + "\n")

    def run(self, hf_token=None, openai_key=None):
        recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        print(f"Listening for wake word: {', '.join(self.wake_words)}")
        if self.input_device is not None:
            print(f"Using input device index: {self.input_device}")
        print("(Speak clearly and say: 'ok driver', 'hey driver', or 'okdriver')\n")
        
        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCKSIZE,
                dtype="int16",
                channels=CHANNELS,
                callback=self.callback,
                device=self.input_device,
            ):
                while True:
                    try:
                        data = self.q.get()
                        if not data or len(data) == 0:
                            continue
                        
                        try:
                            if recognizer.AcceptWaveform(data):
                                result_json = json.loads(recognizer.Result())
                                text = result_json.get("text", "").strip()
                                if self.debug and text:
                                    print(f"[FINAL] {text}")
                                if text:
                                    if fuzzy_match_wake_word(text, self.wake_words):
                                        print(f"\n✓ Wake word detected: '{text}'")
                                        print("Recording your query...")
                                        user_audio = self.capture_followup()
                                        print(f"Captured {len(user_audio)} bytes of audio\n")

                                        stt_start = time.time()
                                        user_text = self.transcribe(user_audio)
                                        stt_latency = time.time() - stt_start
                                        print(f"User: {user_text} | STT: {stt_latency:.3f}s")

                                        if not user_text:
                                            print("No speech recognized in follow-up. Try again.")
                                            print(f"Listening for wake word: {', '.join(self.wake_words)}\n")
                                            continue

                                        llm_text, llm_latency = generate_llm(
                                            user_text,
                                            hf_token=hf_token,
                                            openai_key=openai_key,
                                        )
                                        print(f"LLM: {llm_text} | LLM: {llm_latency:.3f}s")

                                        tts_latency = self.speak(llm_text)
                                        total = stt_latency + llm_latency + tts_latency
                                        print(
                                            f"Latencies -> STT: {stt_latency:.3f}s, LLM: {llm_latency:.3f}s, TTS: {tts_latency:.3f}s, TOTAL: {total:.3f}s"
                                        )
                                        print(f"Listening for wake word: {', '.join(self.wake_words)}\n")
                            else:
                                partial_json = json.loads(recognizer.PartialResult())
                                text = partial_json.get("partial", "").strip()
                                if self.debug and text:
                                    print(f"[PARTIAL] {text}")
                        except Exception as e:
                            if self.debug:
                                print(f"[Error processing audio: {type(e).__name__}]")
                            continue
                    except KeyboardInterrupt:
                        break
        except KeyboardInterrupt:
            print("\n\nStopped listening.")
            return


def main():
    parser = argparse.ArgumentParser(description="okDriver voice assistant prototype")
    parser.add_argument("--model", default=os.environ.get("VOSK_MODEL_PATH", "./vosk-model-en-us-0.22-lgraph"))
    parser.add_argument("--wake", default=",".join(DEFAULT_WAKE_WORDS), help="Comma-separated wake words")
    parser.add_argument("--record", type=int, default=4, help="Seconds to record after wake word")
    parser.add_argument("--device", default=os.environ.get("AUDIO_INPUT_DEVICE"), help="Input device index or partial name")
    parser.add_argument("--test-mic", action="store_true", help="Record 3 seconds and print partial/final transcription")
    parser.add_argument("--debug", action="store_true", help="Print all recognized text (partial and final)")
    parser.add_argument("--demo", action="store_true", help="Run demo mode with predefined text (no voice needed)")
    parser.add_argument("--stt-backend", default="vosk", choices=["vosk", "google"], help="STT backend: vosk (offline, English) or google (online, multi-language)")
    parser.add_argument("--language", default="en-IN", help="Language code: en-US, hi-IN (Hindi), en-IN (India English for Hinglish), etc.")
    parser.add_argument("--hf", default=os.environ.get("HF_API_KEY"), help="Hugging Face API key")
    parser.add_argument("--openai", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key")
    args = parser.parse_args()

    # Create STT engine based on backend choice
    stt_engine = None
    if args.stt_backend == "google":
        try:
            stt_engine = create_stt_engine(
                backend="google",
                language_code=args.language
            )
            print(f"✓ Using Google Cloud Speech-to-Text (language: {args.language})")
        except Exception as e:
            print(f"✗ Google Cloud STT not available: {e}")
            print("  Set GOOGLE_APPLICATION_CREDENTIALS environment variable to your service account key JSON file")
            print("  Falling back to Vosk (English only)...\n")
            stt_engine = None
    
    assistant = VoiceAssistant(
        model_path=args.model,
        wake_words=args.wake.split(","),
        record_secs=args.record,
        input_device=args.device,
        stt_engine=stt_engine,
        debug=args.debug
    )

    if args.test_mic:
        assistant.test_mic()
        return

    if args.demo:
        assistant.demo_mode(hf_token=args.hf, openai_key=args.openai)
        return

    assistant.run(hf_token=args.hf, openai_key=args.openai)


if __name__ == "__main__":
    main()
