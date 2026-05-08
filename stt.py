"""
Speech-to-Text abstraction layer.
Supports both Vosk (offline, English) and Google Cloud Speech-to-Text (online, multi-language).
"""

import json
from typing import Tuple
from vosk import KaldiRecognizer, Model
from google.cloud import speech


class STTEngine:
    """Base class for STT backends."""
    
    def transcribe(self, audio_bytes: bytes) -> str:
        """Convert audio bytes to text."""
        raise NotImplementedError


class VoskSTT(STTEngine):
    """Offline Vosk recognizer (English only)."""
    
    def __init__(self, model_path: str, sample_rate: int = 16000):
        self.model = Model(model_path)
        self.sample_rate = sample_rate
    
    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio using Vosk."""
        recognizer = KaldiRecognizer(self.model, self.sample_rate)
        recognizer.AcceptWaveform(audio_bytes)
        result = json.loads(recognizer.FinalResult())
        return result.get("text", "").strip()


class GoogleCloudSTT(STTEngine):
    """Google Cloud Speech-to-Text recognizer (supports 100+ languages including Hindi)."""
    
    def __init__(self, language_code: str = "en-US"):
        """
        Initialize Google Cloud STT.
        
        Language codes:
        - "en-US": English (US)
        - "hi-IN": Hindi
        - "en-IN": English (India) - better for Hinglish
        """
        self.client = speech.SpeechClient()
        self.language_code = language_code
        self.sample_rate = 16000
    
    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio using Google Cloud Speech-to-Text."""
        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language_code,
            model="latest_long",  # Best for longer utterances
        )
        
        response = self.client.recognize(config=config, audio=audio)
        
        # Combine all transcript results
        transcript = ""
        for result in response.results:
            if result.alternatives:
                transcript += result.alternatives[0].transcript + " "
        
        return transcript.strip()


def create_stt_engine(backend: str = "vosk", **kwargs) -> STTEngine:
    """
    Factory function to create STT engine.
    
    Args:
        backend: "vosk" or "google"
        **kwargs: Backend-specific arguments
            For vosk: model_path, sample_rate
            For google: language_code
    """
    if backend == "vosk":
        return VoskSTT(
            model_path=kwargs.get("model_path", "./vosk-model-en-us-0.22-lgraph"),
            sample_rate=kwargs.get("sample_rate", 16000),
        )
    elif backend == "google":
        return GoogleCloudSTT(
            language_code=kwargs.get("language_code", "en-IN"),  # Default to India English
        )
    else:
        raise ValueError(f"Unknown STT backend: {backend}")
