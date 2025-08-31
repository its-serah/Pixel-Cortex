# Minimal stub of OpenAI Whisper API to allow imports in environments without the package.
# This is only for testing and non-ML environments.

from typing import Any, Dict, Tuple

class _StubWhisperModel:
    def transcribe(self, audio: Any, language: str = "en", fp16: bool = False, verbose: bool = False) -> Dict[str, Any]:
        # Return a minimal structure similar to whisper output
        return {"text": "", "segments": [{"no_speech_prob": 0.5}]}

    def detect_language(self, audio: Any) -> Tuple[None, Dict[str, float]]:
        # Pretend English is detected with high confidence
        return None, {"en": 0.9}


def load_model(name: str = "base", device: str = "cpu") -> _StubWhisperModel:
    return _StubWhisperModel()

