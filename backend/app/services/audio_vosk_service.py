"""
Vosk-based CPU audio transcription service

- Loads a small Vosk model (path from env VOSK_MODEL_PATH or auto-download from VOSK_MODEL_URL)
- Accepts audio bytes in WAV/FLAC/OGG if supported by soundfile
- Resamples to 16kHz mono with NumPy (no ffmpeg/librosa dependency)
- Produces plain text transcript and basic metrics
"""

import io
import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import soundfile as sf

try:
    from vosk import Model, KaldiRecognizer
except Exception as e:
    Model = None
    KaldiRecognizer = None


DEFAULT_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
DEFAULT_MODEL_DIR = "/opt/models/vosk-model-small-en-us-0.15"
TARGET_SR = 16000


class VoskAudioService:
    def __init__(self):
        self.model = None
        self.model_path = os.getenv("VOSK_MODEL_PATH", DEFAULT_MODEL_DIR)
        self.model_url = os.getenv("VOSK_MODEL_URL", DEFAULT_MODEL_URL)

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        if Model is None:
            raise RuntimeError("Vosk not installed. Please install vosk package.")

        model_dir = Path(self.model_path)
        if not model_dir.exists():
            model_dir.parent.mkdir(parents=True, exist_ok=True)
            # Download and extract
            tmp_zip = model_dir.parent / "vosk_model.zip"
            try:
                urllib.request.urlretrieve(self.model_url, tmp_zip)
                with zipfile.ZipFile(tmp_zip, 'r') as zf:
                    zf.extractall(model_dir.parent)
                # If extracted folder name differs, find it
                extracted = None
                for p in model_dir.parent.iterdir():
                    if p.is_dir() and p.name.startswith("vosk-model-small-en-us"):
                        extracted = p
                        break
                if extracted and extracted != model_dir:
                    extracted.rename(model_dir)
            finally:
                if tmp_zip.exists():
                    tmp_zip.unlink(missing_ok=True)

        # Load model
        self.model = Model(self.model_path)

    def _to_mono_16k(self, y: np.ndarray, sr: int) -> np.ndarray:
        # Ensure mono
        if y.ndim == 2:
            y = y.mean(axis=1)
        # Normalize to [-1, 1]
        if y.dtype != np.float32 and y.dtype != np.float64:
            # Assume int PCM, scale
            max_val = np.iinfo(y.dtype).max
            y = y.astype(np.float32) / max_val
        y = y.astype(np.float32)
        if sr == TARGET_SR:
            return y
        # Simple resample using linear interpolation
        duration = len(y) / float(sr)
        t_old = np.linspace(0.0, duration, num=len(y), endpoint=False)
        t_new = np.linspace(0.0, duration, num=int(duration * TARGET_SR), endpoint=False)
        y_new = np.interp(t_new, t_old, y).astype(np.float32)
        return y_new

    def _read_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        # Try soundfile to read a variety of formats (wav, flac, ogg)
        with io.BytesIO(audio_bytes) as bio:
            data, sr = sf.read(bio, dtype='float32', always_2d=False)
        if data.ndim == 2:
            # Convert stereo to mono
            data = data.mean(axis=1)
        return data, sr

    def transcribe(self, audio_bytes: bytes) -> Dict[str, Any]:
        self._ensure_model()

        # Decode audio
        data, sr = self._read_audio(audio_bytes)
        # Resample to 16k mono float32
        y = self._to_mono_16k(data, sr)
        # Convert to 16-bit PCM bytes for Vosk
        pcm16 = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

        rec = KaldiRecognizer(self.model, TARGET_SR)
        rec.SetWords(True)

        # Feed in chunks
        chunk_size = 32000  # ~1s at 16k mono, 16-bit -> 2 bytes/sample -> 32000 bytes ~= 1s
        for i in range(0, len(pcm16), chunk_size):
            rec.AcceptWaveform(pcm16[i:i + chunk_size])

        import json
        final = rec.FinalResult()
        try:
            result = json.loads(final)
        except Exception:
            result = {"text": final}

        text = (result.get("text") or "").strip()
        return {
            "text": text,
            "confidence": None,  # Vosk does not provide a single confidence
            "duration": len(y) / TARGET_SR,
            "processing_time_ms": None,
            "segments": result.get("result", []),
            "language": "en"
        }


vosk_audio_service = VoskAudioService()


# Singleton instance
vosk_service = VoskAudioService()
