"""
Audio Processing Service

Handles speech-to-text conversion using OpenAI Whisper and audio preprocessing
for the IT Support Agent with real-time audio capabilities.
"""

import io
import os
import json
import time
import logging
import tempfile
import threading
from typing import Dict, Any, Optional, BinaryIO, List, Tuple
from pathlib import Path

import torch
# Optional heavy deps with graceful fallbacks
try:
    import whisper  # type: ignore
except Exception:
    import importlib
    whisper = importlib.import_module('whisper')  # Use local stub if available
try:
    import librosa  # type: ignore
except Exception:
    librosa = None
try:
    import soundfile as sf  # type: ignore
except Exception:
    sf = None
try:
    import webrtcvad  # type: ignore
except Exception:
    webrtcvad = None
try:
    import speech_recognition as sr  # type: ignore
except Exception:
    sr = None
try:
    from pydub import AudioSegment  # type: ignore
    from pydub.silence import split_on_silence  # type: ignore
except Exception:
    AudioSegment = None
    split_on_silence = None
import numpy as np
import wave

from app.core.config import settings


logger = logging.getLogger(__name__)


class AudioProcessingService:
    """Optimized audio processing service for IT support"""
    
    def __init__(self):
        self.whisper_model = None
        self.vad = None
        self.recognizer = None
        self.model_lock = threading.Lock()
        self.supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm']
        
        # Audio processing settings
        self.target_sample_rate = 16000  # Whisper optimal sample rate
        self.chunk_duration = 30  # seconds
        self.min_audio_length = 0.5  # minimum seconds
        
        # Performance stats
        self.processing_stats = {
            "total_processed": 0,
            "avg_processing_time": 0.0,
            "total_audio_duration": 0.0,
            "errors": 0
        }
        
        logger.info("Initializing Audio Processing Service")
    
    def load_models(self) -> None:
        """Load audio processing models"""
        if self.whisper_model is not None:
            return
        
        with self.model_lock:
            if self.whisper_model is not None:
                return
            
            logger.info("Loading Whisper model...")
            start_time = time.time()
            
            # Load Whisper model (base model for good speed/accuracy balance)
            self.whisper_model = whisper.load_model("base", device="cpu")  # Use CPU for better compatibility
            
            # Initialize VAD for voice activity detection
            self.vad = webrtcvad.Vad(2) if webrtcvad else None  # Aggressiveness level 2 (balanced)
            
            # Initialize speech recognition for fallback
            self.recognizer = sr.Recognizer() if sr else None
            
            load_time = time.time() - start_time
            logger.info(f"Audio models loaded in {load_time:.2f}s")
    
    def preprocess_audio(self, audio_data: bytes, original_format: str = "wav") -> np.ndarray:
        """Preprocess audio data for optimal transcription"""
        
        try:
            if AudioSegment is not None:
                # Convert bytes to AudioSegment
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format=original_format.replace('.', ''))
                
                # Convert to mono if stereo
                if audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)
                
                # Normalize audio level
                audio_segment = audio_segment.normalize()
                
                # Set target sample rate
                audio_segment = audio_segment.set_frame_rate(self.target_sample_rate)
                
                # Convert to numpy array
                audio_array = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
                audio_array = audio_array / np.iinfo(audio_segment.array_type).max
                
                return audio_array
            else:
                # Fallback: handle WAV using stdlib wave
                fmt = original_format.replace('.', '').lower()
                if fmt != 'wav':
                    raise ValueError("Only WAV format supported without pydub installed")
                with wave.open(io.BytesIO(audio_data), 'rb') as wf:
                    n_channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    frame_rate = wf.getframerate()
                    n_frames = wf.getnframes()
                    raw = wf.readframes(n_frames)
                # Convert to mono if needed
                if n_channels > 1:
                    dtype = np.int16 if sample_width == 2 else np.int8
                    audio = np.frombuffer(raw, dtype=dtype)
                    audio = audio.reshape(-1, n_channels).mean(axis=1)
                else:
                    dtype = np.int16 if sample_width == 2 else np.int8
                    audio = np.frombuffer(raw, dtype=dtype)
                # Normalize to float32 -1..1
                max_val = float(np.iinfo(dtype).max)
                audio_array = (audio.astype(np.float32) / max_val)
                
                # Resample if needed
                if frame_rate != self.target_sample_rate and librosa is not None:
                    audio_array = librosa.resample(audio_array, orig_sr=frame_rate, target_sr=self.target_sample_rate)
                elif frame_rate != self.target_sample_rate:
                    # Simple naive resample: skip
                    pass
                return audio_array
            
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            raise ValueError(f"Failed to preprocess audio: {e}")
    
    def detect_speech_segments(self, audio_data: np.ndarray) -> List[Tuple[float, float]]:
        """Detect speech segments using VAD"""
        
        try:
            if self.vad is None:
                raise RuntimeError("VAD not available")
            # Convert to 16-bit PCM for VAD
            audio_pcm = (audio_data * 32767).astype(np.int16).tobytes()
            
            # VAD operates on 10, 20, or 30ms frames
            frame_duration = 20  # ms
            frame_size = int(self.target_sample_rate * frame_duration / 1000)
            
            speech_segments = []
            current_segment_start = None
            
            for i in range(0, len(audio_pcm), frame_size * 2):  # 2 bytes per sample
                frame = audio_pcm[i:i + frame_size * 2]
                if len(frame) < frame_size * 2:
                    break
                
                # Check if frame contains speech
                is_speech = self.vad.is_speech(frame, self.target_sample_rate)
                timestamp = i / (self.target_sample_rate * 2)  # Convert to seconds
                
                if is_speech and current_segment_start is None:
                    current_segment_start = timestamp
                elif not is_speech and current_segment_start is not None:
                    speech_segments.append((current_segment_start, timestamp))
                    current_segment_start = None
            
            # Close final segment if needed
            if current_segment_start is not None:
                speech_segments.append((current_segment_start, len(audio_data) / self.target_sample_rate))
            
            return speech_segments
            
        except Exception as e:
            logger.warning(f"VAD failed or unavailable, using full audio: {e}")
            # Return full audio as single segment
            return [(0.0, len(audio_data) / self.target_sample_rate)]
    
    def transcribe_audio(
        self,
        audio_data: bytes,
        audio_format: str = "wav",
        language: str = "en",
        use_vad: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text using Whisper
        
        Args:
            audio_data: Raw audio bytes
            audio_format: Audio format (wav, mp3, etc.)
            language: Language code for transcription
            use_vad: Whether to use voice activity detection
            
        Returns:
            Transcription result with metadata
        """
        self.load_models()
        
        start_time = time.time()
        
        try:
            # Preprocess audio
            audio_array = self.preprocess_audio(audio_data, audio_format)
            audio_duration = len(audio_array) / self.target_sample_rate
            
            # Check minimum duration
            if audio_duration < self.min_audio_length:
                return {
                    "text": "",
                    "confidence": 0.0,
                    "error": "Audio too short",
                    "duration": audio_duration,
                    "processing_time_ms": 0
                }
            
            # Detect speech segments if enabled
            if use_vad:
                speech_segments = self.detect_speech_segments(audio_array)
            else:
                speech_segments = [(0.0, audio_duration)]
            
            # Transcribe each speech segment
            full_transcription = []
            total_confidence = 0.0
            
            for start_time_seg, end_time_seg in speech_segments:
                if end_time_seg - start_time_seg < self.min_audio_length:
                    continue
                
                # Extract segment
                start_sample = int(start_time_seg * self.target_sample_rate)
                end_sample = int(end_time_seg * self.target_sample_rate)
                segment_audio = audio_array[start_sample:end_sample]
                
                # Transcribe segment with Whisper
                try:
                    result = self.whisper_model.transcribe(
                        segment_audio,
                        language=language,
                        fp16=False,  # Use FP32 for CPU compatibility
                        verbose=False
                    )
                    
                    segment_text = result["text"].strip()
                    if segment_text:
                        full_transcription.append({
                            "text": segment_text,
                            "start": start_time_seg,
                            "end": end_time_seg,
                            "confidence": self._estimate_confidence(result)
                        })
                        total_confidence += self._estimate_confidence(result)
                
                except Exception as e:
                    logger.warning(f"Segment transcription failed: {e}")
                    continue
            
            # Combine transcriptions
            combined_text = " ".join([seg["text"] for seg in full_transcription])
            avg_confidence = total_confidence / len(full_transcription) if full_transcription else 0.0
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update stats
            self.processing_stats["total_processed"] += 1
            self.processing_stats["total_audio_duration"] += audio_duration
            self.processing_stats["avg_processing_time"] = (
                (self.processing_stats["avg_processing_time"] * (self.processing_stats["total_processed"] - 1) + processing_time)
                / self.processing_stats["total_processed"]
            )
            
            return {
                "text": combined_text,
                "confidence": avg_confidence,
                "duration": audio_duration,
                "processing_time_ms": processing_time,
                "segments": full_transcription,
                "speech_segments_count": len(speech_segments),
                "language": language
            }
            
        except Exception as e:
            self.processing_stats["errors"] += 1
            logger.error(f"Audio transcription error: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "error": str(e),
                "duration": 0.0,
                "processing_time_ms": (time.time() - start_time) * 1000
            }
    
    def _estimate_confidence(self, whisper_result: Dict[str, Any]) -> float:
        """Estimate confidence from Whisper result"""
        # Whisper doesn't provide direct confidence scores
        # Estimate based on segment properties
        
        if "segments" in whisper_result:
            # Average no_speech_prob across segments (lower is better)
            avg_no_speech = np.mean([seg.get("no_speech_prob", 0.5) for seg in whisper_result["segments"]])
            confidence = 1.0 - avg_no_speech
        else:
            # Default moderate confidence
            confidence = 0.7
        
        return max(0.0, min(1.0, confidence))
    
    def transcribe_audio_file(self, file_path: str, language: str = "en") -> Dict[str, Any]:
        """Transcribe audio file directly"""
        
        if not os.path.exists(file_path):
            return {"error": "File not found", "text": "", "confidence": 0.0}
        
        try:
            # Read file as bytes
            with open(file_path, "rb") as f:
                audio_data = f.read()
            
            # Detect format from extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                return {"error": f"Unsupported format: {file_ext}", "text": "", "confidence": 0.0}
            
            return self.transcribe_audio(audio_data, file_ext, language)
            
        except Exception as e:
            logger.error(f"File transcription error: {e}")
            return {"error": str(e), "text": "", "confidence": 0.0}
    
    def detect_language(self, audio_data: bytes, audio_format: str = "wav") -> str:
        """Detect language in audio using Whisper"""
        
        self.load_models()
        
        try:
            # Preprocess audio
            audio_array = self.preprocess_audio(audio_data, audio_format)
            
            # Use first 30 seconds for language detection
            if len(audio_array) > 30 * self.target_sample_rate:
                audio_array = audio_array[:30 * self.target_sample_rate]
            
            # Detect language
            _, probs = self.whisper_model.detect_language(audio_array)
            detected_language = max(probs, key=probs.get)
            confidence = probs[detected_language]
            
            logger.info(f"Detected language: {detected_language} (confidence: {confidence:.2f})")
            
            return detected_language if confidence > 0.7 else "en"  # Default to English
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "en"  # Default to English
    
    def validate_audio_file(self, audio_data: bytes, audio_format: str) -> Dict[str, Any]:
        """Validate audio file before processing"""
        
        try:
            # Check format
            if f".{audio_format.replace('.', '')}" not in self.supported_formats:
                return {
                    "valid": False,
                    "error": f"Unsupported format: {audio_format}",
                    "supported_formats": self.supported_formats
                }
            
            # Check file size (limit to 100MB)
            max_size = 100 * 1024 * 1024
            if len(audio_data) > max_size:
                return {
                    "valid": False,
                    "error": f"File too large: {len(audio_data)} bytes (max: {max_size})",
                    "file_size": len(audio_data)
                }
            
            # Try to load audio to check validity
            if AudioSegment is not None:
                audio_segment = AudioSegment.from_file(
                    io.BytesIO(audio_data), 
                    format=audio_format.replace('.', '')
                )
                
                duration = len(audio_segment) / 1000.0  # Convert to seconds
                
                # Check duration limits
                max_duration = 300  # 5 minutes
                if duration > max_duration:
                    return {
                        "valid": False,
                        "error": f"Audio too long: {duration:.1f}s (max: {max_duration}s)",
                        "duration": duration
                    }
                
                if duration < self.min_audio_length:
                    return {
                        "valid": False,
                        "error": f"Audio too short: {duration:.1f}s (min: {self.min_audio_length}s)",
                        "duration": duration
                    }
                
                return {
                    "valid": True,
                    "duration": duration,
                    "sample_rate": audio_segment.frame_rate,
                    "channels": audio_segment.channels,
                    "file_size": len(audio_data)
                }
            else:
                # Fallback for WAV using stdlib
                fmt = audio_format.replace('.', '').lower()
                if fmt != 'wav':
                    return {
                        "valid": False,
                        "error": f"Unsupported format without pydub: {audio_format}",
                        "file_size": len(audio_data)
                    }
                with wave.open(io.BytesIO(audio_data), 'rb') as wf:
                    n_channels = wf.getnchannels()
                    framerate = wf.getframerate()
                    n_frames = wf.getnframes()
                duration = n_frames / float(framerate)
                if duration < self.min_audio_length:
                    return {
                        "valid": False,
                        "error": f"Audio too short: {duration:.1f}s (min: {self.min_audio_length}s)",
                        "duration": duration
                    }
                if duration > 300:
                    return {
                        "valid": False,
                        "error": f"Audio too long: {duration:.1f}s (max: 300s)",
                        "duration": duration
                    }
                return {
                    "valid": True,
                    "duration": duration,
                    "sample_rate": framerate,
                    "channels": n_channels,
                    "file_size": len(audio_data)
                }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Invalid audio file: {e}",
                "file_size": len(audio_data)
            }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get audio processing statistics"""
        return {
            "model_loaded": self.whisper_model is not None,
            "supported_formats": self.supported_formats,
            "processing_stats": self.processing_stats,
            "target_sample_rate": self.target_sample_rate,
            "chunk_duration": self.chunk_duration
        }


# Global instance
audio_service = AudioProcessingService()
