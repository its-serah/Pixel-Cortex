import os
import tempfile
from typing import Dict, Optional, Any
from datetime import datetime
import whisper
import torch
from faster_whisper import WhisperModel
from app.models.schemas import ExplanationObject, ReasoningStep, TelemetryData

class AudioService:
    """
    Service for processing audio input using Whisper for speech-to-text
    Optimized for fast, lightweight inference
    """
    
    def __init__(self):
        self.model = None
        self.model_size = "base"  # Fast and lightweight
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use faster-whisper for better performance
        try:
            self.model = WhisperModel(self.model_size, device=self.device, compute_type="float16" if self.device == "cuda" else "int8")
            print(f"Initialized Whisper {self.model_size} model on {self.device}")
        except Exception as e:
            print(f"Failed to load faster-whisper, falling back to standard whisper: {e}")
            self.model = whisper.load_model(self.model_size, device=self.device)
    
    def transcribe_audio(self, audio_data: bytes, filename: str = "audio.wav") -> Dict[str, Any]:
        """
        Convert audio bytes to text with explanation object
        
        Args:
            audio_data: Raw audio file bytes
            filename: Original filename for context
            
        Returns:
            Dict with transcribed text and explanation
        """
        start_time = datetime.now()
        
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name
            
            # Transcribe using faster-whisper or standard whisper
            if isinstance(self.model, WhisperModel):
                segments, info = self.model.transcribe(temp_audio_path, language="en")
                transcription = " ".join([segment.text for segment in segments])
                language_detected = info.language
                language_probability = info.language_probability
            else:
                result = self.model.transcribe(temp_audio_path, language="en")
                transcription = result["text"]
                language_detected = result.get("language", "en")
                language_probability = 1.0
            
            # Clean up temp file
            os.unlink(temp_audio_path)
            
            end_time = datetime.now()
            processing_time = int((end_time - start_time).total_seconds() * 1000)
            
            # Build reasoning trace for audio processing
            reasoning_trace = [
                ReasoningStep(
                    step=1,
                    action="audio_preprocessing",
                    rationale=f"Received audio file '{filename}' ({len(audio_data)} bytes)",
                    confidence=0.95,
                    policy_refs=[]
                ),
                ReasoningStep(
                    step=2,
                    action="speech_recognition",
                    rationale=f"Transcribed audio using Whisper {self.model_size} model on {self.device}",
                    confidence=float(language_probability),
                    policy_refs=[]
                ),
                ReasoningStep(
                    step=3,
                    action="language_detection",
                    rationale=f"Detected language: {language_detected} (confidence: {language_probability:.2f})",
                    confidence=float(language_probability),
                    policy_refs=[]
                )
            ]
            
            # Create explanation object for audit logging
            explanation = ExplanationObject(
                answer=f"Audio successfully transcribed: '{transcription[:100]}{'...' if len(transcription) > 100 else ''}'",
                decision=f"transcription_success=true, language={language_detected}, duration_ms={processing_time}",
                confidence=float(language_probability),
                reasoning_trace=reasoning_trace,
                policy_citations=[],
                missing_info=self._identify_audio_quality_issues(transcription, language_probability),
                alternatives_considered=[],
                counterfactuals=[],
                telemetry=TelemetryData(
                    latency_ms=processing_time,
                    retrieval_k=0,
                    triage_time_ms=processing_time,
                    planning_time_ms=0,
                    total_chunks_considered=0
                ),
                timestamp=datetime.now(),
                model_version=f"whisper-{self.model_size}"
            )
            
            return {
                "success": True,
                "transcription": transcription.strip(),
                "language": language_detected,
                "confidence": language_probability,
                "processing_time_ms": processing_time,
                "explanation": explanation
            }
            
        except Exception as e:
            # Handle transcription errors
            error_explanation = self._create_error_explanation(str(e), filename)
            
            return {
                "success": False,
                "error": str(e),
                "transcription": "",
                "explanation": error_explanation
            }
    
    def validate_audio_format(self, filename: str, audio_data: bytes) -> Dict[str, Any]:
        """Validate audio format and quality for transcription"""
        
        # Check file extension
        valid_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        file_ext = os.path.splitext(filename.lower())[1]
        
        if file_ext not in valid_extensions:
            return {
                "valid": False,
                "error": f"Unsupported audio format: {file_ext}. Supported: {valid_extensions}",
                "recommendations": ["Convert to WAV or MP3 format", "Use audio recording with supported format"]
            }
        
        # Check file size (limit to 25MB for performance)
        max_size_mb = 25
        size_mb = len(audio_data) / (1024 * 1024)
        
        if size_mb > max_size_mb:
            return {
                "valid": False,
                "error": f"Audio file too large: {size_mb:.1f}MB (max: {max_size_mb}MB)",
                "recommendations": ["Compress audio file", "Split into smaller segments", "Use lower quality recording"]
            }
        
        return {
            "valid": True,
            "size_mb": round(size_mb, 2),
            "format": file_ext,
            "recommendations": []
        }
    
    def _identify_audio_quality_issues(self, transcription: str, confidence: float) -> List[str]:
        """Identify potential audio quality issues based on transcription results"""
        issues = []
        
        if confidence < 0.7:
            issues.append("Low transcription confidence - audio quality may be poor")
        
        if len(transcription.strip()) < 10:
            issues.append("Very short transcription - check if audio contains speech")
        
        # Check for repeated characters (sign of poor audio)
        if any(char * 4 in transcription for char in "aeiou"):
            issues.append("Detected repeated characters - possible audio distortion")
        
        # Check for common transcription artifacts
        artifacts = ["[inaudible]", "[unclear]", "um", "uh", "hmm"]
        if any(artifact in transcription.lower() for artifact in artifacts):
            issues.append("Detected speech artifacts - consider clearer recording")
        
        return issues
    
    def _create_error_explanation(self, error_msg: str, filename: str) -> ExplanationObject:
        """Create explanation object for transcription errors"""
        
        reasoning_trace = [
            ReasoningStep(
                step=1,
                action="transcription_error",
                rationale=f"Failed to transcribe audio file '{filename}': {error_msg}",
                confidence=0.0,
                policy_refs=[]
            )
        ]
        
        return ExplanationObject(
            answer=f"Audio transcription failed: {error_msg}",
            decision="transcription_success=false",
            confidence=0.0,
            reasoning_trace=reasoning_trace,
            policy_citations=[],
            missing_info=["Valid audio file with clear speech"],
            alternatives_considered=[
                {"option": "Re-record audio", "pros": ["Better quality"], "cons": ["Requires user action"], "confidence": 0.9},
                {"option": "Type ticket manually", "pros": ["Immediate submission"], "cons": ["No voice convenience"], "confidence": 1.0}
            ],
            counterfactuals=[],
            telemetry=TelemetryData(
                latency_ms=0,
                retrieval_k=0,
                triage_time_ms=0,
                planning_time_ms=0,
                total_chunks_considered=0
            ),
            timestamp=datetime.now(),
            model_version=f"whisper-{self.model_size}"
        )
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats"""
        return [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
    
    def estimate_processing_time(self, audio_size_bytes: int) -> int:
        """Estimate processing time in milliseconds based on audio size"""
        # Rough estimate: ~1 second per MB on CPU, ~0.3 seconds per MB on GPU
        size_mb = audio_size_bytes / (1024 * 1024)
        
        if self.device == "cuda":
            return int(size_mb * 300)  # 300ms per MB on GPU
        else:
            return int(size_mb * 1000)  # 1000ms per MB on CPU
