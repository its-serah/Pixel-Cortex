"""
Audio CPU API (Vosk)

Simple endpoints to validate and transcribe audio on CPU using Vosk.
Supported formats (via soundfile/libsndfile): WAV, FLAC, OGG. MP3/M4A may not be available.
"""

from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.audio_vosk_service import vosk_audio_service

router = APIRouter()


@router.get("/test")
async def test_audio_cpu() -> Dict[str, Any]:
    try:
        # Force model load (and download if needed)
        vosk_audio_service._ensure_model()
        return {"status": "ok", "engine": "vosk", "model_path": vosk_audio_service.model_path}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        result = vosk_audio_service.transcribe(audio_bytes)
        return {
            "transcription": result,
            "file_info": {
                "filename": file.filename,
                "size": len(audio_bytes)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

