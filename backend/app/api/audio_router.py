"""
Audio API Router

REST endpoints for audio processing, speech-to-text conversion,
and real-time audio streaming for the IT Support Agent.
"""

import json
import uuid
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.models import User
from app.services.audio_processing_service import audio_service
from app.services.local_llm_service import local_llm_service
from app.services.conversation_memory_service import conversation_memory_service
from app.services.prompt_engineering_service import prompt_engineering_service, PromptTemplate


router = APIRouter()


@router.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    language: str = Form("en"),
    use_vad: bool = Form(True),
    session_id: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload and transcribe audio file
    
    Args:
        file: Audio file (wav, mp3, m4a, flac, ogg, webm)
        language: Language code for transcription (default: en)
        use_vad: Whether to use voice activity detection
        session_id: Optional conversation session ID
        
    Returns:
        Transcription result with metadata
    """
    
    try:
        # Read audio file
        audio_data = await file.read()
        
        # Detect file format
        file_extension = file.filename.split('.')[-1].lower() if file.filename else 'wav'
        
        # Validate audio file
        validation = audio_service.validate_audio_file(audio_data, file_extension)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=validation["error"])
        
        # Auto-detect language if not specified
        if language == "auto":
            language = audio_service.detect_language(audio_data, file_extension)
        
        # Transcribe audio
        transcription_result = audio_service.transcribe_audio(
            audio_data=audio_data,
            audio_format=file_extension,
            language=language,
            use_vad=use_vad
        )
        
        if "error" in transcription_result:
            raise HTTPException(status_code=500, detail=transcription_result["error"])
        
        transcribed_text = transcription_result["text"]
        
        # Generate LLM response if text was successfully transcribed
        llm_response = None
        if transcribed_text.strip():
            # Get conversation context
            context = conversation_memory_service.get_contextual_memory(
                db, transcribed_text, current_user.id, session_id
            )
            
            # Generate intelligent response
            llm_result = local_llm_service.generate_response(
                transcribed_text,
                context={
                    "conversation_history": context["conversation_history"],
                    "relevant_policies": context.get("related_tickets", []),
                    "kg_concepts": []  # Will be populated by KG system
                }
            )
            llm_response = llm_result["response"]
            
            # Log the conversation
            conversation_memory_service.log_conversation(
                db=db,
                user_id=current_user.id,
                user_message=transcribed_text,
                assistant_response=llm_response,
                audio_metadata=transcription_result,
                llm_metadata=llm_result,
                session_id=session_id or str(uuid.uuid4())
            )
        
        return {
            "transcription": transcription_result,
            "llm_response": llm_response,
            "session_id": session_id or str(uuid.uuid4()),
            "user_id": current_user.id,
            "processing_stats": {
                "audio_duration": transcription_result.get("duration", 0),
                "processing_time": transcription_result.get("processing_time_ms", 0),
                "confidence": transcription_result.get("confidence", 0),
                "language_detected": transcription_result.get("language", language)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")


@router.post("/transcribe")
async def transcribe_audio_only(
    file: UploadFile = File(...),
    language: str = Form("en"),
    use_vad: bool = Form(True),
    detailed: bool = Form(False),
    current_user: User = Depends(get_current_user)
):
    """
    Transcribe audio file without generating LLM response
    
    Args:
        file: Audio file
        language: Language code (or 'auto' for detection)
        use_vad: Voice activity detection
        detailed: Return detailed transcription with timestamps
        
    Returns:
        Transcription result only
    """
    
    try:
        audio_data = await file.read()
        file_extension = file.filename.split('.')[-1].lower() if file.filename else 'wav'
        
        # Validate audio
        validation = audio_service.validate_audio_file(audio_data, file_extension)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=validation["error"])
        
        # Auto-detect language if requested
        if language == "auto":
            language = audio_service.detect_language(audio_data, file_extension)
        
        # Transcribe with or without detailed timestamps
        if detailed:
            result = audio_service.transcribe_with_timestamps(audio_data, file_extension)
        else:
            result = audio_service.transcribe_audio(
                audio_data, file_extension, language, use_vad
            )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "transcription": result,
            "user_id": current_user.id,
            "file_info": {
                "filename": file.filename,
                "size": len(audio_data),
                "format": file_extension,
                "duration": result.get("duration", 0)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.get("/stats")
async def get_audio_stats(current_user: User = Depends(get_current_user)):
    """Get audio processing statistics"""
    
    return {
        "audio_service": audio_service.get_processing_stats(),
        "supported_formats": audio_service.supported_formats,
        "user_id": current_user.id
    }


@router.post("/validate")
async def validate_audio_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Validate audio file before processing"""
    
    try:
        audio_data = await file.read()
        file_extension = file.filename.split('.')[-1].lower() if file.filename else 'wav'
        
        validation = audio_service.validate_audio_file(audio_data, file_extension)
        
        return {
            "validation": validation,
            "file_info": {
                "filename": file.filename,
                "size": len(audio_data),
                "format": file_extension
            },
            "user_id": current_user.id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


# WebSocket manager for real-time audio streaming
class AudioWebSocketManager:
    """Manage WebSocket connections for real-time audio streaming"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_buffers: Dict[str, List[bytes]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_buffers[session_id] = []
    
    def disconnect(self, session_id: str):
        """Remove WebSocket connection"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.session_buffers:
            del self.session_buffers[session_id]
    
    async def process_audio_chunk(self, session_id: str, audio_chunk: bytes) -> Optional[str]:
        """Process incoming audio chunk and return transcription when ready"""
        
        if session_id not in self.session_buffers:
            return None
        
        # Add chunk to buffer
        self.session_buffers[session_id].append(audio_chunk)
        
        # Check if we have enough audio for transcription (e.g., 2 seconds worth)
        total_size = sum(len(chunk) for chunk in self.session_buffers[session_id])
        min_chunk_size = 32000  # Approximate 2 seconds of 16kHz audio
        
        if total_size >= min_chunk_size:
            # Combine chunks and transcribe
            combined_audio = b"".join(self.session_buffers[session_id])
            self.session_buffers[session_id] = []  # Clear buffer
            
            try:
                result = audio_service.transcribe_audio(combined_audio, "wav", "en", True)
                return result.get("text", "")
            except Exception as e:
                logger.error(f"Streaming transcription error: {e}")
                return None
        
        return None


audio_websocket_manager = AudioWebSocketManager()


@router.websocket("/stream/{session_id}")
async def websocket_audio_stream(
    websocket: WebSocket,
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for real-time audio streaming and transcription
    
    Args:
        websocket: WebSocket connection
        session_id: Unique session identifier
    """
    
    await audio_websocket_manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Process audio chunk
            transcription = await audio_websocket_manager.process_audio_chunk(session_id, data)
            
            if transcription:
                # Send transcription back to client
                await websocket.send_json({
                    "type": "transcription",
                    "text": transcription,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Optionally generate LLM response for immediate feedback
                if len(transcription.strip()) > 10:  # Only for substantial input
                    try:
                        llm_result = local_llm_service.generate_response(
                            transcription,
                            max_tokens=256,
                            temperature=0.7
                        )
                        
                        await websocket.send_json({
                            "type": "llm_response",
                            "response": llm_result["response"],
                            "session_id": session_id,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "error": f"LLM processing failed: {str(e)}",
                            "session_id": session_id
                        })
    
    except WebSocketDisconnect:
        audio_websocket_manager.disconnect(session_id)
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "error": str(e),
            "session_id": session_id
        })
        audio_websocket_manager.disconnect(session_id)


@router.get("/sessions/{session_id}/summary")
async def get_audio_session_summary(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get summary of an audio conversation session"""
    
    try:
        summary = conversation_memory_service.summarize_conversation_session(
            db, session_id, current_user.id
        )
        
        return {
            "session_summary": summary,
            "session_id": session_id,
            "user_id": current_user.id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")


@router.delete("/sessions/{session_id}")
async def delete_audio_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete an audio conversation session"""
    
    try:
        # Delete conversation logs for this session
        deleted_count = db.query(ConversationLog).filter(
            ConversationLog.metadata.contains({"session_id": session_id}),
            ConversationLog.user_id == current_user.id
        ).delete()
        
        db.commit()
        
        return {
            "deleted": True,
            "session_id": session_id,
            "conversations_deleted": deleted_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


@router.get("/history/search")
async def search_audio_history(
    query: str,
    days_back: int = 30,
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Search through audio conversation history"""
    
    try:
        results = conversation_memory_service.search_conversation_history(
            db, query, current_user.id, days_back, limit
        )
        
        # Filter for audio conversations only
        audio_results = [
            result for result in results 
            if result.get("input_method") == "audio"
        ]
        
        return {
            "search_results": audio_results,
            "query": query,
            "total_found": len(audio_results),
            "user_id": current_user.id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/analyze")
async def analyze_audio_content(
    file: UploadFile = File(...),
    analysis_type: str = Form("full"),  # full, concepts, intent
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze audio content for IT concepts, intent, and context
    
    Args:
        file: Audio file to analyze
        analysis_type: Type of analysis (full, concepts, intent)
        
    Returns:
        Comprehensive analysis of audio content
    """
    
    try:
        # Transcribe audio first
        audio_data = await file.read()
        file_extension = file.filename.split('.')[-1].lower() if file.filename else 'wav'
        
        transcription_result = audio_service.transcribe_audio(
            audio_data, file_extension, "en", True
        )
        
        if "error" in transcription_result:
            raise HTTPException(status_code=500, detail=transcription_result["error"])
        
        transcribed_text = transcription_result["text"]
        analysis_results = {"transcription": transcription_result}
        
        if not transcribed_text.strip():
            return {
                "analysis": analysis_results,
                "message": "No speech detected in audio"
            }
        
        # Perform requested analysis
        if analysis_type in ["full", "concepts"]:
            # Extract IT concepts
            concept_extraction = prompt_engineering_service.generate_safe_response(
                PromptTemplate.CONCEPT_EXTRACTION,
                {
                    "text": transcribed_text,
                    "known_concepts": "VPN, MFA, Remote Access, Firewall, Security Incident"
                }
            )
            analysis_results["concepts"] = concept_extraction
        
        if analysis_type in ["full", "intent"]:
            # Analyze intent
            intent_analysis = prompt_engineering_service.analyze_user_intent(
                transcribed_text, []  # No conversation context for file analysis
            )
            analysis_results["intent"] = intent_analysis
        
        if analysis_type == "full":
            # Get related context from conversation history
            context = conversation_memory_service.get_contextual_memory(
                db, transcribed_text, current_user.id
            )
            analysis_results["context"] = {
                "related_conversations": len(context["related_conversations"]),
                "related_tickets": len(context["related_tickets"]),
                "context_summary": context["context_summary"]
            }
        
        return {
            "analysis": analysis_results,
            "user_id": current_user.id,
            "file_info": {
                "filename": file.filename,
                "duration": transcription_result.get("duration", 0),
                "confidence": transcription_result.get("confidence", 0)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/models/status")
async def get_model_status(current_user: User = Depends(get_current_user)):
    """Get status of audio processing models"""
    
    return {
        "audio_models": audio_service.get_processing_stats(),
        "llm_models": local_llm_service.get_performance_stats(),
        "system_info": {
            "model_loaded": audio_service.whisper_model is not None,
            "supported_formats": audio_service.supported_formats
        }
    }


@router.post("/models/load")
async def load_audio_models(current_user: User = Depends(get_current_user)):
    """Preload audio processing models"""
    
    # Verify user has admin privileges
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Load models
        audio_service.load_models()
        local_llm_service.load_model()
        
        return {
            "models_loaded": True,
            "audio_model": audio_service.whisper_model is not None,
            "llm_model": local_llm_service.model is not None,
            "message": "Models loaded successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")


@router.post("/models/unload")
async def unload_audio_models(current_user: User = Depends(get_current_user)):
    """Unload audio processing models to free memory"""
    
    # Verify user has admin privileges
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Unload models
        local_llm_service.unload_model()
        # Note: Whisper model unloading would need to be implemented in audio_service
        
        return {
            "models_unloaded": True,
            "message": "Models unloaded to free memory"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model unloading failed: {str(e)}")


@router.get("/languages/supported")
async def get_supported_languages():
    """Get list of supported languages for transcription"""
    
    # Languages supported by Whisper
    supported_languages = {
        "en": "English",
        "es": "Spanish", 
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "ja": "Japanese",
        "ko": "Korean",
        "zh": "Chinese",
        "ar": "Arabic",
        "hi": "Hindi",
        "auto": "Auto-detect"
    }
    
    return {
        "supported_languages": supported_languages,
        "default_language": "en",
        "auto_detection_available": True
    }


@router.post("/conversation/voice")
async def voice_conversation(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    context_aware: bool = Form(True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Complete voice conversation flow with transcription, context, and intelligent response
    
    Args:
        file: Audio file with user's voice input
        session_id: Conversation session ID
        context_aware: Whether to use conversation context
        
    Returns:
        Complete conversation response with transcription and AI reply
    """
    
    try:
        # Process audio
        audio_data = await file.read()
        file_extension = file.filename.split('.')[-1].lower() if file.filename else 'wav'
        
        # Transcribe
        transcription_result = audio_service.transcribe_audio(audio_data, file_extension)
        
        if "error" in transcription_result or not transcription_result["text"].strip():
            return {
                "transcription": transcription_result,
                "llm_response": "I couldn't understand the audio. Please try again.",
                "session_id": session_id or str(uuid.uuid4())
            }
        
        transcribed_text = transcription_result["text"]
        
        # Get conversation context if enabled
        context = {}
        if context_aware:
            context = conversation_memory_service.get_contextual_memory(
                db, transcribed_text, current_user.id, session_id
            )
        
        # Generate intelligent response using LLM
        llm_result = local_llm_service.generate_response(
            transcribed_text,
            context={
                "conversation_history": context.get("conversation_history", []),
                "relevant_policies": context.get("related_tickets", []),
                "context_summary": context.get("context_summary", "")
            }
        )
        
        # Log conversation
        final_session_id = session_id or str(uuid.uuid4())
        conversation_memory_service.log_conversation(
            db=db,
            user_id=current_user.id,
            user_message=transcribed_text,
            assistant_response=llm_result["response"],
            audio_metadata=transcription_result,
            llm_metadata=llm_result,
            session_id=final_session_id
        )
        
        return {
            "transcription": transcription_result,
            "llm_response": llm_result["response"],
            "session_id": final_session_id,
            "context_used": context_aware,
            "processing_stats": {
                "audio_processing_ms": transcription_result.get("processing_time_ms", 0),
                "llm_processing_ms": llm_result.get("inference_time_ms", 0),
                "total_processing_ms": (
                    transcription_result.get("processing_time_ms", 0) + 
                    llm_result.get("inference_time_ms", 0)
                ),
                "audio_confidence": transcription_result.get("confidence", 0),
                "from_cache": llm_result.get("from_cache", False)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice conversation failed: {str(e)}")


# Add router tags and metadata
router.tags = ["Audio Processing"]
