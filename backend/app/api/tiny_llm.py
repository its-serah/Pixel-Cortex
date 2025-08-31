"""
Tiny LLM API - Works with small models on free tier
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.tiny_llm_service import tiny_llm_service

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None
    max_length: Optional[int] = 150

class ChatResponse(BaseModel):
    response: str
    mode: str  # "api", "onnx", or "rules"
    concepts: List[str]

@router.post("/chat", response_model=ChatResponse)
async def chat_with_tiny_llm(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Chat with tiny LLM that works on free tier
    Falls back to rules if no LLM available
    """
    try:
        # Build prompt with context
        prompt = request.message
        if request.context:
            prompt = f"Context: {request.context}\n\nUser: {request.message}\n\nAssistant:"
        
        # Generate response
        response = tiny_llm_service.generate(prompt, request.max_length or 150)
        
        # Extract concepts
        concepts = tiny_llm_service.extract_concepts(request.message)
        
        return ChatResponse(
            response=response,
            mode=tiny_llm_service.mode,
            concepts=concepts
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_llm_status():
    """Check which LLM mode is active"""
    return {
        "mode": tiny_llm_service.mode,
        "available_modes": ["api", "onnx", "rules"],
        "description": {
            "api": "Using external LLM API (Together.ai/OpenAI)",
            "onnx": "Using local ONNX tiny model",
            "rules": "Using rule-based responses (no LLM)"
        }
    }
