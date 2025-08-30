"""
LLM Chat API (CPU deterministic KG-RAG)
POST /api/llm/chat
Body: { message: str, augment?: bool, k?: int, include_explanation?: bool }
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.rag_reasoner import rag_reasoner
from app.services.llm_service import LLMService
import os

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    augment: Optional[bool] = True
    k: Optional[int] = 5
    include_explanation: Optional[bool] = True
    engine: Optional[str] = "deterministic"  # "deterministic" | "ollama"
    conversation_history: Optional[list] = None

@router.post("/chat")
async def chat(req: ChatRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    try:
        engine = (req.engine or "deterministic").lower()
        if engine == "ollama":
            # True LLM CoT mode via Ollama
            model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
            svc = LLMService(model_name=model_name)
            try:
                response_text, explanation = svc.generate_response(req.message, db=db, include_cot=True)
                return {"response": response_text, "explanation": explanation.dict()}
            except Exception as e:
                # Provide helpful error if Ollama not running
                raise HTTPException(status_code=500, detail=f"Ollama mode failed: {e}. Ensure ollama serve is running and model '{model_name}' is available (ollama pull {model_name}).")
        else:
            # Deterministic CPU KG-RAG mode
            result = rag_reasoner.chat(db, req.message, augment=req.augment, k=req.k, include_explanation=req.include_explanation)
            return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

