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

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    augment: Optional[bool] = True
    k: Optional[int] = 5
    include_explanation: Optional[bool] = True

@router.post("/chat")
async def chat(req: ChatRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    try:
        result = rag_reasoner.chat(db, req.message, augment=req.augment, k=req.k, include_explanation=req.include_explanation)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

