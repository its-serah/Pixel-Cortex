"""
RAG API
- /api/rag/index: reindex policies and rebuild KG-Lite
- /api/rag/search: BM25+TFIDF search returning simple results structure
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.policy_indexer import PolicyIndexer
from app.services.policy_retriever import PolicyRetriever
from app.services.kg_lite_service import kg_lite_service

router = APIRouter()

class IndexRequest(BaseModel):
    policies_dir: Optional[str] = "./policies"

@router.post("/index")
async def rag_index(req: IndexRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    try:
        idx = PolicyIndexer()
        # Force full reindex for determinism
        idx.reindex_all_policies(req.policies_dir)
        # Rebuild KG-lite from fresh chunks
        kg_info = kg_lite_service.rebuild_from_policies(db)
        return {
            "message": "RAG index + KG rebuilt",
            "policies_dir": req.policies_dir,
            "kg": kg_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5

@router.post("/search")
async def rag_search(req: SearchRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    try:
        retriever = PolicyRetriever()
        citations = retriever.retrieve_relevant_chunks(req.query, k=req.k, db=db)
        results = []
        for c in citations:
            results.append({
                "chunk_id": c.chunk_id,
                "document_title": c.document_title,
                "score": round(float(c.relevance_score), 3),
                "content": c.chunk_content
            })
        return {
            "summary": f"Top {len(results)} results for: '{req.query}'",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

