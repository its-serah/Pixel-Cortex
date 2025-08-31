from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from app.core.database import get_db
from app.core.security import verify_token, require_role
from app.models.models import PolicyDocument, PolicyChunk
from app.services.policy_retriever import PolicyRetriever
from app.services.policy_indexer import PolicyIndexer

router = APIRouter()

@router.get("/documents")
async def list_policy_documents(
    db: Session = Depends(get_db),
    current_user: dict = Depends(verify_token)
):
    """List all policy documents"""
    documents = db.query(PolicyDocument).all()
    
    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "title": doc.title,
            "file_type": doc.file_type,
            "created_at": doc.created_at,
            "updated_at": doc.updated_at,
            "chunk_count": len(doc.chunks)
        }
        for doc in documents
    ]

@router.get("/documents/{document_id}")
async def get_policy_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(verify_token)
):
    """Get a specific policy document with its chunks"""
    document = db.query(PolicyDocument).filter(PolicyDocument.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Policy document not found")
    
    chunks = db.query(PolicyChunk).filter(PolicyChunk.document_id == document_id).order_by(PolicyChunk.chunk_index).all()
    
    return {
        "id": document.id,
        "filename": document.filename,
        "title": document.title,
        "content": document.content,
        "file_type": document.file_type,
        "content_hash": document.content_hash,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
        "chunks": [
            {
                "id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "content_hash": chunk.content_hash
            }
            for chunk in chunks
        ]
    }

@router.post("/search")
async def search_policies(
    query: str,
    k: int = 5,
    db: Session = Depends(get_db),
    current_user: dict = Depends(verify_token)
):
    """Search policy documents using hybrid BM25 + TF-IDF"""
    retriever = PolicyRetriever()
    citations = retriever.retrieve_relevant_chunks(query, k=k, db=db)
    
    return {
        "query": query,
        "results": [
            {
                "document_id": citation.document_id,
                "document_title": citation.document_title,
                "chunk_id": citation.chunk_id,
                "chunk_content": citation.chunk_content,
                "relevance_score": citation.relevance_score
            }
            for citation in citations
        ],
        "total_results": len(citations)
    }

@router.post("/reindex")
async def reindex_policies(
    policies_dir: str = "./policies",
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_role("admin"))
):
    """Reindex all policy documents"""
    indexer = PolicyIndexer()
    
    try:
        await indexer.reindex_all_policies(policies_dir)
        
        # Get updated counts
        total_docs = db.query(PolicyDocument).count()
        total_chunks = db.query(PolicyChunk).count()
        
        return {
            "message": "Policies reindexed successfully",
            "documents_indexed": total_docs,
            "chunks_created": total_chunks
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")

@router.get("/stats")
async def get_policy_stats(
    db: Session = Depends(get_db),
    current_user: dict = Depends(verify_token)
):
    """Get policy indexing statistics"""
    total_docs = db.query(PolicyDocument).count()
    total_chunks = db.query(PolicyChunk).count()
    
    # Get documents by type
    doc_types = db.query(PolicyDocument.file_type, db.func.count(PolicyDocument.id)).group_by(PolicyDocument.file_type).all()
    
    return {
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "average_chunks_per_document": total_chunks / total_docs if total_docs > 0 else 0,
        "documents_by_type": {file_type: count for file_type, count in doc_types}
    }
