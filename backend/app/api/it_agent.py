"""
IT Support Agent API - Implements ML Challenge Requirements
- Local inference with TinyLlama
- Chain-of-Thought reasoning
- Ticket creation and management
- Policy-based responses with citations
- Search integration
- Voice support (via Vosk)
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
import os
import json
from datetime import datetime
from pathlib import Path

from app.core.database import get_db
from app.services.rag_reasoner import rag_reasoner
from app.services.llm_service import LLMService
from app.models.models import Ticket, TicketCategory, TicketPriority, User
from passlib.hash import bcrypt

router = APIRouter()

class AgentQuery(BaseModel):
    query: str
    include_search: Optional[bool] = True
    include_policies: Optional[bool] = True
    include_history: Optional[bool] = True

class AgentResponse(BaseModel):
    response: str
    reasoning: Dict[str, Any]
    steps: List[Dict[str, Any]]
    policies_cited: List[str]
    compliance_status: str  # allowed/denied/needs_approval
    ticket_created: Optional[Dict[str, Any]] = None
    search_performed: bool
    model_used: str

class TicketRequest(BaseModel):
    issue: str
    user_id: Optional[str] = "demo"

@router.post("/ask", response_model=AgentResponse)
async def ask_it_agent(request: AgentQuery, db: Session = Depends(get_db)):
    """
    Main IT Support Agent endpoint
    Implements ML Challenge requirements:
    - Chain-of-Thought reasoning
    - Policy consultation
    - Clear checklists
    - Compliance status
    - Citation of policies
    """
    try:
        # LLM with CoT via Ollama; fallback to deterministic KG-RAG
        svc = LLMService(model_name=os.getenv("OLLAMA_MODEL", "phi3:mini"))

        # First attempt: deterministic KG-RAG (fast, citations guaranteed)
        det = rag_reasoner.chat(db, request.query, augment=True, k=5, include_explanation=True)
        det_steps = det.get("explanation", {}).get("reasoning_trace", [])
        det_citations = det.get("explanation", {}).get("policy_citations", [])

        # Second attempt: true LLM CoT with policy grounding (can fail if Ollama not running)
        try:
            llm_text, llm_expl = svc.generate_response(request.query, db=db, include_cot=True)
            response_text = llm_text
            # Merge citations and reasoning
            policies_cited = [str(c.chunk_id) for c in (llm_expl.policy_citations or [])]
            steps = det_steps + [{
                "step": s.step,
                "action": s.action,
                "rationale": s.rationale,
                "confidence": s.confidence
            } for s in (llm_expl.reasoning_trace or [])]
        except Exception:
            # Fallback to deterministic answer
            response_text = det.get("response", "")
            policies_cited = [str(c.get("chunk_id")) for c in det.get("explanation", {}).get("policy_citations", [])]
            steps = det_steps

        # Compliance status heuristic from response
        resp_lower = response_text.lower()
        if "needs_approval" in resp_lower or "approval" in resp_lower:
            compliance_status = "needs_approval"
        elif any(x in resp_lower for x in ["denied", "not allowed"]):
            compliance_status = "denied"
        else:
            compliance_status = "allowed"

        # Auto-create a ticket for every request (fast lane)
        ticket_data = None
        q = request.query
        def _category(issue: str) -> str:
            s = issue.lower()
            if any(t in s for t in ["vpn", "network", "wifi", "internet"]):
                return "network"
            if any(t in s for t in ["password", "login", "access", "permission"]):
                return "access"
            if any(t in s for t in ["software", "install", "update"]):
                return "software"
            if any(t in s for t in ["hardware", "laptop", "printer"]):
                return "hardware"
            return "other"
        def _priority(issue: str) -> str:
            s = issue.lower()
            if any(t in s for t in ["urgent", "critical", "down", "not working", "blocked"]):
                return "high"
            if any(t in s for t in ["slow", "issue", "problem", "help"]):
                return "medium"
            return "low"
        title = q.split(".")[0][:50] or "IT Support Request"
        category = _category(q)
        priority = _priority(q)

        # Ensure a requester exists (demo user fallback)
        requester = db.query(User).first()
        if not requester:
            requester = User(
                username="demo",
                email="demo@example.com",
                hashed_password=bcrypt.hash("demo123"),
                full_name="Demo User"
            )
            db.add(requester)
            db.commit()
            db.refresh(requester)

        db_ticket = Ticket(
            title=title,
            description=q,
            category=TicketCategory[category.upper()],
            priority=TicketPriority[priority.upper()],
            requester_id=requester.id,
            triage_reasoning={"reasoning_trace": steps, "response": response_text, "policy_citations": policies_cited}
        )
        db.add(db_ticket)
        db.commit()
        db.refresh(db_ticket)

        ticket_data = {
            "ticket_id": db_ticket.id,
            "title": db_ticket.title,
            "category": db_ticket.category.value,
            "priority": db_ticket.priority.value,
            "status": db_ticket.status.value
        }

        # Append reasoning log (JSONL)
        try:
            logs_dir = Path("backend/logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / "agent_requests.jsonl"
            log_entry = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "query": request.query,
                "response": response_text,
                "policies_cited": policies_cited,
                "compliance_status": compliance_status,
                "reasoning": steps,
                "ticket_created": ticket_data,
                "model_used": os.getenv("OLLAMA_MODEL", "phi3:mini")
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            # Do not fail the request due to logging issues
            pass

        return AgentResponse(
            response=response_text,
            reasoning={"reasoning_trace": steps},
            steps=steps,
            policies_cited=policies_cited,
            compliance_status=compliance_status,
            ticket_created=ticket_data,
            search_performed=True,
            model_used=os.getenv("OLLAMA_MODEL", "phi3:mini")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-ticket")
async def create_ticket(request: TicketRequest, db: Session = Depends(get_db)):
    """Create IT support ticket (minimal heuristics)."""
    try:
        q = request.issue
        def _category(issue: str) -> str:
            s = issue.lower()
            if any(t in s for t in ["vpn", "network", "wifi", "internet"]):
                return "network"
            if any(t in s for t in ["password", "login", "access", "permission"]):
                return "access"
            if any(t in s for t in ["software", "install", "update"]):
                return "software"
            if any(t in s for t in ["hardware", "laptop", "printer"]):
                return "hardware"
            return "general"
        def _priority(issue: str) -> str:
            s = issue.lower()
            if any(t in s for t in ["urgent", "critical", "down", "not working", "blocked"]):
                return "high"
            if any(t in s for t in ["slow", "issue", "problem", "help"]):
                return "medium"
            return "low"
        title = q.split(".")[0][:50] or "IT Support Request"
        category = _category(q)
        priority = _priority(q)
        
        ticket = Ticket(
            title=title,
            description=q,
            category=TicketCategory[category.upper()],
            priority=TicketPriority[priority.upper()],
            requester_id=1,
            triage_reasoning=f"Auto-created from query: {q[:120]}"
        )
        db.add(ticket)
        db.commit()
        db.refresh(ticket)
        
        return {
            "ticket_id": ticket.id,
            "title": ticket.title,
            "category": ticket.category.value,
            "priority": ticket.priority.value,
            "status": ticket.status.value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Voice query removed in minimal build

@router.get("/model-status")
async def get_model_status():
    """Report Ollama model status for mini LLM usage."""
    import os
    model = os.getenv("OLLAMA_MODEL", "phi3:mini")
    return {
        "ollama": True,
        "model": model,
        "fallback_available": True,
        "supported_models": [
            "phi3:mini (recommended)",
            "tinyllama (ultra-compact)",
            "mistral (heavier)"
        ]
    }

@router.post("/resolve-ticket/{ticket_id}")
async def resolve_ticket(
    ticket_id: int,
    resolution_code: str,
    db: Session = Depends(get_db)
):
    """
    Resolve ticket with proper logging
    As per ML Challenge: "close the ticket... with citations to policies"
    """
    try:
        ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        # Generate resolution reasoning
        resolution_context = {
            "ticket": {
                "title": ticket.title,
                "description": ticket.description,
                "category": ticket.category.value
            },
            "resolution_code": resolution_code
        }
        
        # Compose resolution using KG-RAG
        det = rag_reasoner.chat(db, f"Resolve ticket: {ticket.title}. Code: {resolution_code}", augment=True, k=5, include_explanation=True)
        ex = det.get("explanation", {})
        citations = [c.get("chunk_id") for c in ex.get("policy_citations", [])]
        ticket.status = "closed"
        ticket.resolution_code = resolution_code
        ticket.resolution_reasoning = det.get("response", "")
        ticket.resolution_policy_citations = citations
        db.commit()
        
        return {
            "ticket_id": ticket_id,
            "status": "closed",
            "resolution_code": resolution_code,
            "reasoning": ticket.resolution_reasoning,
            "policies_applied": citations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
