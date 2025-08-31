"""
IT Support Agent API - Implements ML Challenge Requirements
- Local inference with TinyLlama
- Chain-of-Thought reasoning
- Ticket creation and management
- Policy-based responses with citations
- Search integration
- Voice support (via Vosk)
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
import io

from app.core.database import get_db
from app.services.local_sml_service import local_sml_service
from app.services.policy_retriever import policy_retriever
from app.services.audio_vosk_service import vosk_service
from app.models.models import Ticket, TicketCategory, TicketPriority

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
        context = {}
        
        # Step 1: Check if we need to search
        should_search, search_terms = local_sml_service.should_search(request.query)
        
        # Step 2: Retrieve relevant policies if requested
        if request.include_policies:
            policies = policy_retriever.retrieve_relevant_chunks(
                request.query, 
                k=3, 
                db=db
            )
            context["policies"] = [
                {
                    "title": f"Policy {p.document_id}",
                    "content": p.chunk_content,
                    "relevance": p.relevance_score
                }
                for p in policies
            ]
        
        # Step 3: Get previous solutions from history if requested
        if request.include_history:
            # Mock history for demo - in production, query ConversationLog
            context["history"] = [
                {
                    "problem": "VPN connection failed",
                    "solution": "Reset MFA token and clear cache"
                }
            ]
        
        # Step 4: Perform search if needed
        search_results = []
        if should_search and search_terms:
            # Mock search - in production, use search service
            search_results = [f"Found info about {term}" for term in search_terms]
            context["search_results"] = search_results
        
        # Step 5: Generate Chain-of-Thought reasoning
        cot_result = local_sml_service.reason_with_cot(request.query, context)
        
        # Step 6: Extract compliance status and policy citations
        compliance_status = "allowed"  # Default
        policies_cited = []
        
        for step in cot_result["steps"]:
            if "status" in step:
                if "DENIED" in step["status"]:
                    compliance_status = "denied"
                elif "NEEDS_APPROVAL" in step["status"]:
                    compliance_status = "needs_approval"
                else:
                    compliance_status = "allowed"
            
            if "policy_ref" in step:
                policies_cited.append(step["policy_ref"])
        
        # Step 7: Check if we need to create a ticket
        ticket_data = None
        if "ticket" in request.query.lower() or "issue" in request.query.lower():
            ticket_data = local_sml_service.create_ticket(request.query)
        
        # Step 8: Format response
        response_text = cot_result["reasoning"]
        if ticket_data:
            response_text += f"\n\nTicket Created: {ticket_data['id']}"
        
        return AgentResponse(
            response=response_text,
            reasoning=cot_result,
            steps=cot_result["steps"],
            policies_cited=policies_cited,
            compliance_status=compliance_status,
            ticket_created=ticket_data,
            search_performed=should_search,
            model_used=cot_result["model"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-ticket")
async def create_ticket(request: TicketRequest, db: Session = Depends(get_db)):
    """
    Create IT support ticket
    As per ML Challenge: automatically create tickets for IT issues
    """
    try:
        ticket_data = local_sml_service.create_ticket(request.issue, request.user_id)
        
        # Save to database
        ticket = Ticket(
            title=ticket_data["title"],
            description=ticket_data["description"],
            category=TicketCategory[ticket_data["category"].upper()],
            priority=TicketPriority[ticket_data["priority"].upper()],
            requester_id=1,  # Demo user
            triage_reasoning=ticket_data["reasoning"]
        )
        
        db.add(ticket)
        db.commit()
        db.refresh(ticket)
        
        return {
            "ticket_id": ticket.id,
            "title": ticket.title,
            "category": ticket.category.value,
            "priority": ticket.priority.value,
            "status": ticket.status.value,
            "reasoning": ticket_data["reasoning"],
            "steps": ticket_data["steps"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/voice-query")
async def voice_query(
    audio: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Handle voice queries
    As per ML Challenge: "Communicate with the LM via your voice"
    """
    try:
        # Read audio file
        audio_bytes = await audio.read()
        
        # Transcribe using Vosk
        transcription = vosk_service.transcribe_audio(
            audio_bytes,
            audio.filename.split(".")[-1]
        )
        
        if not transcription:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")
        
        # Process the transcribed query
        query_request = AgentQuery(query=transcription)
        response = await ask_it_agent(query_request, db)
        
        # Add transcription to response
        response_dict = response.dict()
        response_dict["transcription"] = transcription
        
        return response_dict
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-status")
async def get_model_status():
    """Check which model is being used for inference"""
    has_model = local_sml_service.model is not None
    
    return {
        "local_inference": has_model,
        "model": "TinyLlama-1.1B" if has_model else "rule-based",
        "model_path": local_sml_service.model_path,
        "fallback_available": True,
        "supported_models": [
            "TinyLlama (1.1B) - Ultra-compact",
            "Phi-3-mini - Lightweight",
            "Mistral 7B - If you have RAM",
            "Rule-based - Always available"
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
        
        cot_result = local_sml_service.reason_with_cot(
            f"Resolve ticket: {ticket.title} with code: {resolution_code}",
            resolution_context
        )
        
        # Update ticket
        ticket.status = "closed"
        ticket.resolution_code = resolution_code
        ticket.resolution_reasoning = cot_result["reasoning"]
        
        db.commit()
        
        return {
            "ticket_id": ticket_id,
            "status": "closed",
            "resolution_code": resolution_code,
            "reasoning": cot_result["reasoning"],
            "policies_applied": [
                step.get("policy_ref", "") 
                for step in cot_result["steps"] 
                if "policy_ref" in step
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
