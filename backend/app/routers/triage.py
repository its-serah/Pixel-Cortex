from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import verify_token
from app.models.schemas import TriageRequest, TriageResponse
from app.services.triage_service import TriageService
from app.services.planner_service import PlannerService
from app.services.xai_builder_service import XAIBuilderService
from app.services.audit_service import AuditService
from app.services.decision_service import DecisionService, DecisionType
from app.models.models import Ticket

router = APIRouter()

@router.post("/analyze", response_model=TriageResponse)
async def analyze_ticket(
    request: TriageRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(verify_token)
):
    """
    Analyze a ticket for category and priority with XAI explanation
    """
    triage_service = TriageService()
    
    result = triage_service.triage_ticket(
        request.title, 
        request.description
    )
    
    return TriageResponse(
        category=result["category"],
        priority=result["priority"],
        confidence=result["confidence"],
        explanation=result["explanation"]
    )

@router.post("/process-ticket/{ticket_id}")
async def process_ticket(
    ticket_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(verify_token)
):
    """
    Full ticket processing: triage → decision → planning → audit logging
    Returns only final decision summary, full explanations logged securely
    """
    # Initialize services
    triage_service = TriageService()
    decision_service = DecisionService()
    planner_service = PlannerService()
    xai_builder = XAIBuilderService()
    audit_service = AuditService()
    
    # Get ticket
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # 1. Triage ticket (category/priority)
    triage_result = triage_service.triage_ticket(ticket.title, ticket.description)
    triage_explanation = triage_result["explanation"]
    
    # Update ticket with triage results
    ticket.category = triage_result["category"]
    ticket.priority = triage_result["priority"]
    
    # 2. Make policy-grounded decision
    decision, decision_explanation = decision_service.make_decision(
        ticket.title, ticket.description, ticket.category, ticket.priority, db
    )
    
    # 3. Plan resolution steps
    plan_explanation = planner_service.generate_plan(
        ticket.title, ticket.description, ticket.category, ticket.priority, db
    )
    
    # 4. Build composite XAI explanation (FULL REASONING - LOGS ONLY)
    final_explanation = xai_builder.build_composite_explanation(
        triage_explanation, plan_explanation, decision_explanation
    )
    
    # 5. Log everything to tamper-evident audit trail (PII redacted)
    audit_service.log_audit_entry(
        db=db,
        ticket_id=ticket.id,
        action="ticket_processed",
        details={
            "category": ticket.category.value,
            "priority": ticket.priority.value,
            "decision": decision.value,
            "confidence": final_explanation.confidence
        },
        explanation=final_explanation,
        performed_by=current_user.id
    )
    
    db.commit()
    
    # Return ONLY final decision summary (NOT full explanation)
    return {
        "ticket_id": ticket_id,
        "category": ticket.category.value,
        "priority": ticket.priority.value,
        "decision": decision.value,
        "decision_summary": _get_user_friendly_summary(decision, ticket.category),
        "confidence": round(final_explanation.confidence, 2),
        "message": "Ticket processed and logged successfully"
    }

def _get_user_friendly_summary(decision: DecisionType, category) -> str:
    """Convert decision to user-friendly message"""
    summaries = {
        DecisionType.ALLOWED: f"✅ {category.value.title()} request approved - proceeding with standard process",
        DecisionType.DENIED: f"❌ {category.value.title()} request denied - please review company policies",
        DecisionType.NEEDS_APPROVAL: f"⏳ {category.value.title()} request requires management approval"
    }
    return summaries.get(decision, "Decision processed")

@router.get("/categories")
async def get_categories(current_user: dict = Depends(verify_token)):
    """Get available ticket categories"""
    from app.models.models import TicketCategory
    return {
        "categories": [category.value for category in TicketCategory]
    }

@router.get("/priorities")
async def get_priorities(current_user: dict = Depends(verify_token)):
    """Get available ticket priorities"""
    from app.models.models import TicketPriority
    return {
        "priorities": [priority.value for priority in TicketPriority]
    }
