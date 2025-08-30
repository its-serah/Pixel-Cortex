from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from app.core.database import get_db
from app.core.security import verify_token, require_role
from app.models.models import Ticket, TicketEvent, User
from app.models.schemas import TicketCreate, TicketResponse, TicketUpdate, TicketEventResponse, TicketEventCreate
from app.services.xai_service import XAIService
from app.services.audit_service import AuditService
from app.services.decision_service import DecisionService, DecisionType

router = APIRouter()

@router.post("/", response_model=TicketResponse)
async def create_ticket(
    ticket: TicketCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(verify_token),
    audit_service: AuditService = Depends()
):
    # Get user from database
    user = db.query(User).filter(User.username == current_user["username"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Use XAI service for automatic triage if category not specified
    xai_service = XAIService()
    
    if not ticket.category:
        result = xai_service.explain_ticket_processing(
            ticket.title, ticket.description, db, include_planning=False
        )
        category = result["category"]
        priority = result["priority"]
        triage_confidence = result["confidence"]
        explanation_data = result["explanation"].dict()
    else:
        category = ticket.category
        priority = ticket.priority
        triage_confidence = None
        explanation_data = None
    
    # Create ticket
    db_ticket = Ticket(
        title=ticket.title,
        description=ticket.description,
        category=category,
        priority=priority,
        requester_id=user.id,
        triage_confidence=triage_confidence,
        triage_reasoning=explanation_data
    )
    
    db.add(db_ticket)
    db.commit()
    db.refresh(db_ticket)
    
    # Create initial event
    event = TicketEvent(
        ticket_id=db_ticket.id,
        user_id=user.id,
        event_type="created",
        new_value=f"Ticket created with {category.value} category and {priority.value} priority",
        explanation_data=explanation_data
    )
    
    db.add(event)
    db.commit()
    
    # Log in audit trail
    await audit_service.log_event(
        action="create_ticket",
        resource_type="ticket",
        resource_id=str(db_ticket.id),
        user_id=current_user["username"],
        event_data={
            "ticket_id": db_ticket.id,
            "title": ticket.title,
            "category": category.value,
            "priority": priority.value,
            "auto_triaged": triage_confidence is not None
        }
    )
    
    return db_ticket

@router.get("/", response_model=List[TicketResponse])
async def list_tickets(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    category: Optional[str] = None,
    priority: Optional[str] = None,
    assigned_to_me: bool = False,
    db: Session = Depends(get_db),
    current_user: dict = Depends(verify_token)
):
    query = db.query(Ticket)
    
    # Filter by assignment if requested
    if assigned_to_me:
        user = db.query(User).filter(User.username == current_user["username"]).first()
        if user:
            query = query.filter(Ticket.assigned_agent_id == user.id)
    
    # Apply filters
    if status:
        query = query.filter(Ticket.status == status)
    if category:
        query = query.filter(Ticket.category == category)
    if priority:
        query = query.filter(Ticket.priority == priority)
    
    # Non-admin users can only see their own tickets or assigned tickets
    if "admin" not in current_user.get("roles", []):
        user = db.query(User).filter(User.username == current_user["username"]).first()
        if user:
            query = query.filter(
                (Ticket.requester_id == user.id) | (Ticket.assigned_agent_id == user.id)
            )
    
    tickets = query.offset(skip).limit(limit).all()
    return tickets

@router.get("/{ticket_id}", response_model=TicketResponse)
async def get_ticket(
    ticket_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(verify_token)
):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Check permissions
    if "admin" not in current_user.get("roles", []):
        user = db.query(User).filter(User.username == current_user["username"]).first()
        if user and ticket.requester_id != user.id and ticket.assigned_agent_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
    
    return ticket

@router.put("/{ticket_id}", response_model=TicketResponse)
async def update_ticket(
    ticket_id: int,
    ticket_update: TicketUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(verify_token),
    audit_service: AuditService = Depends()
):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Check permissions
    user = db.query(User).filter(User.username == current_user["username"]).first()
    can_edit = (
        "admin" in current_user.get("roles", []) or
        "agent" in current_user.get("roles", []) or
        (user and ticket.requester_id == user.id)
    )
    
    if not can_edit:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Track changes for audit
    changes = {}
    update_data = ticket_update.dict(exclude_unset=True)
    
    for field, new_value in update_data.items():
        old_value = getattr(ticket, field)
        if old_value != new_value:
            changes[field] = {"old": str(old_value), "new": str(new_value)}
            setattr(ticket, field, new_value)
    
    if changes:
        db.commit()
        db.refresh(ticket)
        
        # Create event for significant changes
        for field, change in changes.items():
            event = TicketEvent(
                ticket_id=ticket.id,
                user_id=user.id,
                event_type="updated",
                old_value=change["old"],
                new_value=change["new"]
            )
            db.add(event)
        
        db.commit()
        
        # Log in audit trail
        await audit_service.log_event(
            action="update_ticket",
            resource_type="ticket",
            resource_id=str(ticket.id),
            user_id=current_user["username"],
            event_data={
                "ticket_id": ticket.id,
                "changes": changes
            }
        )
    
    return ticket


@router.get("/{ticket_id}/events", response_model=List[TicketEventResponse])
async def get_ticket_events(
    ticket_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(verify_token)
):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Check permissions
    if "admin" not in current_user.get("roles", []):
        user = db.query(User).filter(User.username == current_user["username"]).first()
        if user and ticket.requester_id != user.id and ticket.assigned_agent_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
    
    events = db.query(TicketEvent).filter(TicketEvent.ticket_id == ticket_id).order_by(TicketEvent.created_at).all()
    return events

@router.post("/{ticket_id}/events", response_model=TicketEventResponse)
async def add_ticket_event(
    ticket_id: int,
    event: TicketEventCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(verify_token),
    audit_service: AuditService = Depends()
):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    user = db.query(User).filter(User.username == current_user["username"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Create event
    db_event = TicketEvent(
        ticket_id=ticket_id,
        user_id=user.id,
        event_type=event.event_type,
        old_value=event.old_value,
        new_value=event.new_value,
        comment=event.comment
    )
    
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    
    # Log in audit trail
    await audit_service.log_event(
        action="add_ticket_event",
        resource_type="ticket_event",
        resource_id=str(db_event.id),
        user_id=current_user["username"],
        event_data={
            "ticket_id": ticket_id,
            "event_type": event.event_type,
            "comment": event.comment
        }
    )
    
    return db_event

@router.post("/{ticket_id}/assign")
async def assign_ticket(
    ticket_id: int,
    agent_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_role("admin")),
    audit_service: AuditService = Depends()
):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    agent = db.query(User).filter(User.id == agent_id).first()
    if not agent or agent.role.value not in ["admin", "agent"]:
        raise HTTPException(status_code=400, detail="Invalid agent")
    
    old_agent_id = ticket.assigned_agent_id
    ticket.assigned_agent_id = agent_id
    
    db.commit()
    
    # Create assignment event
    user = db.query(User).filter(User.username == current_user["username"]).first()
    event = TicketEvent(
        ticket_id=ticket_id,
        user_id=user.id,
        event_type="assigned",
        old_value=str(old_agent_id) if old_agent_id else None,
        new_value=str(agent_id)
    )
    
    db.add(event)
    db.commit()
    
    # Log in audit trail
    await audit_service.log_event(
        action="assign_ticket",
        resource_type="ticket",
        resource_id=str(ticket_id),
        user_id=current_user["username"],
        event_data={
            "ticket_id": ticket_id,
            "old_agent_id": old_agent_id,
            "new_agent_id": agent_id,
            "agent_username": agent.username
        }
    )
    
    return {"message": f"Ticket assigned to {agent.username}"}

@router.post("/{ticket_id}/process")
async def process_ticket_for_decision(
    ticket_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(verify_token)
):
    """Process ticket through AI decision system - returns clean decision summary only"""
    from app.services.triage_service import TriageService
    from app.services.planner_service import PlannerService
    from app.services.xai_builder_service import XAIBuilderService
    
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Initialize services
    triage_service = TriageService()
    decision_service = DecisionService()
    planner_service = PlannerService()
    xai_builder = XAIBuilderService()
    audit_service = AuditService()
    
    # 1. Triage
    triage_result = triage_service.triage_ticket(ticket.title, ticket.description)
    ticket.category = triage_result["category"]
    ticket.priority = triage_result["priority"]
    
    # 2. Decision
    decision, decision_explanation = decision_service.make_decision(
        ticket.title, ticket.description, ticket.category, ticket.priority, db
    )
    
    # 3. Planning
    plan_explanation = planner_service.generate_plan(
        ticket.title, ticket.description, ticket.category, ticket.priority, db
    )
    
    # 4. Build full explanation (for audit logs)
    full_explanation = xai_builder.build_composite_explanation(
        triage_result["explanation"], plan_explanation, decision_explanation
    )
    
    # 5. Secure audit logging
    audit_service.log_audit_entry(
        db=db,
        ticket_id=ticket_id,
        action="ai_decision_complete",
        details={
            "category": ticket.category.value,
            "priority": ticket.priority.value,
            "decision": decision.value,
            "confidence": full_explanation.confidence
        },
        explanation=full_explanation,
        performed_by=current_user["user_id"]
    )
    
    db.commit()
    
    # Return ONLY user-friendly decision summary
    return {
        "ticket_id": ticket_id,
        "decision": decision.value,
        "category": ticket.category.value,
        "priority": ticket.priority.value,
        "summary": _get_clean_decision_summary(decision, ticket.category, ticket.priority),
        "confidence": round(full_explanation.confidence, 2),
        "timestamp": datetime.now().isoformat()
    }

def _get_clean_decision_summary(decision: DecisionType, category, priority) -> str:
    """Generate clean decision summary for users (no internal reasoning exposed)"""
    category_name = category.value.replace('_', ' ').title()
    priority_name = priority.value.title()
    
    summaries = {
        DecisionType.ALLOWED: f"✅ {category_name} request ({priority_name} priority) has been approved and will proceed according to standard procedures.",
        DecisionType.DENIED: f"❌ {category_name} request ({priority_name} priority) has been denied based on current policies. Please review requirements and resubmit if needed.",
        DecisionType.NEEDS_APPROVAL: f"⏳ {category_name} request ({priority_name} priority) requires additional approval before proceeding. Management review has been requested."
    }
    
    return summaries.get(decision, f"Decision completed for {category_name} request.")

@router.get("/{ticket_id}/explanation")
async def get_ticket_explanation(
    ticket_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(verify_token)
):
    """Get the XAI explanation for how this ticket was processed"""
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Check permissions
    if "admin" not in current_user.get("roles", []):
        user = db.query(User).filter(User.username == current_user["username"]).first()
        if user and ticket.requester_id != user.id and ticket.assigned_agent_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
    
    # Get latest explanation from triage reasoning or events
    explanation_data = ticket.triage_reasoning
    
    if not explanation_data:
        # Look for explanation in recent events
        latest_event = db.query(TicketEvent).filter(
            TicketEvent.ticket_id == ticket_id,
            TicketEvent.explanation_data.isnot(None)
        ).order_by(TicketEvent.created_at.desc()).first()
        
        if latest_event:
            explanation_data = latest_event.explanation_data
    
    if not explanation_data:
        # Generate fresh explanation
        xai_service = XAIService()
        result = xai_service.explain_ticket_processing(
            ticket.title, ticket.description, db, include_planning=True
        )
        explanation_data = result["explanation"].dict()
    
    return {
        "ticket_id": ticket_id,
        "explanation": explanation_data,
        "generated_at": datetime.now()
    }
