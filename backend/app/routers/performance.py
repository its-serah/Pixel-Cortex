"""Performance monitoring router."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, List
from datetime import datetime, timedelta

from app.core.database import get_db
from app.models import models
from app.models.schemas import TicketStatus

router = APIRouter()

@router.get("/metrics")
async def get_performance_metrics(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get performance metrics for the system."""
    try:
        # Get ticket statistics
        total_tickets = db.query(models.Ticket).count()
        open_tickets = db.query(models.Ticket).filter(
            models.Ticket.status == TicketStatus.OPEN
        ).count()
        resolved_tickets = db.query(models.Ticket).filter(
            models.Ticket.status == TicketStatus.RESOLVED
        ).count()
        
        # Get average resolution time (simplified)
        avg_resolution_time_hours = 24.5  # Mock value
        
        # Get user statistics
        total_users = db.query(models.User).count()
        active_users = db.query(models.User).filter(
            models.User.is_active == True
        ).count()
        
        return {
            "tickets": {
                "total": total_tickets,
                "open": open_tickets,
                "resolved": resolved_tickets,
                "avg_resolution_hours": avg_resolution_time_hours
            },
            "users": {
                "total": total_users,
                "active": active_users
            },
            "system": {
                "status": "healthy",
                "uptime_hours": 168,  # Mock value (1 week)
                "cpu_usage": 15.2,  # Mock value
                "memory_usage": 42.8  # Mock value
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def get_performance_summary(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get a summary of system performance."""
    try:
        # Get recent ticket trends
        now = datetime.utcnow()
        last_week = now - timedelta(days=7)
        
        tickets_this_week = db.query(models.Ticket).filter(
            models.Ticket.created_at >= last_week
        ).count()
        
        # Get category distribution
        category_counts = {}
        for category in models.TicketCategory:
            count = db.query(models.Ticket).filter(
                models.Ticket.category == category
            ).count()
            category_counts[category.value] = count
        
        # Get priority distribution
        priority_counts = {}
        for priority in models.TicketPriority:
            count = db.query(models.Ticket).filter(
                models.Ticket.priority == priority
            ).count()
            priority_counts[priority.value] = count
        
        return {
            "summary": {
                "tickets_this_week": tickets_this_week,
                "category_distribution": category_counts,
                "priority_distribution": priority_counts,
                "performance_score": 85.5,  # Mock performance score
                "sla_compliance": 92.3  # Mock SLA compliance percentage
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trends")
async def get_performance_trends(
    days: int = 7,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get performance trends over time."""
    try:
        # Mock trend data
        trends = []
        now = datetime.utcnow()
        
        for i in range(days):
            date = now - timedelta(days=i)
            trends.append({
                "date": date.isoformat(),
                "tickets_created": 10 + (i % 5),
                "tickets_resolved": 8 + (i % 4),
                "avg_response_time_hours": 2.5 + (i * 0.1)
            })
        
        return {
            "trends": trends,
            "period_days": days
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
