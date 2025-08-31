"""
Runtime-safe AuditService (no external heavy deps)
- async log_event for simple audit entries with hash chaining
- log_audit_entry for compatibility with existing codepaths
"""
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib
import uuid
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.core.database import SessionLocal
from app.models.models import AuditLog

class AuditService:
    def __init__(self):
        pass

    def _calc_hash(
        self,
        event_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        user_id: str,
        timestamp: datetime,
        event_data: Dict[str, Any],
        previous_hash: Optional[str] = None,
    ) -> str:
        payload = {
            "event_id": event_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "user_id": user_id,
            "timestamp": timestamp.isoformat(),
            "event_data": event_data or {},
            "previous_hash": previous_hash or "",
        }
        # Deterministic serialization
        import json
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _last_hash(self, db: Session) -> Optional[str]:
        last = db.query(AuditLog).order_by(desc(AuditLog.id)).first()
        return last.current_hash if last else None

    async def log_event(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        user_id: str,
        event_data: Dict[str, Any],
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> str:
        sess = db or SessionLocal()
        try:
            event_id = str(uuid.uuid4())
            ts = datetime.utcnow()
            prev = self._last_hash(sess)
            cur = self._calc_hash(event_id, action, resource_type, resource_id, str(user_id), ts, event_data, prev)
            entry = AuditLog(
                event_id=event_id,
                previous_hash=prev,
                current_hash=cur,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                user_id=user_id if isinstance(user_id, int) else 0,
                timestamp=ts,
                event_data=event_data,
                ip_address=ip_address,
                user_agent=user_agent,
            )
            sess.add(entry)
            sess.commit()
            return event_id
        finally:
            if db is None:
                sess.close()

    # Sync variant used by some flows
    def log_audit_entry(
        self,
        db: Session,
        ticket_id: int,
        action: str,
        details: Dict[str, Any],
        explanation: Any,
        performed_by: Any,
    ) -> str:
        event_id = str(uuid.uuid4())
        ts = datetime.utcnow()
        prev = self._last_hash(db)
        cur = self._calc_hash(event_id, action, "ticket", str(ticket_id), str(performed_by), ts, details, prev)
        entry = AuditLog(
            event_id=event_id,
            previous_hash=prev,
            current_hash=cur,
            action=action,
            resource_type="ticket",
            resource_id=str(ticket_id),
            user_id=performed_by if isinstance(performed_by, int) else 0,
            timestamp=ts,
            event_data={**details, "explanation_model": getattr(explanation, "model_version", "1.0.0")},
            ip_address=None,
            user_agent=None,
        )
        db.add(entry)
        db.commit()
        return event_id

# Dependency injection helper

def get_audit_service() -> AuditService:
    return AuditService()

