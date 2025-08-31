"""
KG Query API (compat layer for UI)
- /api/kg/query: accepts { concepts: [str], max_hops?: int }
  returns { related: [str], hops: [{from,to,depth,relationship}] }
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.models import KnowledgeGraphConcept
from app.services.kg_lite_service import kg_lite_service
from sqlalchemy import text

router = APIRouter()

class KGQueryRequest(BaseModel):
    concepts: List[str]
    max_hops: Optional[int] = 2

@router.post("/query")
async def kg_query(req: KGQueryRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    try:
        # Resolve concepts to IDs
        concept_ids = []
        for name in req.concepts:
            c = db.query(KnowledgeGraphConcept).filter(KnowledgeGraphConcept.name.ilike(f"%{name}%")).first()
            if c:
                concept_ids.append(c.id)
        if not concept_ids:
            return {"related": [], "hops": []}

        # Collect immediate neighbors as related list
        related_set = set()
        for cid in concept_ids:
            for nb in kg_lite_service.neighbors(db, cid):
                if nb not in concept_ids:
                    related_set.add(nb)
        # Map IDs to names
        related_names = []
        if related_set:
            rows = db.query(KnowledgeGraphConcept).filter(KnowledgeGraphConcept.id.in_(list(related_set))).all()
            related_names = [r.name for r in rows]

        # Build simple hops via BFS up to max_hops
        hops = []
        frontier = [(cid, 0) for cid in concept_ids]
        visited = set(concept_ids)
        while frontier:
            current, depth = frontier.pop(0)
            if depth >= (req.max_hops or 2):
                continue
            for nb in kg_lite_service.neighbors(db, current):
                if nb not in visited:
                    visited.add(nb)
                    frontier.append((nb, depth + 1))
                    cur_name = db.query(KnowledgeGraphConcept).get(current).name
                    nb_name = db.query(KnowledgeGraphConcept).get(nb).name
                    hops.append({
                        "from": cur_name,
                        "to": nb_name,
                        "depth": depth + 1,
                        "relationship": "related_to"
                    })

        return {"related": sorted(related_names), "hops": hops}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

