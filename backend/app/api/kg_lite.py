"""
KG Lite API
- /rebuild: build concepts and co-occurrence relationships
- /stats: counts
- /concepts: list
- /path: shortest path by names
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List, Optional

from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.kg_lite_service import kg_lite_service

router = APIRouter()

@router.post("/rebuild")
async def rebuild(db: Session = Depends(get_db)) -> Dict[str, Any]:
    try:
        return kg_lite_service.rebuild_from_policies(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def stats(db: Session = Depends(get_db)) -> Dict[str, Any]:
    try:
        return kg_lite_service.get_stats(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/concepts")
async def concepts(limit: int = 50, db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    try:
        return kg_lite_service.get_concepts(db, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/path")
async def path(source: str, target: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    try:
        names = kg_lite_service.path_by_name(db, source, target)
        if not names:
            raise HTTPException(status_code=404, detail="No path found")
        return {"path": names, "length": len(names)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

