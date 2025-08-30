"""
Knowledge Graph API Router

Provides endpoints for knowledge graph visualization, querying, and management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Any, Optional
import json

from app.core.database import get_db
from app.core.security import get_current_user, require_role
from app.models.models import User, UserRole, KnowledgeGraphConcept, PolicyConceptExtraction
from app.models.schemas import (
    KnowledgeGraphConceptResponse, GraphVisualization,
    PolicyConceptExtractionResponse
)
from app.services.knowledge_graph_builder import PolicyKnowledgeGraphBuilder
from app.services.knowledge_graph_query import KnowledgeGraphQueryService
from app.services.policy_retriever import PolicyRetriever

router = APIRouter(prefix="/api/knowledge-graph", tags=["knowledge-graph"])

# Initialize services
kg_builder = PolicyKnowledgeGraphBuilder()
kg_query = KnowledgeGraphQueryService()
policy_retriever = PolicyRetriever()


@router.get("/stats")
async def get_knowledge_graph_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get statistics about the knowledge graph"""
    stats = kg_builder.get_graph_statistics(db)
    return stats


@router.get("/concepts", response_model=List[KnowledgeGraphConceptResponse])
async def get_all_concepts(
    limit: int = 50,
    concept_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all concepts in the knowledge graph"""
    query = db.query(KnowledgeGraphConcept)
    
    if concept_type:
        query = query.filter(KnowledgeGraphConcept.concept_type == concept_type)
    
    concepts = query.limit(limit).all()
    return concepts


@router.get("/concepts/{concept_name}/neighborhood")
async def get_concept_neighborhood(
    concept_name: str,
    radius: int = 2,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get concepts within a certain radius of the given concept"""
    neighborhood = kg_query.get_concept_neighborhood(db, concept_name, radius)
    return neighborhood


@router.get("/concepts/most-connected")
async def get_most_connected_concepts(
    top_k: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the most highly connected concepts in the graph"""
    concepts = kg_query.get_most_connected_concepts(db, top_k)
    return concepts


@router.get("/path/{source_concept}/{target_concept}")
async def find_shortest_path(
    source_concept: str,
    target_concept: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Find the shortest path between two concepts"""
    path = kg_query.find_shortest_path_between_concepts(db, source_concept, target_concept)
    
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"No path found between '{source_concept}' and '{target_concept}'"
        )
    
    return {"path": path, "length": len(path)}


@router.get("/visualization")
async def get_graph_visualization(
    highlight_concept: Optional[str] = None,
    max_nodes: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get graph visualization data for frontend display"""
    
    # Get all concepts (limited by max_nodes)
    concepts = db.query(KnowledgeGraphConcept).limit(max_nodes).all()
    
    if not concepts:
        return GraphVisualization(nodes=[], edges=[], query_concepts=[])
    
    # Build nodes for visualization
    nodes = []
    for concept in concepts:
        node = {
            "id": concept.id,
            "name": concept.name,
            "type": concept.concept_type,
            "importance": concept.importance_score,
            "chunk_count": len(concept.policy_chunks) if concept.policy_chunks else 0,
            "color": _get_node_color(concept.concept_type),
            "size": max(10, concept.importance_score * 30)
        }
        
        # Highlight specific concept if requested
        if highlight_concept and concept.name.lower() == highlight_concept.lower():
            node["highlighted"] = True
            node["color"] = "#FF6B6B"  # Red highlight
        
        nodes.append(node)
    
    # Get relationships
    concept_ids = [c.id for c in concepts]
    if concept_ids:
        relationships = db.execute(text("""
            SELECT cr.source_concept_id, cr.target_concept_id, cr.relationship_type, cr.weight,
                   source.name as source_name, target.name as target_name
            FROM concept_relationships cr
            JOIN kg_concepts source ON cr.source_concept_id = source.id
            JOIN kg_concepts target ON cr.target_concept_id = target.id
            WHERE source.id = ANY(:concept_ids) AND target.id = ANY(:concept_ids)
        """), {"concept_ids": concept_ids}).fetchall()
    else:
        relationships = []
    
    # Build edges for visualization
    edges = []
    for rel in relationships:
        edge = {
            "source": rel[0],
            "target": rel[1],
            "relationship": rel[2],
            "weight": rel[3],
            "label": rel[2],
            "color": _get_edge_color(rel[2]),
            "width": max(1, rel[3] * 3)
        }
        edges.append(edge)
    
    visualization = GraphVisualization(
        nodes=nodes,
        edges=edges,
        highlighted_path=None,
        query_concepts=[]
    )
    
    return visualization


@router.post("/query-concepts")
async def query_concepts_in_text(
    request: Dict[str, str],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Find IT concepts mentioned in arbitrary text"""
    text = request.get("text", "")
    concepts = kg_builder.find_concepts_in_text(text)
    return {
        "text": text,
        "concepts_found": [
            {"concept": concept, "confidence": confidence}
            for concept, confidence in concepts
        ]
    }


@router.post("/rebuild")
async def rebuild_knowledge_graph(
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: Session = Depends(get_db)
):
    """Rebuild the knowledge graph from policy documents (Admin only)"""
    try:
        result = kg_builder.rebuild_graph(db)
        return {
            "success": True,
            "message": "Knowledge graph rebuilt successfully",
            **result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rebuild knowledge graph: {str(e)}"
        )


@router.post("/relationships")
async def add_manual_relationship(
    request: Dict[str, Any],
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: Session = Depends(get_db)
):
    """Manually add a relationship between two concepts (Admin only)"""
    
    source_concept = request.get("source_concept")
    target_concept = request.get("target_concept")
    relationship_type = request.get("relationship_type")
    weight = request.get("weight", 1.0)
    description = request.get("description", "")
    
    if not all([source_concept, target_concept, relationship_type]):
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: source_concept, target_concept, relationship_type"
        )
    
    success = kg_builder.add_manual_relationship(
        db=db,
        source_concept=source_concept,
        target_concept=target_concept,
        relationship_type=relationship_type,
        weight=weight,
        description=description
    )
    
    if success:
        return {
            "success": True,
            "message": f"Added {relationship_type} relationship: {source_concept} → {target_concept}"
        }
    else:
        raise HTTPException(
            status_code=400,
            detail="Failed to add relationship. Check that both concepts exist."
        )


@router.post("/test-kg-rag")
async def test_kg_enhanced_rag(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Test the KG-Enhanced RAG system with a query"""
    
    query = request.get("query", "")
    enable_kg = request.get("enable_kg", True)
    max_hops = request.get("max_hops", 2)
    k = request.get("k", 5)
    
    if not query:
        raise HTTPException(status_code=400, detail="Query text is required")
    
    try:
        enhanced_citations, graph_hops, metadata = policy_retriever.kg_enhanced_retrieve(
            query=query,
            k=k,
            enable_kg=enable_kg,
            max_graph_hops=max_hops,
            db=db
        )
        
        # Generate explanation
        explanation = policy_retriever.explain_retrieval_reasoning(
            query=query,
            enhanced_citations=enhanced_citations,
            graph_hops=graph_hops,
            metadata=metadata
        )
        
        return {
            "query": query,
            "enhanced_citations": [
                {
                    "document_title": c.document_title,
                    "chunk_id": c.chunk_id,
                    "semantic_score": c.semantic_score,
                    "graph_boost_score": c.graph_boost_score,
                    "combined_score": c.combined_score,
                    "graph_path_length": len(c.graph_path)
                }
                for c in enhanced_citations
            ],
            "graph_hops": graph_hops,
            "metadata": metadata,
            "explanation": explanation
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"KG-Enhanced RAG test failed: {str(e)}"
        )


@router.get("/debug/extractions/{chunk_id}")
async def get_concept_extractions_for_chunk(
    chunk_id: int,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: Session = Depends(get_db)
):
    """Get all concept extractions for a specific chunk (Admin only)"""
    
    extractions = db.query(PolicyConceptExtraction).filter(
        PolicyConceptExtraction.chunk_id == chunk_id
    ).all()
    
    return [
        {
            "chunk_id": ext.chunk_id,
            "concept_id": ext.concept_id,
            "concept_name": ext.concept.name,
            "confidence_score": ext.confidence_score,
            "context_window": ext.context_window,
            "extraction_method": ext.extraction_method,
            "created_at": ext.created_at
        }
        for ext in extractions
    ]


def _get_node_color(concept_type: str) -> str:
    """Get visualization color for concept type"""
    color_map = {
        "technology": "#4ECDC4",  # Teal
        "security": "#FF6B6B",    # Red  
        "policy": "#45B7D1",      # Blue
        "procedure": "#96CEB4",   # Green
        "requirement": "#FFEAA7"  # Yellow
    }
    return color_map.get(concept_type, "#DDA0DD")  # Default: Plum


def _get_edge_color(relationship_type: str) -> str:
    """Get visualization color for relationship type"""
    color_map = {
        "requires": "#E74C3C",     # Red
        "depends_on": "#F39C12",   # Orange
        "overrides": "#8E44AD",    # Purple
        "related_to": "#3498DB",   # Blue
        "enables": "#27AE60",      # Green
        "affects": "#F1C40F",      # Yellow
        "protects_from": "#E67E22" # Dark Orange
    }
    return color_map.get(relationship_type, "#95A5A6")  # Default: Gray
