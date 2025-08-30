"""
Knowledge Graph API Router

Provides endpoints for knowledge graph visualization, querying, and management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import json

from app.core.database import get_db
from app.core.security import get_current_user, require_role
from app.models.models import User, UserRole, KnowledgeGraphConcept
from app.models.schemas import (
    KnowledgeGraphConceptResponse, KnowledgeGraphConceptCreate,
    GraphVisualization, ConceptRelationship,
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


@router.get("/stats", response_model=Dict[str, Any])
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
            detail=f"No path found between '{source_concept}' and '{target_concept}'\"\n        )\n    \n    return {\"path\": path, \"length\": len(path)}\n\n\n@router.get(\"/visualization\")\nasync def get_graph_visualization(\n    highlight_concept: Optional[str] = None,\n    max_nodes: int = 50,\n    current_user: User = Depends(get_current_user),\n    db: Session = Depends(get_db)\n):\n    \"\"\"Get graph visualization data for frontend display\"\"\"\n    \n    # Get all concepts (limited by max_nodes)\n    concepts = db.query(KnowledgeGraphConcept).limit(max_nodes).all()\n    \n    # Build nodes for visualization\n    nodes = []\n    for concept in concepts:\n        node = {\n            \"id\": concept.id,\n            \"name\": concept.name,\n            \"type\": concept.concept_type,\n            \"importance\": concept.importance_score,\n            \"chunk_count\": len(concept.policy_chunks) if concept.policy_chunks else 0,\n            \"color\": _get_node_color(concept.concept_type),\n            \"size\": max(10, concept.importance_score * 30)\n        }\n        \n        # Highlight specific concept if requested\n        if highlight_concept and concept.name.lower() == highlight_concept.lower():\n            node[\"highlighted\"] = True\n            node[\"color\"] = \"#FF6B6B\"  # Red highlight\n        \n        nodes.append(node)\n    \n    # Get relationships\n    from sqlalchemy import text\n    relationships = db.execute(text(\"\"\"\n        SELECT cr.source_concept_id, cr.target_concept_id, cr.relationship_type, cr.weight,\n               source.name as source_name, target.name as target_name\n        FROM concept_relationships cr\n        JOIN kg_concepts source ON cr.source_concept_id = source.id\n        JOIN kg_concepts target ON cr.target_concept_id = target.id\n        WHERE source.id IN :concept_ids AND target.id IN :concept_ids\n    \"\"\"), {\"concept_ids\": tuple([c.id for c in concepts])}).fetchall()\n    \n    # Build edges for visualization\n    edges = []\n    for rel in relationships:\n        edge = {\n            \"source\": rel[0],\n            \"target\": rel[1],\n            \"relationship\": rel[2],\n            \"weight\": rel[3],\n            \"label\": rel[2],\n            \"color\": _get_edge_color(rel[2]),\n            \"width\": max(1, rel[3] * 3)\n        }\n        edges.append(edge)\n    \n    visualization = GraphVisualization(\n        nodes=nodes,\n        edges=edges,\n        highlighted_path=None,\n        query_concepts=[]\n    )\n    \n    return visualization\n\n\n@router.post(\"/query-concepts\")\nasync def query_concepts_in_text(\n    text: str,\n    current_user: User = Depends(get_current_user),\n    db: Session = Depends(get_db)\n):\n    \"\"\"Find IT concepts mentioned in arbitrary text\"\"\"\n    concepts = kg_builder.find_concepts_in_text(text)\n    return {\n        \"text\": text,\n        \"concepts_found\": [\n            {\"concept\": concept, \"confidence\": confidence}\n            for concept, confidence in concepts\n        ]\n    }\n\n\n@router.post(\"/rebuild\")\nasync def rebuild_knowledge_graph(\n    current_user: User = Depends(require_role(UserRole.ADMIN)),\n    db: Session = Depends(get_db)\n):\n    \"\"\"Rebuild the knowledge graph from policy documents (Admin only)\"\"\"\n    try:\n        result = kg_builder.rebuild_graph(db)\n        return {\n            \"success\": True,\n            \"message\": \"Knowledge graph rebuilt successfully\",\n            **result\n        }\n    except Exception as e:\n        raise HTTPException(\n            status_code=500,\n            detail=f\"Failed to rebuild knowledge graph: {str(e)}\"\n        )\n\n\n@router.post(\"/relationships\")\nasync def add_manual_relationship(\n    source_concept: str,\n    target_concept: str,\n    relationship_type: str,\n    weight: float = 1.0,\n    description: str = \"\",\n    current_user: User = Depends(require_role(UserRole.ADMIN)),\n    db: Session = Depends(get_db)\n):\n    \"\"\"Manually add a relationship between two concepts (Admin only)\"\"\"\n    \n    success = kg_builder.add_manual_relationship(\n        db=db,\n        source_concept=source_concept,\n        target_concept=target_concept,\n        relationship_type=relationship_type,\n        weight=weight,\n        description=description\n    )\n    \n    if success:\n        return {\n            \"success\": True,\n            \"message\": f\"Added {relationship_type} relationship: {source_concept} → {target_concept}\"\n        }\n    else:\n        raise HTTPException(\n            status_code=400,\n            detail=\"Failed to add relationship. Check that both concepts exist.\"\n        )\n\n\n@router.post(\"/test-kg-rag\")\nasync def test_kg_enhanced_rag(\n    query: str,\n    enable_kg: bool = True,\n    max_hops: int = 2,\n    k: int = 5,\n    current_user: User = Depends(get_current_user),\n    db: Session = Depends(get_db)\n):\n    \"\"\"Test the KG-Enhanced RAG system with a query\"\"\"\n    \n    try:\n        enhanced_citations, graph_hops, metadata = policy_retriever.kg_enhanced_retrieve(\n            query=query,\n            k=k,\n            enable_kg=enable_kg,\n            max_graph_hops=max_hops,\n            db=db\n        )\n        \n        # Generate explanation\n        explanation = policy_retriever.explain_retrieval_reasoning(\n            query=query,\n            enhanced_citations=enhanced_citations,\n            graph_hops=graph_hops,\n            metadata=metadata\n        )\n        \n        return {\n            \"query\": query,\n            \"enhanced_citations\": enhanced_citations,\n            \"graph_hops\": graph_hops,\n            \"metadata\": metadata,\n            \"explanation\": explanation\n        }\n        \n    except Exception as e:\n        raise HTTPException(\n            status_code=500,\n            detail=f\"KG-Enhanced RAG test failed: {str(e)}\"\n        )\n\n\n@router.get(\"/debug/extractions/{chunk_id}\")\nasync def get_concept_extractions_for_chunk(\n    chunk_id: int,\n    current_user: User = Depends(require_role(UserRole.ADMIN)),\n    db: Session = Depends(get_db)\n):\n    \"\"\"Get all concept extractions for a specific chunk (Admin only)\"\"\"\n    \n    from app.models.models import PolicyConceptExtraction\n    \n    extractions = db.query(PolicyConceptExtraction).filter(\n        PolicyConceptExtraction.chunk_id == chunk_id\n    ).all()\n    \n    return [\n        PolicyConceptExtractionResponse(\n            chunk_id=ext.chunk_id,\n            concept_id=ext.concept_id,\n            concept_name=ext.concept.name,\n            confidence_score=ext.confidence_score,\n            context_window=ext.context_window,\n            extraction_method=ext.extraction_method,\n            created_at=ext.created_at\n        )\n        for ext in extractions\n    ]\n\n\ndef _get_node_color(concept_type: str) -> str:\n    \"\"\"Get visualization color for concept type\"\"\"\n    color_map = {\n        \"technology\": \"#4ECDC4\",  # Teal\n        \"security\": \"#FF6B6B\",    # Red  \n        \"policy\": \"#45B7D1\",      # Blue\n        \"procedure\": \"#96CEB4\",   # Green\n        \"requirement\": \"#FFEAA7\" # Yellow\n    }\n    return color_map.get(concept_type, \"#DDA0DD\")  # Default: Plum\n\n\ndef _get_edge_color(relationship_type: str) -> str:\n    \"\"\"Get visualization color for relationship type\"\"\"\n    color_map = {\n        \"requires\": \"#E74C3C\",     # Red\n        \"depends_on\": \"#F39C12\",  # Orange\n        \"overrides\": \"#8E44AD\",   # Purple\n        \"related_to\": \"#3498DB\",  # Blue\n        \"enables\": \"#27AE60\",     # Green\n        \"affects\": \"#F1C40F\",     # Yellow\n        \"protects_from\": \"#E67E22\" # Dark Orange\n    }\n    return color_map.get(relationship_type, \"#95A5A6\")  # Default: Gray\n"
