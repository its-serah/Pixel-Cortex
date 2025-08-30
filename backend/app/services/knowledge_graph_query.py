"""
Knowledge Graph Query Service

This service traverses the knowledge graph to find related concepts and retrieve
connected policy information based on initial semantic search results.
"""

import json
import networkx as nx
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.models.models import (
    KnowledgeGraphConcept, PolicyChunk, PolicyDocument,
    PolicyConceptExtraction
)
from app.models.schemas import (
    GraphHop, KGEnhancedPolicyCitation, PolicyCitation,
    KnowledgeGraphQuery
)


class KnowledgeGraphQueryService:
    def __init__(self):
        """Initialize the Knowledge Graph Query Service"""
        self.max_depth = 3  # Maximum graph traversal depth
        self.min_relationship_weight = 0.3  # Minimum weight for considering relationships
    
    def query_graph_for_related_policies(
        self,
        db: Session,
        initial_concepts: List[str],
        semantic_citations: List[PolicyCitation],
        max_hops: int = 2,
        min_weight: float = 0.5
    ) -> Tuple[List[KGEnhancedPolicyCitation], List[GraphHop], Dict[str, Any]]:
        """
        Query the knowledge graph to find related concepts and retrieve connected policies
        
        Args:
            db: Database session
            initial_concepts: List of concept names found in semantic search
            semantic_citations: Original semantic search results
            max_hops: Maximum number of graph hops to traverse
            min_weight: Minimum relationship weight to follow
            
        Returns:
            Tuple of (enhanced_citations, graph_hops, query_metadata)
        """
        start_time = datetime.now()
        
        # Convert initial concepts to concept IDs
        initial_concept_ids = []
        for concept_name in initial_concepts:
            concept = self._get_concept_by_name_or_alias(db, concept_name)
            if concept:
                initial_concept_ids.append(concept.id)
        
        # Traverse the graph to find related concepts
        visited_concepts = set(initial_concept_ids)
        graph_hops = []
        concepts_to_explore = [(cid, 0) for cid in initial_concept_ids]  # (concept_id, depth)
        
        while concepts_to_explore and len(graph_hops) < max_hops * len(initial_concept_ids):
            current_concept_id, current_depth = concepts_to_explore.pop(0)
            
            if current_depth >= max_hops:
                continue
            
            # Find related concepts
            related_concepts = self._get_related_concepts(db, current_concept_id, min_weight)
            
            for related_id, relationship_type, weight in related_concepts:
                if related_id not in visited_concepts:
                    visited_concepts.add(related_id)
                    concepts_to_explore.append((related_id, current_depth + 1))
                    
                    # Record the graph hop
                    current_concept = db.query(KnowledgeGraphConcept).get(current_concept_id)
                    related_concept = db.query(KnowledgeGraphConcept).get(related_id)
                    
                    hop = GraphHop(
                        from_concept=current_concept.name,
                        to_concept=related_concept.name,
                        relationship_type=relationship_type,
                        hop_number=current_depth + 1,
                        reasoning=f"Found {relationship_type} relationship between {current_concept.name} and {related_concept.name} (weight: {weight:.2f})"
                    )
                    graph_hops.append(hop)
        
        # Get all policy chunks related to visited concepts
        related_chunk_ids = self._get_chunks_for_concepts(db, list(visited_concepts))
        
        # Enhance original semantic citations with graph information
        enhanced_citations = self._enhance_citations_with_graph_info(
            db, semantic_citations, graph_hops, visited_concepts
        )
        
        # Add new citations from graph-discovered concepts
        new_citations = self._get_citations_for_discovered_concepts(
            db, related_chunk_ids, semantic_citations, graph_hops
        )
        enhanced_citations.extend(new_citations)
        
        # Sort by combined score (semantic + graph boost)
        enhanced_citations.sort(key=lambda x: x.combined_score, reverse=True)
        
        end_time = datetime.now()
        processing_time = int((end_time - start_time).total_seconds() * 1000)
        
        query_metadata = {
            "processing_time_ms": processing_time,
            "graph_traversal_depth": max_hops,
            "total_concepts_visited": len(visited_concepts),
            "initial_concepts": len(initial_concept_ids),
            "concepts_discovered": len(visited_concepts) - len(initial_concept_ids),
            "graph_hops_taken": len(graph_hops)
        }
        
        return enhanced_citations, graph_hops, query_metadata
    
    def _get_concept_by_name_or_alias(self, db: Session, concept_name: str) -> Optional[KnowledgeGraphConcept]:
        """Find concept by name or alias"""
        concept_name_lower = concept_name.lower()
        
        # Try exact name match first
        concept = db.query(KnowledgeGraphConcept).filter(
            KnowledgeGraphConcept.name.ilike(f"%{concept_name}%")
        ).first()
        
        if concept:
            return concept
        
        # Search through aliases
        concepts = db.query(KnowledgeGraphConcept).all()
        for concept in concepts:
            if concept.aliases:
                for alias in concept.aliases:
                    if concept_name_lower in alias.lower() or alias.lower() in concept_name_lower:
                        return concept
        
        return None
    
    def _get_related_concepts(self, db: Session, concept_id: int, min_weight: float) -> List[Tuple[int, str, float]]:
        """Get concepts related to the given concept ID"""
        
        # Query outgoing relationships
        outgoing = db.execute(text("""
            SELECT target_concept_id, relationship_type, weight
            FROM concept_relationships
            WHERE source_concept_id = :concept_id AND weight >= :min_weight
            ORDER BY weight DESC
        """), {"concept_id": concept_id, "min_weight": min_weight}).fetchall()
        
        # Query incoming relationships  
        incoming = db.execute(text("""
            SELECT source_concept_id, relationship_type, weight
            FROM concept_relationships
            WHERE target_concept_id = :concept_id AND weight >= :min_weight
            ORDER BY weight DESC
        """), {"concept_id": concept_id, "min_weight": min_weight}).fetchall()
        
        # Combine and deduplicate
        related = []
        seen = set()
        
        for rel in outgoing:
            if rel[0] not in seen:
                related.append((rel[0], rel[1], rel[2]))
                seen.add(rel[0])
        
        for rel in incoming:
            if rel[0] not in seen:
                # Reverse the relationship type for incoming edges
                related.append((rel[0], f"inverse_{rel[1]}", rel[2]))
                seen.add(rel[0])
        
        return related
    
    def _get_chunks_for_concepts(self, db: Session, concept_ids: List[int]) -> Set[int]:
        """Get all policy chunk IDs that mention any of the given concepts"""
        
        if not concept_ids:
            return set()
        
        # Get chunk IDs from concept extractions
        chunk_ids = set()
        
        for concept_id in concept_ids:
            extractions = db.query(PolicyConceptExtraction).filter(
                PolicyConceptExtraction.concept_id == concept_id
            ).all()
            
            for extraction in extractions:
                chunk_ids.add(extraction.chunk_id)
        
        return chunk_ids
    
    def _enhance_citations_with_graph_info(
        self,
        db: Session,
        semantic_citations: List[PolicyCitation],
        graph_hops: List[GraphHop],
        visited_concepts: Set[int]
    ) -> List[KGEnhancedPolicyCitation]:
        """Enhance semantic search citations with graph reasoning information"""
        
        enhanced_citations = []
        
        for citation in semantic_citations:
            # Find graph path that led to this chunk (if any)
            graph_path = self._find_graph_path_to_chunk(db, citation.chunk_id, graph_hops)
            
            # Calculate graph boost score based on concept importance and relationships
            graph_boost = self._calculate_graph_boost(db, citation.chunk_id, visited_concepts)
            
            # Combine semantic and graph scores
            combined_score = citation.relevance_score + graph_boost * 0.3  # 30% boost from graph
            
            enhanced_citation = KGEnhancedPolicyCitation(
                document_id=citation.document_id,
                document_title=citation.document_title,
                chunk_id=citation.chunk_id,
                chunk_content=citation.chunk_content,
                relevance_score=citation.relevance_score,
                graph_path=graph_path,
                semantic_score=citation.relevance_score,
                graph_boost_score=graph_boost,
                combined_score=combined_score
            )
            enhanced_citations.append(enhanced_citation)
        
        return enhanced_citations
    
    def _get_citations_for_discovered_concepts(
        self,
        db: Session,
        related_chunk_ids: Set[int],
        original_citations: List[PolicyCitation],
        graph_hops: List[GraphHop]
    ) -> List[KGEnhancedPolicyCitation]:
        """Get citations for chunks discovered through graph traversal"""
        
        # Get chunk IDs that were already in semantic search results
        original_chunk_ids = {c.chunk_id for c in original_citations}
        
        # Find new chunks discovered through graph
        new_chunk_ids = related_chunk_ids - original_chunk_ids
        
        new_citations = []
        for chunk_id in new_chunk_ids:
            chunk = db.query(PolicyChunk).filter(PolicyChunk.id == chunk_id).first()
            if not chunk:
                continue
            
            # Find which graph path led to this chunk
            graph_path = self._find_graph_path_to_chunk(db, chunk_id, graph_hops)
            
            # Calculate relevance based on graph connections
            graph_boost = self._calculate_graph_boost(db, chunk_id, {})
            
            enhanced_citation = KGEnhancedPolicyCitation(
                document_id=chunk.document_id,
                document_title=chunk.document.title,
                chunk_id=chunk.id,
                chunk_content=chunk.content,
                relevance_score=0.0,  # No semantic score for graph-discovered chunks
                graph_path=graph_path,
                semantic_score=0.0,
                graph_boost_score=graph_boost,
                combined_score=graph_boost  # Pure graph-based relevance
            )
            new_citations.append(enhanced_citation)
        
        return new_citations
    
    def _find_graph_path_to_chunk(self, db: Session, chunk_id: int, graph_hops: List[GraphHop]) -> List[GraphHop]:
        """Find the graph traversal path that led to a specific chunk"""
        
        # Get concepts that mention this chunk
        extractions = db.query(PolicyConceptExtraction).filter(
            PolicyConceptExtraction.chunk_id == chunk_id
        ).all()
        
        chunk_concepts = {extraction.concept.name for extraction in extractions}
        
        # Find hops that led to any of these concepts
        relevant_hops = []
        for hop in graph_hops:
            if hop.to_concept in chunk_concepts:
                relevant_hops.append(hop)
        
        return relevant_hops
    
    def _calculate_graph_boost(self, db: Session, chunk_id: int, visited_concepts: Set[int]) -> float:
        """Calculate graph-based relevance boost for a chunk"""
        
        # Get concepts that mention this chunk
        extractions = db.query(PolicyConceptExtraction).filter(
            PolicyConceptExtraction.chunk_id == chunk_id
        ).all()
        
        if not extractions:
            return 0.0
        
        # Calculate boost based on concept importance and extraction confidence
        total_boost = 0.0
        for extraction in extractions:
            concept = extraction.concept
            importance_boost = concept.importance_score * extraction.confidence_score
            
            # Extra boost if this concept was discovered through graph traversal
            if concept.id in visited_concepts:
                importance_boost *= 1.5
            
            total_boost += importance_boost
        
        # Normalize to 0-1 range
        return min(1.0, total_boost / len(extractions))
    
    def find_shortest_path_between_concepts(
        self, 
        db: Session, 
        source_concept: str, 
        target_concept: str
    ) -> Optional[List[GraphHop]]:
        """Find the shortest path between two concepts in the graph"""
        
        # Get concept IDs
        source = self._get_concept_by_name_or_alias(db, source_concept)
        target = self._get_concept_by_name_or_alias(db, target_concept)
        
        if not source or not target:
            return None
        
        # Build NetworkX graph
        G = self._build_networkx_graph(db)
        
        try:
            # Find shortest path
            path = nx.shortest_path(G, source.id, target.id)
            
            # Convert path to GraphHop objects
            hops = []
            for i in range(len(path) - 1):
                current_id = path[i]
                next_id = path[i + 1]
                
                current_concept = db.query(KnowledgeGraphConcept).get(current_id)
                next_concept = db.query(KnowledgeGraphConcept).get(next_id)
                
                # Get relationship info
                edge_data = G.get_edge_data(current_id, next_id)
                relationship_type = edge_data.get('relationship_type', 'related_to')
                
                hop = GraphHop(
                    from_concept=current_concept.name,
                    to_concept=next_concept.name,
                    relationship_type=relationship_type,
                    hop_number=i + 1,
                    reasoning=f"Shortest path hop {i + 1}: {current_concept.name} → {next_concept.name}"
                )
                hops.append(hop)
            
            return hops
            
        except nx.NetworkXNoPath:
            return None
    
    def get_concept_neighborhood(
        self, 
        db: Session, 
        concept_name: str, 
        radius: int = 1
    ) -> Dict[str, List[str]]:
        """Get all concepts within a certain radius of the given concept"""
        
        concept = self._get_concept_by_name_or_alias(db, concept_name)
        if not concept:
            return {}
        
        G = self._build_networkx_graph(db)
        
        # Get neighborhood
        neighborhood = {}
        for depth in range(1, radius + 1):
            neighbors_at_depth = []
            
            # Get nodes at exactly this distance
            try:
                for node_id in G.nodes():
                    if node_id != concept.id:
                        try:
                            distance = nx.shortest_path_length(G, concept.id, node_id)
                            if distance == depth:
                                neighbor_concept = db.query(KnowledgeGraphConcept).get(node_id)
                                neighbors_at_depth.append(neighbor_concept.name)
                        except nx.NetworkXNoPath:
                            continue
            except:
                pass
            
            neighborhood[f"depth_{depth}"] = neighbors_at_depth
        
        return neighborhood
    
    def get_most_connected_concepts(self, db: Session, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get the most highly connected concepts in the graph"""
        
        G = self._build_networkx_graph(db)
        
        # Calculate various centrality measures
        try:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            pagerank = nx.pagerank(G)
        except:
            # Handle empty or disconnected graph
            return []
        
        # Combine centrality measures
        concept_scores = []
        for node_id in G.nodes():
            concept = db.query(KnowledgeGraphConcept).get(node_id)
            if concept:
                combined_score = (
                    degree_centrality.get(node_id, 0) * 0.4 +
                    betweenness_centrality.get(node_id, 0) * 0.3 +
                    pagerank.get(node_id, 0) * 0.3
                )
                
                concept_scores.append({
                    "concept_name": concept.name,
                    "concept_type": concept.concept_type,
                    "importance_score": concept.importance_score,
                    "combined_centrality": combined_score,
                    "degree": G.degree(node_id),
                    "chunk_count": len(concept.policy_chunks) if concept.policy_chunks else 0
                })
        
        # Sort by combined centrality score
        concept_scores.sort(key=lambda x: x["combined_centrality"], reverse=True)
        
        return concept_scores[:top_k]
    
    def _build_networkx_graph(self, db: Session) -> nx.DiGraph:
        """Build a NetworkX graph from the database"""
        
        G = nx.DiGraph()
        
        # Add nodes
        concepts = db.query(KnowledgeGraphConcept).all()
        for concept in concepts:
            G.add_node(concept.id, 
                      name=concept.name,
                      concept_type=concept.concept_type,
                      importance=concept.importance_score)
        
        # Add edges
        relationships = db.execute(text("""
            SELECT source_concept_id, target_concept_id, relationship_type, weight
            FROM concept_relationships
        """)).fetchall()
        
        for rel in relationships:
            G.add_edge(rel[0], rel[1], 
                      relationship_type=rel[2],
                      weight=rel[3])
        
        return G
    
    def analyze_query_coverage(
        self, 
        db: Session, 
        query_text: str, 
        retrieved_concepts: List[str]
    ) -> Dict[str, Any]:
        """Analyze how well the graph covered the user's query"""
        
        # Extract concepts mentioned in the original query
        query_concepts = []
        query_lower = query_text.lower()
        
        concepts = db.query(KnowledgeGraphConcept).all()
        for concept in concepts:
            if concept.aliases:
                for alias in concept.aliases:
                    if alias.lower() in query_lower:
                        query_concepts.append(concept.name)
                        break
        
        # Calculate coverage metrics
        concepts_in_query = set(query_concepts)
        concepts_retrieved = set(retrieved_concepts)
        
        coverage_ratio = len(concepts_in_query & concepts_retrieved) / max(len(concepts_in_query), 1)
        expansion_ratio = len(concepts_retrieved - concepts_in_query) / max(len(concepts_retrieved), 1)
        
        return {
            "query_concepts": list(concepts_in_query),
            "retrieved_concepts": list(concepts_retrieved),
            "coverage_ratio": coverage_ratio,  # How many query concepts were found
            "expansion_ratio": expansion_ratio,  # How many additional concepts were discovered
            "total_concepts_found": len(concepts_retrieved),
            "missing_concepts": list(concepts_in_query - concepts_retrieved)
        }
    
    def get_explanation_for_graph_traversal(
        self, 
        graph_hops: List[GraphHop], 
        initial_concepts: List[str]
    ) -> str:
        """Generate human-readable explanation of graph traversal reasoning"""
        
        if not graph_hops:
            return f"Used semantic search only. Found concepts: {', '.join(initial_concepts)}"
        
        explanation = f"Started with concepts: {', '.join(initial_concepts)}\n\n"
        explanation += "Graph traversal reasoning:\n"
        
        current_hop = 1
        for hop in graph_hops:
            explanation += f"{current_hop}. {hop.from_concept} → {hop.to_concept} "
            explanation += f"(via '{hop.relationship_type}' relationship)\n"
            explanation += f"   Reasoning: {hop.reasoning}\n"
            current_hop += 1
        
        discovered_concepts = list(set([hop.to_concept for hop in graph_hops]))
        explanation += f"\nDiscovered related concepts: {', '.join(discovered_concepts)}"
        
        return explanation
