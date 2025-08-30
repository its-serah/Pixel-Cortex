"""
Policy Knowledge Graph Builder Service

This service extracts key IT concepts (VPN, MFA, Remote Access, etc.) from policy documents
and builds a knowledge graph showing relationships between these concepts.
"""

import re
import json
import spacy
import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.models.models import (
    PolicyDocument, PolicyChunk, KnowledgeGraphConcept, 
    PolicyConceptExtraction, concept_relationships
)
from app.models.schemas import KnowledgeGraphConceptCreate
from app.services.local_llm_service import local_llm_service
from app.services.prompt_engineering_service import prompt_engineering_service, PromptTemplate


class PolicyKnowledgeGraphBuilder:
    def __init__(self):
        """Initialize the Knowledge Graph Builder with NLP models and IT concept patterns"""
        # Load spaCy model for NER and dependency parsing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Core IT concepts and their aliases
        self.it_concepts = {
            "VPN": {
                "aliases": ["vpn", "virtual private network", "remote access vpn", "site-to-site vpn"],
                "type": "technology",
                "importance": 0.9
            },
            "MFA": {
                "aliases": ["mfa", "multi-factor authentication", "two-factor authentication", "2fa", "multifactor"],
                "type": "security",
                "importance": 0.95
            },
            "Remote Access": {
                "aliases": ["remote access", "remote login", "remote desktop", "rdp", "ssh access"],
                "type": "technology",
                "importance": 0.8
            },
            "Firewall": {
                "aliases": ["firewall", "network firewall", "security firewall", "packet filtering"],
                "type": "security",
                "importance": 0.85
            },
            "Password Policy": {
                "aliases": ["password policy", "password requirements", "password complexity", "password rules"],
                "type": "policy",
                "importance": 0.8
            },
            "Software Installation": {
                "aliases": ["software installation", "software deployment", "app installation", "program install"],
                "type": "procedure",
                "importance": 0.7
            },
            "Hardware Replacement": {
                "aliases": ["hardware replacement", "hardware upgrade", "device replacement", "equipment change"],
                "type": "procedure",
                "importance": 0.7
            },
            "Security Incident": {
                "aliases": ["security incident", "security breach", "cyber attack", "malware infection"],
                "type": "security",
                "importance": 0.9
            },
            "Data Backup": {
                "aliases": ["data backup", "backup procedures", "data recovery", "backup policy"],
                "type": "procedure",
                "importance": 0.8
            },
            "Network Access": {
                "aliases": ["network access", "network permissions", "network connectivity", "lan access"],
                "type": "technology",
                "importance": 0.75
            }
        }
        
        # Relationship patterns to detect in policy text
        self.relationship_patterns = {
            "requires": [
                r"(\w+(?:\s+\w+)*)\s+requires?\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+must\s+have\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+needs?\s+(\w+(?:\s+\w+)*)",
            ],
            "depends_on": [
                r"(\w+(?:\s+\w+)*)\s+depends?\s+on\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+relies?\s+on\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+after\s+(\w+(?:\s+\w+)*)",
            ],
            "overrides": [
                r"(\w+(?:\s+\w+)*)\s+overrides?\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+supersedes?\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+takes?\s+precedence\s+over\s+(\w+(?:\s+\w+)*)",
            ],
            "related_to": [
                r"(\w+(?:\s+\w+)*)\s+(?:and|with|alongside)\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+integration\s+with\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+combined\s+with\s+(\w+(?:\s+\w+)*)",
            ]
        }
    
    def extract_concepts_from_chunk(self, chunk_content: str, use_llm: bool = True) -> List[Tuple[str, float, str]]:
        """
        Extract IT concepts from a policy chunk using LLM or fallback to regex
        
        Args:
            chunk_content: Text content to analyze
            use_llm: Whether to use LLM for extraction (default: True)
        
        Returns:
            List of (concept_name, confidence_score, context_window) tuples
        """
        if use_llm:
            try:
                # Use LLM for intelligent concept extraction
                return self._extract_concepts_with_llm(chunk_content)
            except Exception as e:
                print(f"LLM extraction failed, falling back to regex: {e}")
                # Fall back to regex-based extraction
                return self._extract_concepts_with_regex(chunk_content)
        else:
            return self._extract_concepts_with_regex(chunk_content)
    
    def _extract_concepts_with_llm(self, chunk_content: str) -> List[Tuple[str, float, str]]:
        """Extract concepts using LLM with structured prompts"""
        
        # Get known concepts for context
        known_concept_names = list(self.it_concepts.keys())
        
        # Use prompt engineering service for safe extraction
        result = prompt_engineering_service.generate_safe_response(
            PromptTemplate.CONCEPT_EXTRACTION,
            {
                "text": chunk_content,
                "known_concepts": ", ".join(known_concept_names)
            }
        )
        
        if result["success"] and result["response"]:
            concepts_data = result["response"]
            extracted_concepts = []
            
            for concept_data in concepts_data:
                concept_name = concept_data.get("concept")
                confidence = concept_data.get("confidence", 0.7)
                context = concept_data.get("context", "")[:100]  # Limit context length
                
                # Validate concept exists in our known concepts
                if concept_name in self.it_concepts:
                    extracted_concepts.append((concept_name, confidence, context))
            
            return extracted_concepts
        else:
            # LLM extraction failed, fall back to regex
            return self._extract_concepts_with_regex(chunk_content)
    
    def _extract_concepts_with_regex(self, chunk_content: str) -> List[Tuple[str, float, str]]:
        """Fallback regex-based concept extraction"""
        concepts_found = []
        chunk_lower = chunk_content.lower()
        
        for concept_name, concept_data in self.it_concepts.items():
            # Check for concept mentions using aliases
            for alias in concept_data["aliases"]:
                if alias in chunk_lower:
                    # Calculate confidence based on exact vs partial match
                    if alias == concept_name.lower():
                        confidence = 0.9
                    else:
                        confidence = 0.7
                    
                    # Extract context window around the match
                    match_pos = chunk_lower.find(alias)
                    start = max(0, match_pos - 50)
                    end = min(len(chunk_content), match_pos + len(alias) + 50)
                    context = chunk_content[start:end].strip()
                    
                    concepts_found.append((concept_name, confidence, context))
                    break  # Found concept, don't check other aliases
        
        return concepts_found
    
    def extract_relationships_from_chunk(self, chunk_content: str, concepts_in_chunk: List[str], use_llm: bool = True) -> List[Tuple[str, str, str, float]]:
        """
        Extract relationships between concepts mentioned in the same chunk
        
        Args:
            chunk_content: Text content to analyze
            concepts_in_chunk: List of concepts found in this chunk
            use_llm: Whether to use LLM for extraction (default: True)
        
        Returns:
            List of (source_concept, target_concept, relationship_type, confidence) tuples
        """
        if len(concepts_in_chunk) < 2:
            return []  # Need at least 2 concepts for relationships
        
        if use_llm:
            try:
                # Use LLM for intelligent relationship extraction
                return self._extract_relationships_with_llm(chunk_content, concepts_in_chunk)
            except Exception as e:
                print(f"LLM relationship extraction failed, falling back to regex: {e}")
                # Fall back to regex-based extraction
                return self._extract_relationships_with_regex(chunk_content, concepts_in_chunk)
        else:
            return self._extract_relationships_with_regex(chunk_content, concepts_in_chunk)
    
    def _extract_relationships_with_llm(self, chunk_content: str, concepts_in_chunk: List[str]) -> List[Tuple[str, str, str, float]]:
        """Extract relationships using LLM with structured prompts"""
        
        # Use prompt engineering service for safe relationship extraction
        result = prompt_engineering_service.generate_safe_response(
            PromptTemplate.RELATIONSHIP_EXTRACTION,
            {
                "text": chunk_content,
                "concepts": ", ".join(concepts_in_chunk)
            }
        )
        
        if result["success"] and result["response"]:
            relationships_data = result["response"]
            extracted_relationships = []
            
            for rel_data in relationships_data:
                source = rel_data.get("source")
                target = rel_data.get("target")
                relationship = rel_data.get("relationship")
                confidence = rel_data.get("confidence", 0.7)
                
                # Validate that both concepts are in our chunk and known
                if (source in concepts_in_chunk and target in concepts_in_chunk and 
                    source in self.it_concepts and target in self.it_concepts and
                    source != target):
                    extracted_relationships.append((source, target, relationship, confidence))
            
            return extracted_relationships
        else:
            # LLM extraction failed, fall back to regex
            return self._extract_relationships_with_regex(chunk_content, concepts_in_chunk)
    
    def _extract_relationships_with_regex(self, chunk_content: str, concepts_in_chunk: List[str]) -> List[Tuple[str, str, str, float]]:
        """Fallback regex-based relationship extraction"""
        relationships = []
        
        for rel_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, chunk_content, re.IGNORECASE)
                for match in matches:
                    source_text = match.group(1).strip()
                    target_text = match.group(2).strip()
                    
                    # Map extracted text to known concepts
                    source_concept = self._map_text_to_concept(source_text)
                    target_concept = self._map_text_to_concept(target_text)
                    
                    if source_concept and target_concept and source_concept != target_concept:
                        # Both concepts are in our chunk
                        if source_concept in concepts_in_chunk and target_concept in concepts_in_chunk:
                            confidence = 0.8  # High confidence for explicit relationships
                            relationships.append((source_concept, target_concept, rel_type, confidence))
        
        return relationships
    
    def _map_text_to_concept(self, text: str) -> Optional[str]:
        """Map extracted text to a known IT concept"""
        text_lower = text.lower().strip()
        
        for concept_name, concept_data in self.it_concepts.items():
            for alias in concept_data["aliases"]:
                if alias in text_lower or text_lower in alias:
                    return concept_name
        return None
    
    def build_knowledge_graph(self, db: Session) -> Dict[str, Any]:
        """
        Build the complete knowledge graph from all policy documents
        
        Returns:
            Dictionary with build statistics and graph info
        """
        start_time = datetime.now()
        
        # Clear existing graph data
        db.execute(text("DELETE FROM policy_concept_extractions"))
        db.execute(text("DELETE FROM concept_relationships"))
        db.execute(text("DELETE FROM kg_concepts"))
        db.commit()
        
        # Create concept nodes
        concepts_created = {}
        for concept_name, concept_data in self.it_concepts.items():
            concept = KnowledgeGraphConcept(
                name=concept_name,
                concept_type=concept_data["type"],
                description=f"IT concept: {concept_name}",
                aliases=concept_data["aliases"],
                importance_score=concept_data["importance"],
                policy_chunks=[]
            )
            db.add(concept)
            db.flush()  # Get the ID
            concepts_created[concept_name] = concept.id
        
        db.commit()
        
        # Process all policy chunks
        chunks = db.query(PolicyChunk).join(PolicyDocument).all()
        total_extractions = 0
        total_relationships = 0
        
        for chunk in chunks:
            # Extract concepts from this chunk
            chunk_concepts = self.extract_concepts_from_chunk(chunk.content)
            concepts_in_chunk = []
            
            for concept_name, confidence, context in chunk_concepts:
                concept_id = concepts_created[concept_name]
                
                # Record the extraction
                extraction = PolicyConceptExtraction(
                    chunk_id=chunk.id,
                    concept_id=concept_id,
                    confidence_score=confidence,
                    context_window=context,
                    extraction_method="regex_pattern"
                )
                db.add(extraction)
                
                # Update concept's policy chunks list
                concept = db.query(KnowledgeGraphConcept).get(concept_id)
                if concept.policy_chunks is None:
                    concept.policy_chunks = []
                if chunk.id not in concept.policy_chunks:
                    concept.policy_chunks.append(chunk.id)
                
                concepts_in_chunk.append(concept_name)
                total_extractions += 1
            
            # Extract relationships between concepts in this chunk
            relationships = self.extract_relationships_from_chunk(chunk.content, concepts_in_chunk)
            
            for source_concept, target_concept, rel_type, confidence in relationships:
                source_id = concepts_created[source_concept]
                target_id = concepts_created[target_concept]
                
                # Check if relationship already exists
                existing = db.execute(text("""
                    SELECT COUNT(*) FROM concept_relationships 
                    WHERE source_concept_id = :source AND target_concept_id = :target 
                    AND relationship_type = :rel_type
                """), {
                    "source": source_id, 
                    "target": target_id, 
                    "rel_type": rel_type
                }).scalar()
                
                if existing == 0:
                    # Insert new relationship
                    db.execute(text("""
                        INSERT INTO concept_relationships 
                        (source_concept_id, target_concept_id, relationship_type, weight, metadata)
                        VALUES (:source, :target, :rel_type, :weight, :metadata)
                    """), {
                        "source": source_id,
                        "target": target_id,
                        "rel_type": rel_type,
                        "weight": confidence,
                        "metadata": json.dumps({"chunk_id": chunk.id, "auto_extracted": True})
                    })
                    total_relationships += 1
        
        # Add some predefined important relationships for IT support
        self._add_predefined_relationships(db, concepts_created)
        
        db.commit()
        
        end_time = datetime.now()
        build_time = int((end_time - start_time).total_seconds() * 1000)
        
        return {
            "concepts_created": len(concepts_created),
            "total_extractions": total_extractions,
            "total_relationships": total_relationships,
            "build_time_ms": build_time,
            "timestamp": end_time
        }
    
    def _add_predefined_relationships(self, db: Session, concepts_created: Dict[str, int]):
        """Add predefined important relationships for IT support scenarios"""
        
        predefined_relationships = [
            # VPN relationships
            ("VPN", "MFA", "requires", 0.9, "VPN access typically requires MFA"),
            ("VPN", "Remote Access", "enables", 0.8, "VPN enables secure remote access"),
            ("VPN", "Network Access", "depends_on", 0.7, "VPN depends on network connectivity"),
            
            # MFA relationships  
            ("Remote Access", "MFA", "requires", 0.85, "Remote access requires MFA verification"),
            ("Security Incident", "MFA", "related_to", 0.7, "Security incidents often involve MFA bypass"),
            
            # Software and security
            ("Software Installation", "Security Incident", "can_cause", 0.6, "Unauthorized software can cause security incidents"),
            ("Software Installation", "Password Policy", "must_follow", 0.5, "Software installation accounts must follow password policy"),
            
            # Hardware and access
            ("Hardware Replacement", "Network Access", "affects", 0.6, "Hardware changes can affect network access"),
            ("Hardware Replacement", "Remote Access", "affects", 0.5, "Hardware changes can affect remote access"),
            
            # Data and security
            ("Data Backup", "Security Incident", "protects_from", 0.8, "Data backups protect against security incidents"),
            ("Data Backup", "Hardware Replacement", "required_before", 0.7, "Data backup required before hardware replacement"),
        ]
        
        for source, target, rel_type, weight, description in predefined_relationships:
            if source in concepts_created and target in concepts_created:
                source_id = concepts_created[source]
                target_id = concepts_created[target]
                
                # Check if relationship already exists
                existing = db.execute(text("""
                    SELECT COUNT(*) FROM concept_relationships 
                    WHERE source_concept_id = :source AND target_concept_id = :target 
                    AND relationship_type = :rel_type
                """), {
                    "source": source_id, 
                    "target": target_id, 
                    "rel_type": rel_type
                }).scalar()
                
                if existing == 0:
                    db.execute(text("""
                        INSERT INTO concept_relationships 
                        (source_concept_id, target_concept_id, relationship_type, weight, metadata)
                        VALUES (:source, :target, :rel_type, :weight, :metadata)
                    """), {
                        "source": source_id,
                        "target": target_id,
                        "rel_type": rel_type,
                        "weight": weight,
                        "metadata": json.dumps({"predefined": True, "description": description})
                    })
    
    def get_concept_by_name(self, db: Session, concept_name: str) -> Optional[KnowledgeGraphConcept]:
        """Get concept by name or alias"""
        # First try exact name match
        concept = db.query(KnowledgeGraphConcept).filter(
            KnowledgeGraphConcept.name == concept_name
        ).first()
        
        if concept:
            return concept
        
        # Try alias match
        concepts = db.query(KnowledgeGraphConcept).all()
        concept_name_lower = concept_name.lower()
        
        for concept in concepts:
            if concept.aliases:
                for alias in concept.aliases:
                    if alias.lower() == concept_name_lower:
                        return concept
        
        return None
    
    def find_concepts_in_text(self, text: str) -> List[Tuple[str, float]]:
        """
        Find IT concepts mentioned in arbitrary text (e.g., ticket description)
        
        Returns:
            List of (concept_name, confidence_score) tuples
        """
        concepts_found = []
        text_lower = text.lower()
        
        for concept_name, concept_data in self.it_concepts.items():
            for alias in concept_data["aliases"]:
                if alias in text_lower:
                    # Calculate confidence based on context and alias match quality
                    confidence = 0.8 if alias == concept_name.lower() else 0.6
                    
                    # Boost confidence if concept appears multiple times
                    occurrences = text_lower.count(alias)
                    confidence = min(0.95, confidence + (occurrences - 1) * 0.1)
                    
                    concepts_found.append((concept_name, confidence))
                    break  # Found concept, don't check other aliases
        
        return concepts_found
    
    def add_manual_relationship(
        self, 
        db: Session, 
        source_concept: str, 
        target_concept: str, 
        relationship_type: str,
        weight: float = 1.0,
        description: str = ""
    ) -> bool:
        """Manually add a relationship between two concepts"""
        
        source = self.get_concept_by_name(db, source_concept)
        target = self.get_concept_by_name(db, target_concept)
        
        if not source or not target:
            return False
        
        # Check if relationship already exists
        existing = db.execute(text("""
            SELECT COUNT(*) FROM concept_relationships 
            WHERE source_concept_id = :source AND target_concept_id = :target 
            AND relationship_type = :rel_type
        """), {
            "source": source.id, 
            "target": target.id, 
            "rel_type": relationship_type
        }).scalar()
        
        if existing == 0:
            db.execute(text("""
                INSERT INTO concept_relationships 
                (source_concept_id, target_concept_id, relationship_type, weight, metadata)
                VALUES (:source, :target, :rel_type, :weight, :metadata)
            """), {
                "source": source.id,
                "target": target.id,
                "rel_type": relationship_type,
                "weight": weight,
                "metadata": json.dumps({"manual": True, "description": description})
            })
            db.commit()
            return True
        
        return False
    
    def get_graph_statistics(self, db: Session) -> Dict[str, Any]:
        """Get statistics about the current knowledge graph"""
        
        concepts_count = db.query(KnowledgeGraphConcept).count()
        relationships_count = db.execute(text("SELECT COUNT(*) FROM concept_relationships")).scalar()
        extractions_count = db.query(PolicyConceptExtraction).count()
        
        # Get relationship type distribution
        rel_types = db.execute(text("""
            SELECT relationship_type, COUNT(*) 
            FROM concept_relationships 
            GROUP BY relationship_type
        """)).fetchall()
        
        return {
            "total_concepts": concepts_count,
            "total_relationships": relationships_count,
            "total_extractions": extractions_count,
            "relationship_types": dict(rel_types),
            "avg_relationships_per_concept": relationships_count / max(concepts_count, 1)
        }
    
    def rebuild_graph(self, db: Session) -> Dict[str, Any]:
        """Completely rebuild the knowledge graph from scratch"""
        print("Rebuilding knowledge graph from policy documents...")
        return self.build_knowledge_graph(db)
    
    def export_graph_to_networkx(self, db: Session) -> nx.DiGraph:
        """Export the knowledge graph as a NetworkX directed graph for analysis"""
        
        G = nx.DiGraph()
        
        # Add nodes
        concepts = db.query(KnowledgeGraphConcept).all()
        for concept in concepts:
            G.add_node(concept.id, 
                      name=concept.name,
                      concept_type=concept.concept_type,
                      importance=concept.importance_score,
                      chunk_count=len(concept.policy_chunks) if concept.policy_chunks else 0)
        
        # Add edges
        relationships = db.execute(text("""
            SELECT source_concept_id, target_concept_id, relationship_type, weight, metadata
            FROM concept_relationships
        """)).fetchall()
        
        for rel in relationships:
            G.add_edge(rel[0], rel[1], 
                      relationship_type=rel[2],
                      weight=rel[3],
                      metadata=json.loads(rel[4]) if rel[4] else {})
        
        return G
