import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from app.models.models import TicketCategory, TicketPriority
from app.models.schemas import (
    ExplanationObject, ReasoningStep, TelemetryData, 
    PolicyCitation, KGEnhancedPolicyCitation, GraphHop,
    KGEnhancedExplanationObject
)
from app.services.policy_retriever import PolicyRetriever
from sqlalchemy.orm import Session

class TriageService:
    def __init__(self):
        # Initialize policy retriever for KG-Enhanced explanations
        self.policy_retriever = PolicyRetriever()
        # Rule-based patterns for categorization
        self.category_rules = {
            TicketCategory.HARDWARE: {
                "patterns": [
                    r"(?i)(computer|laptop|desktop|monitor|keyboard|mouse|printer|scanner)",
                    r"(?i)(hardware|device|equipment|peripheral)",
                    r"(?i)(not working|broken|damaged|won't turn on|black screen)",
                    r"(?i)(cpu|memory|ram|hard drive|ssd|graphics card|motherboard)"
                ],
                "keywords": ["hardware", "device", "computer", "laptop", "printer", "monitor", "broken", "damaged"]
            },
            TicketCategory.SOFTWARE: {
                "patterns": [
                    r"(?i)(software|application|program|app)",
                    r"(?i)(install|uninstall|update|upgrade|patch)",
                    r"(?i)(windows|linux|macos|office|excel|word|outlook)",
                    r"(?i)(crashes|freezes|error message|blue screen|won't start)"
                ],
                "keywords": ["software", "application", "install", "update", "program", "crashes", "error"]
            },
            TicketCategory.NETWORK: {
                "patterns": [
                    r"(?i)(network|internet|wifi|ethernet|connection)",
                    r"(?i)(can't connect|no internet|slow connection|vpn)",
                    r"(?i)(router|switch|firewall|dns|ip address)",
                    r"(?i)(email not working|can't access website)"
                ],
                "keywords": ["network", "internet", "wifi", "connection", "router", "vpn", "dns"]
            },
            TicketCategory.ACCESS: {
                "patterns": [
                    r"(?i)(password|login|access|permission|locked out)",
                    r"(?i)(can't log in|forgot password|account disabled)",
                    r"(?i)(authentication|authorization|user account)",
                    r"(?i)(active directory|ldap|sso|single sign)"
                ],
                "keywords": ["password", "login", "access", "permission", "account", "authentication"]
            },
            TicketCategory.SECURITY: {
                "patterns": [
                    r"(?i)(virus|malware|phishing|security|breach)",
                    r"(?i)(suspicious|hacked|infected|spam)",
                    r"(?i)(antivirus|firewall|encryption|backup)",
                    r"(?i)(data loss|ransomware|trojan|suspicious email)"
                ],
                "keywords": ["virus", "malware", "security", "suspicious", "hacked", "antivirus", "breach"]
            }
        }
        
        # Priority rules based on keywords and urgency indicators
        self.priority_rules = {
            TicketPriority.CRITICAL: {
                "patterns": [
                    r"(?i)(critical|urgent|emergency|down|outage)",
                    r"(?i)(all users affected|entire office|production down)",
                    r"(?i)(security breach|data loss|ransomware|hacked)"
                ],
                "keywords": ["critical", "urgent", "emergency", "down", "outage", "breach", "hacked"]
            },
            TicketPriority.HIGH: {
                "patterns": [
                    r"(?i)(high priority|important|asap|blocking work)",
                    r"(?i)(multiple users|department|team affected)",
                    r"(?i)(deadline|meeting|presentation)"
                ],
                "keywords": ["high", "important", "asap", "blocking", "deadline", "meeting"]
            },
            TicketPriority.LOW: {
                "patterns": [
                    r"(?i)(low priority|when you have time|minor issue)",
                    r"(?i)(cosmetic|enhancement|nice to have)",
                    r"(?i)(training|question|how to)"
                ],
                "keywords": ["low", "minor", "cosmetic", "enhancement", "training", "question"]
            }
        }

    def triage_ticket(self, title: str, description: str) -> Tuple[TicketCategory, TicketPriority, float, ExplanationObject]:
        """
        Perform rule-based triage on a ticket.
        Returns: (category, priority, confidence, explanation)
        """
        start_time = datetime.now()
        
        # Combine title and description for analysis
        combined_text = f"{title} {description}".lower()
        
        # Category scoring
        category_scores = {}
        category_matches = {}
        
        for category, rules in self.category_rules.items():
            score = 0
            matches = []
            
            # Pattern matching
            for pattern in rules["patterns"]:
                pattern_matches = re.findall(pattern, combined_text)
                if pattern_matches:
                    score += len(pattern_matches) * 2  # Weight patterns higher
                    matches.extend(pattern_matches)
            
            # Keyword matching
            for keyword in rules["keywords"]:
                if keyword.lower() in combined_text:
                    score += 1
                    matches.append(keyword)
            
            category_scores[category] = score
            category_matches[category] = list(set(matches))  # Remove duplicates
        
        # Determine best category
        best_category = max(category_scores, key=category_scores.get) if any(category_scores.values()) else TicketCategory.OTHER
        category_confidence = category_scores[best_category] / max(sum(category_scores.values()), 1)
        
        # Priority scoring
        priority_scores = {}
        priority_matches = {}
        
        for priority, rules in self.priority_rules.items():
            score = 0
            matches = []
            
            # Pattern matching
            for pattern in rules["patterns"]:
                pattern_matches = re.findall(pattern, combined_text)
                if pattern_matches:
                    score += len(pattern_matches) * 2
                    matches.extend(pattern_matches)
            
            # Keyword matching
            for keyword in rules["keywords"]:
                if keyword.lower() in combined_text:
                    score += 1
                    matches.append(keyword)
            
            priority_scores[priority] = score
            priority_matches[priority] = list(set(matches))
        
        # Determine priority (default to MEDIUM if no matches)
        if any(priority_scores.values()):
            best_priority = max(priority_scores, key=priority_scores.get)
            priority_confidence = priority_scores[best_priority] / max(sum(priority_scores.values()), 1)
        else:
            best_priority = TicketPriority.MEDIUM
            priority_confidence = 0.5  # Default confidence for medium priority
        
        # Overall confidence (weighted average)
        overall_confidence = (category_confidence * 0.6 + priority_confidence * 0.4)
        
        # Build reasoning trace
        reasoning_trace = []
        step = 1
        
        reasoning_trace.append(ReasoningStep(
            step=step,
            action="text_analysis",
            rationale=f"Analyzed combined text: '{title[:50]}...' for pattern and keyword matches",
            confidence=0.9,
            policy_refs=[]
        ))
        step += 1
        
        if category_matches[best_category]:
            reasoning_trace.append(ReasoningStep(
                step=step,
                action="category_classification",
                rationale=f"Classified as {best_category.value} based on matches: {category_matches[best_category][:3]}",
                confidence=category_confidence,
                policy_refs=[]
            ))
            step += 1
        
        if priority_matches.get(best_priority):
            reasoning_trace.append(ReasoningStep(
                step=step,
                action="priority_assignment",
                rationale=f"Assigned {best_priority.value} priority based on matches: {priority_matches[best_priority][:3]}",
                confidence=priority_confidence,
                policy_refs=[]
            ))
        else:
            reasoning_trace.append(ReasoningStep(
                step=step,
                action="priority_assignment",
                rationale=f"Assigned default {best_priority.value} priority (no urgency indicators found)",
                confidence=priority_confidence,
                policy_refs=[]
            ))
        
        end_time = datetime.now()
        latency_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Build explanation object
        explanation = ExplanationObject(
            answer=f"Ticket categorized as {best_category.value} with {best_priority.value} priority",
            decision=f"category={best_category.value}, priority={best_priority.value}",
            confidence=overall_confidence,
            reasoning_trace=reasoning_trace,
            policy_citations=[],  # No policy citations for rule-based triage
            missing_info=[],
            alternatives_considered=[],
            counterfactuals=[],
            telemetry=TelemetryData(
                latency_ms=latency_ms,
                retrieval_k=0,
                triage_time_ms=latency_ms,
                planning_time_ms=0,
                total_chunks_considered=0
            ),
            timestamp=datetime.now(),
            model_version="1.0.0"
        )
        
        return best_category, best_priority, overall_confidence, explanation

    def get_category_keywords(self, category: TicketCategory) -> List[str]:
        """Get keywords for a specific category"""
        return self.category_rules.get(category, {}).get("keywords", [])
    
    def get_priority_keywords(self, priority: TicketPriority) -> List[str]:
        """Get keywords for a specific priority"""
        return self.priority_rules.get(priority, {}).get("keywords", [])
    
    def kg_enhanced_triage(
        self, 
        title: str, 
        description: str, 
        db: Session
    ) -> Tuple[TicketCategory, TicketPriority, float, KGEnhancedExplanationObject]:
        """
        Perform KG-Enhanced triage that includes knowledge graph reasoning
        
        This method:
        1. Performs rule-based triage (same as triage_ticket)
        2. Uses KG-Enhanced RAG to find relevant policies
        3. Includes graph reasoning in the explanation
        4. Provides enhanced policy citations with graph context
        
        Returns: (category, priority, confidence, kg_enhanced_explanation)
        """
        start_time = datetime.now()
        
        # Step 1: Perform standard rule-based triage
        category, priority, confidence, base_explanation = self.triage_ticket(title, description)
        
        # Step 2: Use KG-Enhanced RAG to find relevant policies
        query = f"{title} {description}"
        enhanced_citations, graph_hops, kg_metadata = self.policy_retriever.kg_enhanced_retrieve(
            query=query,
            k=5,
            enable_kg=True,
            db=db
        )
        
        # Step 3: Add knowledge graph reasoning to the explanation
        enhanced_reasoning_trace = list(base_explanation.reasoning_trace)
        
        # Add KG reasoning steps
        if enhanced_citations or graph_hops:
            step_num = len(enhanced_reasoning_trace) + 1
            
            # Add concept identification step
            concepts_found = kg_metadata.get('initial_concepts_found', [])
            if concepts_found:
                enhanced_reasoning_trace.append(ReasoningStep(
                    step=step_num,
                    action="concept_identification",
                    rationale=f"Identified IT concepts in ticket: {', '.join(concepts_found)}",
                    confidence=0.8,
                    policy_refs=[]
                ))
                step_num += 1
            
            # Add graph traversal step
            if graph_hops:
                hop_summary = f"Traversed knowledge graph: {len(graph_hops)} connections found"
                enhanced_reasoning_trace.append(ReasoningStep(
                    step=step_num,
                    action="graph_traversal",
                    rationale=hop_summary,
                    confidence=0.85,
                    policy_refs=[c.chunk_id for c in enhanced_citations[:3]]
                ))
                step_num += 1
            
            # Add policy retrieval step
            if enhanced_citations:
                semantic_count = sum(1 for c in enhanced_citations if c.semantic_score > 0)
                graph_count = len(enhanced_citations) - semantic_count
                enhanced_reasoning_trace.append(ReasoningStep(
                    step=step_num,
                    action="policy_retrieval",
                    rationale=f"Retrieved {semantic_count} semantic + {graph_count} graph-based policy citations",
                    confidence=0.9,
                    policy_refs=[c.chunk_id for c in enhanced_citations]
                ))
        
        # Step 4: Build enhanced explanation object
        end_time = datetime.now()
        total_latency = int((end_time - start_time).total_seconds() * 1000)
        
        # Get concepts discovered through graph
        discovered_concepts = list(set([hop.to_concept for hop in graph_hops]))
        
        # Calculate graph coverage score
        graph_coverage = self._calculate_graph_coverage_score(enhanced_citations, graph_hops)
        
        # Convert regular policy citations to KG-enhanced citations
        kg_policy_citations = []
        for citation in base_explanation.policy_citations:
            kg_citation = KGEnhancedPolicyCitation(
                document_id=citation.document_id,
                document_title=citation.document_title,
                chunk_id=citation.chunk_id,
                chunk_content=citation.chunk_content,
                relevance_score=citation.relevance_score,
                graph_path=[],
                semantic_score=citation.relevance_score,
                graph_boost_score=0.0,
                combined_score=citation.relevance_score
            )
            kg_policy_citations.append(kg_citation)
        
        # Add the enhanced citations from KG-RAG
        kg_policy_citations.extend(enhanced_citations)
        
        enhanced_explanation = KGEnhancedExplanationObject(
            # Base explanation fields
            answer=base_explanation.answer,
            decision=base_explanation.decision,
            confidence=confidence,
            reasoning_trace=enhanced_reasoning_trace,
            policy_citations=base_explanation.policy_citations,
            missing_info=base_explanation.missing_info,
            alternatives_considered=base_explanation.alternatives_considered,
            counterfactuals=base_explanation.counterfactuals,
            telemetry=TelemetryData(
                latency_ms=total_latency,
                retrieval_k=len(enhanced_citations),
                triage_time_ms=base_explanation.telemetry.triage_time_ms,
                planning_time_ms=kg_metadata.get('processing_time_ms', 0),
                total_chunks_considered=len(enhanced_citations)
            ),
            timestamp=datetime.now(),
            model_version="1.1.0",  # Updated version for KG-Enhanced
            
            # KG-Enhanced fields
            kg_policy_citations=kg_policy_citations,
            graph_reasoning=graph_hops,
            concepts_discovered=discovered_concepts,
            graph_coverage_score=graph_coverage
        )
        
        return category, priority, confidence, enhanced_explanation
    
    def _calculate_graph_coverage_score(self, citations: List[KGEnhancedPolicyCitation], graph_hops: List[GraphHop]) -> float:
        """Calculate how well the knowledge graph covered the query"""
        if not citations:
            return 0.0
        
        # Count citations that were enhanced by graph reasoning
        graph_enhanced_count = sum(1 for c in citations if c.graph_boost_score > 0 or c.graph_path)
        
        # Base coverage from graph enhancement ratio
        enhancement_ratio = graph_enhanced_count / len(citations)
        
        # Boost based on number of meaningful graph hops
        hop_bonus = min(0.3, len(graph_hops) * 0.1)
        
        return min(1.0, enhancement_ratio + hop_bonus)
