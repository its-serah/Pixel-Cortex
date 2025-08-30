from typing import List, Dict, Tuple, Any
from datetime import datetime
from enum import Enum
from sqlalchemy.orm import Session
from app.models.models import TicketCategory, TicketPriority
from app.models.schemas import (
    ExplanationObject, ReasoningStep, PolicyCitation, TelemetryData,
    KGEnhancedExplanationObject, KGEnhancedPolicyCitation, GraphHop
)
from app.services.policy_retriever import PolicyRetriever

class DecisionType(str, Enum):
    ALLOWED = "allowed"
    DENIED = "denied"
    NEEDS_APPROVAL = "needs_approval"

class DecisionService:
    def __init__(self):
        self.policy_retriever = PolicyRetriever()
        
        # Decision rules based on policy compliance
        self.decision_rules = {
            # Automatic approval rules
            "auto_allow": {
                "patterns": [
                    r"(?i)(password reset|unlock account|software update)",
                    r"(?i)(standard procedure|routine maintenance|approved software)",
                    r"(?i)(low priority|minor issue|cosmetic)"
                ],
                "categories": [TicketCategory.ACCESS, TicketCategory.SOFTWARE],
                "max_priority": TicketPriority.MEDIUM
            },
            
            # Automatic denial rules
            "auto_deny": {
                "patterns": [
                    r"(?i)(install unauthorized|personal software|gaming)",
                    r"(?i)(bypass security|disable firewall|admin rights)",
                    r"(?i)(external access|port forwarding|vpn exception)"
                ],
                "security_violations": True
            },
            
            # Approval required rules
            "needs_approval": {
                "patterns": [
                    r"(?i)(budget|expensive|hardware purchase|new equipment)",
                    r"(?i)(data recovery|system restoration|major change)",
                    r"(?i)(critical system|production server|database)"
                ],
                "high_cost": True,
                "critical_systems": True
            }
        }
    
    def make_decision(
        self, 
        title: str, 
        description: str, 
        category: TicketCategory,
        priority: TicketPriority,
        db: Session
    ) -> Tuple[DecisionType, ExplanationObject]:
        """
        Make policy-grounded decision (Allowed/Denied/Needs Approval)
        
        Returns: (decision, explanation)
        """
        start_time = datetime.now()
        
        # Search for relevant policy chunks
        search_query = f"{category.value} {title} {description}"
        policy_citations = self.policy_retriever.retrieve_relevant_chunks(
            search_query, k=3, db=db
        )
        
        # Analyze text for decision patterns
        combined_text = f"{title} {description}".lower()
        
        decision = DecisionType.ALLOWED  # Default
        reasoning_trace = []
        step = 1
        
        # Check for auto-denial patterns first (highest priority)
        deny_matches = []
        for pattern in self.decision_rules["auto_deny"]["patterns"]:
            import re
            if re.search(pattern, combined_text):
                deny_matches.append(pattern)
        
        if deny_matches:
            decision = DecisionType.DENIED
            reasoning_trace.append(ReasoningStep(
                step=step,
                action="security_violation_check",
                rationale=f"Denied due to security policy violations: {deny_matches[:2]}",
                confidence=0.95,
                policy_refs=[c.chunk_id for c in policy_citations if "security" in c.chunk_content.lower()]
            ))
            step += 1
        
        # Check for approval required patterns
        elif any(re.search(pattern, combined_text) for pattern in self.decision_rules["needs_approval"]["patterns"]):
            decision = DecisionType.NEEDS_APPROVAL
            reasoning_trace.append(ReasoningStep(
                step=step,
                action="approval_requirement_check",
                rationale=f"Requires approval due to high cost/impact or critical system involvement",
                confidence=0.9,
                policy_refs=[c.chunk_id for c in policy_citations]
            ))
            step += 1
        
        # Check priority-based rules
        elif priority == TicketPriority.CRITICAL:
            if category in [TicketCategory.SECURITY, TicketCategory.NETWORK]:
                decision = DecisionType.NEEDS_APPROVAL
                reasoning_trace.append(ReasoningStep(
                    step=step,
                    action="critical_priority_check",
                    rationale=f"Critical {category.value} issues require management approval",
                    confidence=0.95,
                    policy_refs=[]
                ))
                step += 1
        
        # Policy-grounded decision validation
        policy_decision = self._validate_against_policies(policy_citations, decision, category)
        if policy_decision != decision:
            reasoning_trace.append(ReasoningStep(
                step=step,
                action="policy_override",
                rationale=f"Decision changed from {decision.value} to {policy_decision.value} based on policy guidance",
                confidence=0.85,
                policy_refs=[c.chunk_id for c in policy_citations[:2]]
            ))
            decision = policy_decision
            step += 1
        
        # Add policy analysis step
        reasoning_trace.append(ReasoningStep(
            step=step,
            action="policy_analysis",
            rationale=f"Analyzed {len(policy_citations)} policy chunks for compliance and requirements",
            confidence=0.8,
            policy_refs=[c.chunk_id for c in policy_citations]
        ))
        
        end_time = datetime.now()
        decision_time = int((end_time - start_time).total_seconds() * 1000)
        
        # Build comprehensive explanation (STORED IN LOGS ONLY)
        explanation = ExplanationObject(
            answer=f"Decision: {decision.value.upper()} - {self._get_decision_rationale(decision, category, priority)}",
            decision=f"decision={decision.value}, category={category.value}, priority={priority.value}",
            confidence=self._calculate_decision_confidence(decision, policy_citations),
            reasoning_trace=reasoning_trace,
            policy_citations=policy_citations,
            missing_info=self._identify_missing_info_for_decision(title, description, category),
            alternatives_considered=self._generate_decision_alternatives(decision, category),
            counterfactuals=self._generate_decision_counterfactuals(decision, category, priority),
            telemetry=TelemetryData(
                latency_ms=decision_time,
                retrieval_k=len(policy_citations),
                triage_time_ms=0,
                planning_time_ms=decision_time,
                total_chunks_considered=len(policy_citations)
            ),
            timestamp=datetime.now(),
            model_version="1.0.0"
        )
        
        return decision, explanation
    
    def _validate_against_policies(
        self, 
        citations: List[PolicyCitation], 
        initial_decision: DecisionType,
        category: TicketCategory
    ) -> DecisionType:
        """Validate decision against retrieved policy content"""
        
        if not citations:
            return initial_decision
        
        # Look for explicit policy guidance in citations
        for citation in citations:
            content = citation.chunk_content.lower()
            
            # Check for explicit denial keywords
            if any(term in content for term in ["denied", "prohibited", "not allowed", "forbidden"]):
                return DecisionType.DENIED
            
            # Check for approval requirements
            if any(term in content for term in ["approval required", "manager approval", "authorization needed"]):
                return DecisionType.NEEDS_APPROVAL
            
            # Check for automatic allowance
            if any(term in content for term in ["automatically approved", "standard procedure", "routine"]):
                return DecisionType.ALLOWED
        
        return initial_decision
    
    def _get_decision_rationale(self, decision: DecisionType, category: TicketCategory, priority: TicketPriority) -> str:
        """Get human-readable rationale for decision"""
        rationales = {
            DecisionType.ALLOWED: f"Standard {category.value} request with {priority.value} priority approved per policy",
            DecisionType.DENIED: f"Request denied due to security policy violations or unauthorized access",
            DecisionType.NEEDS_APPROVAL: f"High-impact {category.value} request requires management approval"
        }
        return rationales.get(decision, "Decision based on policy analysis")
    
    def _calculate_decision_confidence(self, decision: DecisionType, citations: List[PolicyCitation]) -> float:
        """Calculate confidence in decision based on policy support"""
        if not citations:
            return 0.6  # Lower confidence without policy backing
        
        # Higher confidence for decisions with strong policy support
        avg_relevance = sum(c.relevance_score for c in citations) / len(citations)
        
        # Boost confidence for explicit policy matches
        if decision == DecisionType.DENIED:
            return min(0.95, 0.8 + avg_relevance * 0.2)  # High confidence for denials
        elif decision == DecisionType.NEEDS_APPROVAL:
            return min(0.9, 0.7 + avg_relevance * 0.3)
        else:
            return min(0.85, 0.6 + avg_relevance * 0.4)
    
    def _identify_missing_info_for_decision(self, title: str, description: str, category: TicketCategory) -> List[str]:
        """Identify missing information that could affect the decision"""
        missing = []
        
        text = f"{title} {description}".lower()
        
        if category == TicketCategory.SOFTWARE and "cost" not in text:
            missing.append("Software licensing cost and budget approval")
        
        if category == TicketCategory.HARDWARE and "model" not in text:
            missing.append("Specific hardware model and compatibility requirements")
        
        if category == TicketCategory.ACCESS and "business justification" not in text:
            missing.append("Business justification for access request")
        
        if "urgent" in text and "deadline" not in text:
            missing.append("Specific deadline and business impact assessment")
        
        return missing
    
    def _generate_decision_alternatives(self, decision: DecisionType, category: TicketCategory) -> List[Dict[str, Any]]:
        """Generate alternative decision options"""
        alternatives = []
        
        if decision == DecisionType.DENIED:
            alternatives.append({
                "option": "Request with modifications",
                "pros": ["May address security concerns", "Allows some functionality"],
                "cons": ["Requires additional approval", "May not meet all user needs"],
                "confidence": 0.7
            })
        
        if decision == DecisionType.NEEDS_APPROVAL:
            alternatives.append({
                "option": "Implement temporary solution",
                "pros": ["Immediate relief", "Maintains security"],
                "cons": ["Not permanent", "May require additional work"],
                "confidence": 0.8
            })
        
        return alternatives
    
    def _generate_decision_counterfactuals(
        self, 
        decision: DecisionType, 
        category: TicketCategory, 
        priority: TicketPriority
    ) -> List[Dict[str, Any]]:
        """Generate counterfactual scenarios for decision"""
        counterfactuals = []
        
        if decision != DecisionType.DENIED:
            counterfactuals.append({
                "condition": "If this request violated security policies",
                "outcome": "Would be automatically denied",
                "likelihood": 0.2
            })
        
        if priority != TicketPriority.CRITICAL:
            counterfactuals.append({
                "condition": "If this were a critical priority issue",
                "outcome": "Would require immediate approval and escalation",
                "likelihood": 0.3
            })
        
        if category != TicketCategory.SECURITY:
            counterfactuals.append({
                "condition": "If this involved security systems",
                "outcome": "Would require additional security review and approval",
                "likelihood": 0.25
            })
        
        return counterfactuals
    
    def kg_enhanced_make_decision(
        self, 
        title: str, 
        description: str, 
        category: TicketCategory,
        priority: TicketPriority,
        db: Session
    ) -> Tuple[DecisionType, KGEnhancedExplanationObject]:
        """
        Make policy-grounded decision using KG-Enhanced RAG system
        
        This method uses the knowledge graph to find related policies and dependencies
        that might affect the decision, providing more comprehensive policy coverage.
        
        Returns: (decision, kg_enhanced_explanation)
        """
        start_time = datetime.now()
        
        # Step 1: Use KG-Enhanced RAG to search for relevant policies
        search_query = f"{category.value} {title} {description}"
        enhanced_citations, graph_hops, kg_metadata = self.policy_retriever.kg_enhanced_retrieve(
            query=search_query, 
            k=5, 
            enable_kg=True,
            max_graph_hops=2,
            db=db
        )
        
        # Step 2: Analyze text for decision patterns (same as before)
        combined_text = f"{title} {description}".lower()
        
        decision = DecisionType.ALLOWED  # Default
        reasoning_trace = []
        step = 1
        
        # Add concept identification step
        concepts_found = kg_metadata.get('initial_concepts_found', [])
        if concepts_found:
            reasoning_trace.append(ReasoningStep(
                step=step,
                action="concept_identification",
                rationale=f"Identified IT concepts: {', '.join(concepts_found)}",
                confidence=0.85,
                policy_refs=[]
            ))
            step += 1
        
        # Add graph traversal step if applicable
        if graph_hops:
            reasoning_trace.append(ReasoningStep(
                step=step,
                action="graph_traversal",
                rationale=f"Found {len(graph_hops)} related concepts through knowledge graph traversal",
                confidence=0.8,
                policy_refs=[c.chunk_id for c in enhanced_citations[:3]]
            ))
            step += 1
        
        # Step 3: Check for auto-denial patterns first (highest priority)
        deny_matches = []
        for pattern in self.decision_rules["auto_deny"]["patterns"]:
            import re
            if re.search(pattern, combined_text):
                deny_matches.append(pattern)
        
        if deny_matches:
            decision = DecisionType.DENIED
            reasoning_trace.append(ReasoningStep(
                step=step,
                action="security_violation_check",
                rationale=f"Denied due to security policy violations: {deny_matches[:2]}",
                confidence=0.95,
                policy_refs=[c.chunk_id for c in enhanced_citations if "security" in c.chunk_content.lower()]
            ))
            step += 1
        
        # Step 4: Check enhanced policy citations for additional constraints
        elif enhanced_citations:
            graph_policy_decision = self._validate_against_kg_enhanced_policies(enhanced_citations, decision, category)
            if graph_policy_decision != decision:
                reasoning_trace.append(ReasoningStep(
                    step=step,
                    action="kg_policy_override",
                    rationale=f"Decision changed from {decision.value} to {graph_policy_decision.value} based on graph-enhanced policy analysis",
                    confidence=0.9,
                    policy_refs=[c.chunk_id for c in enhanced_citations[:3]]
                ))
                decision = graph_policy_decision
                step += 1
        
        # Step 5: Check for approval required patterns
        if decision == DecisionType.ALLOWED:  # Only check if not already denied
            if any(re.search(pattern, combined_text) for pattern in self.decision_rules["needs_approval"]["patterns"]):
                decision = DecisionType.NEEDS_APPROVAL
                reasoning_trace.append(ReasoningStep(
                    step=step,
                    action="approval_requirement_check",
                    rationale=f"Requires approval due to high cost/impact or critical system involvement",
                    confidence=0.9,
                    policy_refs=[c.chunk_id for c in enhanced_citations]
                ))
                step += 1
        
        # Step 6: Check priority-based rules
        if decision == DecisionType.ALLOWED and priority == TicketPriority.CRITICAL:
            if category in [TicketCategory.SECURITY, TicketCategory.NETWORK]:
                decision = DecisionType.NEEDS_APPROVAL
                reasoning_trace.append(ReasoningStep(
                    step=step,
                    action="critical_priority_check",
                    rationale=f"Critical {category.value} issues require management approval",
                    confidence=0.95,
                    policy_refs=[]
                ))
                step += 1
        
        # Step 7: Final policy analysis step
        reasoning_trace.append(ReasoningStep(
            step=step,
            action="kg_enhanced_policy_analysis",
            rationale=f"Analyzed {len(enhanced_citations)} KG-enhanced policy chunks ({kg_metadata.get('semantic_citations_count', 0)} semantic + {len(enhanced_citations) - kg_metadata.get('semantic_citations_count', 0)} graph-based)",
            confidence=0.85,
            policy_refs=[c.chunk_id for c in enhanced_citations]
        ))
        
        end_time = datetime.now()
        decision_time = int((end_time - start_time).total_seconds() * 1000)
        
        # Step 8: Build KG-Enhanced explanation object
        discovered_concepts = list(set([hop.to_concept for hop in graph_hops]))
        graph_coverage = self._calculate_kg_decision_coverage(enhanced_citations, graph_hops)
        
        # Convert KG-enhanced citations to regular citations for base explanation
        regular_citations = [
            PolicyCitation(
                document_id=c.document_id,
                document_title=c.document_title,
                chunk_id=c.chunk_id,
                chunk_content=c.chunk_content,
                relevance_score=c.combined_score
            )
            for c in enhanced_citations
        ]
        
        kg_enhanced_explanation = KGEnhancedExplanationObject(
            # Base explanation fields
            answer=f"Decision: {decision.value.upper()} - {self._get_decision_rationale(decision, category, priority)}",
            decision=f"decision={decision.value}, category={category.value}, priority={priority.value}",
            confidence=self._calculate_kg_decision_confidence(decision, enhanced_citations),
            reasoning_trace=reasoning_trace,
            policy_citations=regular_citations,
            missing_info=self._identify_missing_info_for_decision(title, description, category),
            alternatives_considered=self._generate_decision_alternatives(decision, category),
            counterfactuals=self._generate_decision_counterfactuals(decision, category, priority),
            telemetry=TelemetryData(
                latency_ms=decision_time,
                retrieval_k=len(enhanced_citations),
                triage_time_ms=0,
                planning_time_ms=kg_metadata.get('total_processing_time_ms', 0),
                total_chunks_considered=len(enhanced_citations)
            ),
            timestamp=datetime.now(),
            model_version="1.1.0",  # Updated version for KG-Enhanced
            
            # KG-Enhanced fields
            kg_policy_citations=enhanced_citations,
            graph_reasoning=graph_hops,
            concepts_discovered=discovered_concepts,
            graph_coverage_score=graph_coverage
        )
        
        return decision, kg_enhanced_explanation
    
    def _validate_against_kg_enhanced_policies(
        self, 
        enhanced_citations: List[KGEnhancedPolicyCitation], 
        initial_decision: DecisionType,
        category: TicketCategory
    ) -> DecisionType:
        """
        Validate decision against KG-enhanced policy citations
        
        This considers both semantic and graph-discovered policies for decision validation
        """
        if not enhanced_citations:
            return initial_decision
        
        # Prioritize graph-discovered policies as they show dependencies
        graph_discovered = [c for c in enhanced_citations if c.semantic_score == 0.0 and c.graph_boost_score > 0]
        semantic_policies = [c for c in enhanced_citations if c.semantic_score > 0]
        
        # Check graph-discovered policies first (dependencies matter!)
        for citation in graph_discovered:
            content = citation.chunk_content.lower()
            
            # Check for dependency requirements
            if any(term in content for term in ["requires", "depends on", "must have", "prerequisite"]):
                if "approval" in content or "authorization" in content:
                    return DecisionType.NEEDS_APPROVAL
                elif "denied" in content or "prohibited" in content:
                    return DecisionType.DENIED
        
        # Then check semantic policies
        for citation in semantic_policies:
            content = citation.chunk_content.lower()
            
            # Check for explicit policy guidance
            if any(term in content for term in ["denied", "prohibited", "not allowed", "forbidden"]):
                return DecisionType.DENIED
            
            if any(term in content for term in ["approval required", "manager approval", "authorization needed"]):
                return DecisionType.NEEDS_APPROVAL
            
            if any(term in content for term in ["automatically approved", "standard procedure", "routine"]):
                return DecisionType.ALLOWED
        
        return initial_decision
    
    def _calculate_kg_decision_confidence(self, decision: DecisionType, enhanced_citations: List[KGEnhancedPolicyCitation]) -> float:
        """Calculate confidence in decision based on KG-enhanced policy support"""
        if not enhanced_citations:
            return 0.6  # Lower confidence without policy backing
        
        # Calculate average combined score (semantic + graph)
        avg_combined_score = sum(c.combined_score for c in enhanced_citations) / len(enhanced_citations)
        
        # Count graph-enhanced citations (show better coverage)
        graph_enhanced_count = sum(1 for c in enhanced_citations if c.graph_boost_score > 0)
        graph_enhancement_ratio = graph_enhanced_count / len(enhanced_citations)
        
        # Base confidence from policy relevance
        base_confidence = min(0.85, 0.6 + avg_combined_score * 0.3)
        
        # Boost confidence based on graph enhancement (better coverage = higher confidence)
        graph_boost = graph_enhancement_ratio * 0.15
        
        # Decision-specific confidence adjustments
        if decision == DecisionType.DENIED:
            final_confidence = min(0.98, base_confidence + graph_boost + 0.1)  # High confidence for denials
        elif decision == DecisionType.NEEDS_APPROVAL:
            final_confidence = min(0.95, base_confidence + graph_boost + 0.05)
        else:
            final_confidence = min(0.9, base_confidence + graph_boost)
        
        return final_confidence
    
    def _calculate_kg_decision_coverage(self, enhanced_citations: List[KGEnhancedPolicyCitation], graph_hops: List[GraphHop]) -> float:
        """Calculate how well the KG-Enhanced RAG covered the decision scenario"""
        if not enhanced_citations:
            return 0.0
        
        # Base coverage from citation count
        citation_coverage = min(1.0, len(enhanced_citations) / 5)  # Assume 5 is optimal
        
        # Graph enhancement coverage
        graph_citations = sum(1 for c in enhanced_citations if c.graph_boost_score > 0)
        graph_coverage = graph_citations / max(len(enhanced_citations), 1)
        
        # Relationship depth coverage
        depth_coverage = min(1.0, len(graph_hops) / 3)  # Assume 3 hops is good coverage
        
        # Combine coverage metrics
        total_coverage = (citation_coverage * 0.5 + graph_coverage * 0.3 + depth_coverage * 0.2)
        
        return min(1.0, total_coverage)
