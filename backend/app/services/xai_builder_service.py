from typing import List, Dict, Any
from datetime import datetime
from app.models.schemas import ExplanationObject, ReasoningStep, PolicyCitation, TelemetryData

class XAIBuilderService:
    """
    Service for building composite XAI explanations from multiple processing stages
    """
    
    def __init__(self):
        pass
    
    def build_composite_explanation(
        self, 
        triage_explanation: ExplanationObject, 
        plan_explanation: ExplanationObject,
        decision_explanation: ExplanationObject = None
    ) -> ExplanationObject:
        """
        Build a comprehensive explanation by merging triage, decision, and planning explanations
        """
        
        # Merge reasoning traces
        combined_reasoning = []
        step_offset = 0
        
        # Add triage reasoning
        for step in triage_explanation.reasoning_trace:
            combined_reasoning.append(ReasoningStep(
                step=step.step + step_offset,
                action=f"triage_{step.action}",
                rationale=step.rationale,
                confidence=step.confidence,
                policy_refs=step.policy_refs
            ))
        step_offset += len(triage_explanation.reasoning_trace)
        
        # Add decision reasoning (if provided)
        if decision_explanation:
            for step in decision_explanation.reasoning_trace:
                combined_reasoning.append(ReasoningStep(
                    step=step.step + step_offset,
                    action=f"decision_{step.action}",
                    rationale=step.rationale,
                    confidence=step.confidence,
                    policy_refs=step.policy_refs
                ))
            step_offset += len(decision_explanation.reasoning_trace)
        
        # Add planning reasoning
        for step in plan_explanation.reasoning_trace:
            combined_reasoning.append(ReasoningStep(
                step=step.step + step_offset,
                action=f"planning_{step.action}",
                rationale=step.rationale,
                confidence=step.confidence,
                policy_refs=step.policy_refs
            ))
        
        # Merge policy citations
        all_citations = triage_explanation.policy_citations + plan_explanation.policy_citations
        if decision_explanation:
            all_citations.extend(decision_explanation.policy_citations)
        unique_citations = self._deduplicate_citations(all_citations)
        
        # Combine missing information
        combined_missing = list(set(
            triage_explanation.missing_info + plan_explanation.missing_info +
            (decision_explanation.missing_info if decision_explanation else [])
        ))
        
        # Combine alternatives and counterfactuals
        combined_alternatives = triage_explanation.alternatives_considered + plan_explanation.alternatives_considered
        if decision_explanation:
            combined_alternatives.extend(decision_explanation.alternatives_considered)
            combined_counterfactuals = triage_explanation.counterfactuals + plan_explanation.counterfactuals + decision_explanation.counterfactuals
        else:
            combined_counterfactuals = triage_explanation.counterfactuals + plan_explanation.counterfactuals
        
        # Calculate composite confidence
        explanations = [triage_explanation, plan_explanation]
        if decision_explanation:
            explanations.append(decision_explanation)
        confidence = sum(exp.confidence for exp in explanations) / len(explanations)
        
        # Merge telemetry
        total_latency = triage_explanation.telemetry.latency_ms + plan_explanation.telemetry.latency_ms
        total_chunks = (
            triage_explanation.telemetry.total_chunks_considered + 
            plan_explanation.telemetry.total_chunks_considered
        )
        
        if decision_explanation:
            total_latency += decision_explanation.telemetry.latency_ms
            total_chunks += decision_explanation.telemetry.total_chunks_considered
        
        combined_telemetry = TelemetryData(
            latency_ms=total_latency,
            retrieval_k=max(
                triage_explanation.telemetry.retrieval_k, 
                plan_explanation.telemetry.retrieval_k,
                decision_explanation.telemetry.retrieval_k if decision_explanation else 0
            ),
            triage_time_ms=triage_explanation.telemetry.triage_time_ms,
            planning_time_ms=plan_explanation.telemetry.planning_time_ms,
            total_chunks_considered=total_chunks
        )
        
        # Build composite answer
        if decision_explanation:
            composite_answer = f"""TRIAGE: {triage_explanation.answer}

DECISION: {decision_explanation.answer}

PLAN: {plan_explanation.answer}

COMPOSITE: Ticket processed with full policy-grounded decision and action plan at {confidence:.1%} confidence"""
        else:
            composite_answer = f"""TRIAGE: {triage_explanation.answer}

PLAN: {plan_explanation.answer}

COMPOSITE: Ticket categorized and action plan generated with {confidence:.1%} confidence"""
        
        # Build composite decision summary
        if decision_explanation:
            composite_decision = f"{triage_explanation.decision}, {decision_explanation.decision}, {plan_explanation.decision}"
        else:
            composite_decision = f"{triage_explanation.decision}, {plan_explanation.decision}"
        
        return ExplanationObject(
            answer=composite_answer,
            decision=composite_decision,
            confidence=confidence,
            reasoning_trace=combined_reasoning,
            policy_citations=unique_citations,
            missing_info=combined_missing,
            alternatives_considered=combined_alternatives,
            counterfactuals=combined_counterfactuals,
            telemetry=combined_telemetry,
            timestamp=datetime.now(),
            model_version="1.0.0"
        )
    
    def _deduplicate_citations(self, citations: List[PolicyCitation]) -> List[PolicyCitation]:
        """Remove duplicate policy citations based on chunk_id"""
        seen_chunks = set()
        unique_citations = []
        
        for citation in citations:
            if citation.chunk_id not in seen_chunks:
                unique_citations.append(citation)
                seen_chunks.add(citation.chunk_id)
        
        # Sort by relevance score (highest first)
        return sorted(unique_citations, key=lambda c: c.relevance_score, reverse=True)
    
    def generate_summary_for_user(self, explanation: ExplanationObject, decision_type: str = None) -> str:
        """
        Generate a clean, user-friendly summary hiding complex reasoning
        This is what users see - NOT the full explanation
        """
        summary_parts = []
        
        # Extract basic facts from decision
        if "category=" in explanation.decision:
            category = explanation.decision.split("category=")[1].split(",")[0]
            summary_parts.append(f"Categorized as: {category.title()}")
        
        if "priority=" in explanation.decision:
            priority = explanation.decision.split("priority=")[1].split(",")[0]
            summary_parts.append(f"Priority: {priority.title()}")
        
        if decision_type:
            decision_icons = {
                "allowed": "✅ Approved",
                "denied": "❌ Denied", 
                "needs_approval": "⏳ Requires Approval"
            }
            summary_parts.append(decision_icons.get(decision_type, f"Decision: {decision_type}"))
        
        # Add confidence if high enough
        if explanation.confidence >= 0.8:
            summary_parts.append(f"Confidence: {explanation.confidence:.1%}")
        
        return " | ".join(summary_parts)
