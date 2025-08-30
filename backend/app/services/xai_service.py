from typing import List, Dict, Any, Optional
from datetime import datetime
from app.models.models import TicketCategory, TicketPriority
from app.models.schemas import ExplanationObject, ReasoningStep, PolicyCitation, AlternativeOption, Counterfactual, TelemetryData
from app.services.triage_service import TriageService
from app.services.planner_service import PlannerService
from sqlalchemy.orm import Session

class XAIService:
    def __init__(self):
        self.triage_service = TriageService()
        self.planner_service = PlannerService()
    
    def explain_ticket_processing(
        self, 
        title: str, 
        description: str, 
        db: Session,
        include_planning: bool = True
    ) -> Dict[str, Any]:
        """
        Generate complete XAI explanation for ticket processing
        
        Returns comprehensive explanation including triage and planning
        """
        start_time = datetime.now()
        
        # Perform triage with explanation
        category, priority, triage_confidence, triage_explanation = self.triage_service.triage_ticket(
            title, description
        )
        
        # Generate action plan if requested
        action_plan = []
        planning_explanation = None
        
        if include_planning:
            action_plan, planning_explanation = self.planner_service.create_action_plan(
                category, priority, title, description, db
            )
        
        # Combine explanations
        combined_explanation = self._merge_explanations(
            triage_explanation, 
            planning_explanation,
            category,
            priority,
            action_plan
        )
        
        end_time = datetime.now()
        total_latency = int((end_time - start_time).total_seconds() * 1000)
        
        # Update telemetry with total time
        combined_explanation.telemetry.latency_ms = total_latency
        
        return {
            "category": category,
            "priority": priority,
            "confidence": triage_confidence,
            "action_plan": action_plan,
            "explanation": combined_explanation,
            "processing_time_ms": total_latency
        }
    
    def _merge_explanations(
        self, 
        triage_explanation: ExplanationObject,
        planning_explanation: Optional[ExplanationObject],
        category: TicketCategory,
        priority: TicketPriority,
        action_plan: List[str]
    ) -> ExplanationObject:
        """Merge triage and planning explanations into a comprehensive explanation"""
        
        # Start with triage explanation as base
        merged_reasoning = triage_explanation.reasoning_trace.copy()
        
        # Add planning reasoning if available
        if planning_explanation:
            # Renumber planning steps to continue from triage
            for step in planning_explanation.reasoning_trace:
                step.step += len(merged_reasoning)
                merged_reasoning.append(step)
        
        # Combine policy citations
        all_citations = triage_explanation.policy_citations.copy()
        if planning_explanation:
            all_citations.extend(planning_explanation.policy_citations)
        
        # Remove duplicate citations based on chunk_id
        unique_citations = []
        seen_chunks = set()
        for citation in all_citations:
            if citation.chunk_id not in seen_chunks:
                unique_citations.append(citation)
                seen_chunks.add(citation.chunk_id)
        
        # Combine missing info
        all_missing_info = triage_explanation.missing_info.copy()
        if planning_explanation:
            for info in planning_explanation.missing_info:
                if info not in all_missing_info:
                    all_missing_info.append(info)
        
        # Combine alternatives
        all_alternatives = triage_explanation.alternatives_considered.copy()
        if planning_explanation:
            all_alternatives.extend(planning_explanation.alternatives_considered)
        
        # Combine counterfactuals
        all_counterfactuals = triage_explanation.counterfactuals.copy()
        if planning_explanation:
            all_counterfactuals.extend(planning_explanation.counterfactuals)
        
        # Calculate combined confidence
        if planning_explanation:
            combined_confidence = (triage_explanation.confidence + planning_explanation.confidence) / 2
        else:
            combined_confidence = triage_explanation.confidence
        
        # Build comprehensive answer
        answer_parts = [
            f"Ticket categorized as {category.value} with {priority.value} priority"
        ]
        
        if action_plan:
            answer_parts.append(f"Generated {len(action_plan)}-step action plan")
        
        if unique_citations:
            answer_parts.append(f"Grounded in {len(unique_citations)} policy citations")
        
        # Merge telemetry
        total_chunks = (triage_explanation.telemetry.total_chunks_considered + 
                       (planning_explanation.telemetry.total_chunks_considered if planning_explanation else 0))
        
        merged_telemetry = TelemetryData(
            latency_ms=triage_explanation.telemetry.latency_ms,  # Will be updated by caller
            retrieval_k=len(unique_citations),
            triage_time_ms=triage_explanation.telemetry.triage_time_ms,
            planning_time_ms=planning_explanation.telemetry.planning_time_ms if planning_explanation else 0,
            total_chunks_considered=total_chunks
        )
        
        # Create merged explanation
        return ExplanationObject(
            answer=". ".join(answer_parts),
            decision=f"category={category.value}, priority={priority.value}, action_steps={len(action_plan)}",
            confidence=combined_confidence,
            reasoning_trace=merged_reasoning,
            policy_citations=unique_citations,
            missing_info=all_missing_info,
            alternatives_considered=all_alternatives,
            counterfactuals=all_counterfactuals,
            telemetry=merged_telemetry,
            timestamp=datetime.now(),
            model_version="1.0.0"
        )
    
    def explain_decision_change(
        self, 
        old_category: TicketCategory, 
        new_category: TicketCategory,
        old_priority: TicketPriority,
        new_priority: TicketPriority,
        reason: str,
        changed_by: str
    ) -> ExplanationObject:
        """Generate explanation for manual decision changes"""
        
        reasoning_trace = []
        step = 1
        
        if old_category != new_category:
            reasoning_trace.append(ReasoningStep(
                step=step,
                action="category_change",
                rationale=f"Category changed from {old_category.value} to {new_category.value} by {changed_by}",
                confidence=1.0,  # Human decision
                policy_refs=[]
            ))
            step += 1
        
        if old_priority != new_priority:
            reasoning_trace.append(ReasoningStep(
                step=step,
                action="priority_change",
                rationale=f"Priority changed from {old_priority.value} to {new_priority.value} by {changed_by}",
                confidence=1.0,  # Human decision
                policy_refs=[]
            ))
            step += 1
        
        if reason:
            reasoning_trace.append(ReasoningStep(
                step=step,
                action="manual_override_reason",
                rationale=f"Human provided reason: {reason}",
                confidence=1.0,
                policy_refs=[]
            ))
        
        return ExplanationObject(
            answer=f"Ticket classification manually updated by {changed_by}",
            decision=f"category={new_category.value}, priority={new_priority.value}",
            confidence=1.0,  # Human decisions are treated as certain
            reasoning_trace=reasoning_trace,
            policy_citations=[],
            missing_info=[],
            alternatives_considered=[],
            counterfactuals=[],
            telemetry=TelemetryData(
                latency_ms=0,
                retrieval_k=0,
                triage_time_ms=0,
                planning_time_ms=0,
                total_chunks_considered=0
            ),
            timestamp=datetime.now(),
            model_version="1.0.0"
        )
    
    def validate_explanation_schema(self, explanation: ExplanationObject) -> Dict[str, Any]:
        """Validate explanation object against expected schema"""
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        if not explanation.answer:
            validation_results["errors"].append("Missing answer field")
        
        if not explanation.decision:
            validation_results["errors"].append("Missing decision field")
        
        if explanation.confidence < 0 or explanation.confidence > 1:
            validation_results["errors"].append("Confidence must be between 0 and 1")
        
        if not explanation.reasoning_trace:
            validation_results["warnings"].append("No reasoning trace provided")
        
        # Check reasoning trace consistency
        for i, step in enumerate(explanation.reasoning_trace):
            if step.step != i + 1:
                validation_results["warnings"].append(f"Reasoning step {step.step} out of sequence")
        
        # Check citations
        if explanation.policy_citations:
            for citation in explanation.policy_citations:
                if citation.relevance_score < 0 or citation.relevance_score > 1:
                    validation_results["warnings"].append(f"Citation relevance score out of range: {citation.relevance_score}")
        
        validation_results["is_valid"] = len(validation_results["errors"]) == 0
        
        return validation_results
