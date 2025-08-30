from typing import List, Dict, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.models import TicketCategory, TicketPriority
from app.models.schemas import ExplanationObject, ReasoningStep, PolicyCitation, AlternativeOption, Counterfactual, TelemetryData
from app.services.policy_retriever import PolicyRetriever

class PlannerService:
    def __init__(self):
        self.policy_retriever = PolicyRetriever()
        
        # Default action plans by category when policies are not available
        self.default_plans = {
            TicketCategory.HARDWARE: [
                "Verify hardware connections and power status",
                "Check device manager for hardware errors",
                "Test with known good hardware components",
                "Update device drivers if applicable",
                "Document findings and escalate if unresolved"
            ],
            TicketCategory.SOFTWARE: [
                "Identify specific software and error messages",
                "Check for recent software updates or changes",
                "Verify software licensing and compatibility",
                "Attempt software repair or reinstallation",
                "Test functionality and document resolution"
            ],
            TicketCategory.NETWORK: [
                "Test network connectivity at multiple layers",
                "Check network configuration and settings",
                "Verify DNS and gateway connectivity",
                "Test with alternative network connections",
                "Document network diagnostics and findings"
            ],
            TicketCategory.ACCESS: [
                "Verify user account status and permissions",
                "Reset password following security protocols",
                "Check group memberships and access rights",
                "Test access from different devices/locations",
                "Update access logs and notify user"
            ],
            TicketCategory.SECURITY: [
                "Isolate affected systems immediately",
                "Perform security scan and malware detection",
                "Review security logs for indicators of compromise",
                "Apply security patches and updates",
                "Document incident and implement preventive measures"
            ],
            TicketCategory.OTHER: [
                "Gather detailed information about the issue",
                "Research similar issues in knowledge base",
                "Consult relevant documentation and policies",
                "Implement appropriate solution or escalate",
                "Follow up with user and document resolution"
            ]
        }
    
    def create_action_plan(
        self, 
        category: TicketCategory, 
        priority: TicketPriority, 
        title: str, 
        description: str,
        db: Session
    ) -> Tuple[List[str], ExplanationObject]:
        """
        Create a policy-grounded action plan for a ticket
        
        Returns: (action_steps, explanation)
        """
        start_time = datetime.now()
        
        # Search for relevant policy chunks
        search_query = f"{category.value} {title} {description}"
        policy_citations = self.policy_retriever.retrieve_relevant_chunks(
            search_query, k=5, db=db
        )
        
        # Build action plan
        action_steps = []
        reasoning_trace = []
        alternatives = []
        
        if policy_citations and policy_citations[0].relevance_score > 0.3:
            # Policy-grounded planning
            action_steps = self._extract_actions_from_policies(policy_citations, category)
            
            reasoning_trace.append(ReasoningStep(
                step=1,
                action="policy_retrieval",
                rationale=f"Retrieved {len(policy_citations)} relevant policy chunks with top score {policy_citations[0].relevance_score:.3f}",
                confidence=0.8,
                policy_refs=[c.chunk_id for c in policy_citations]
            ))
            
            reasoning_trace.append(ReasoningStep(
                step=2,
                action="plan_generation",
                rationale=f"Generated {len(action_steps)} action steps based on policy guidance",
                confidence=0.9,
                policy_refs=[c.chunk_id for c in policy_citations[:3]]
            ))
            
            # Consider alternatives
            alternatives.append(AlternativeOption(
                option="Use default category-based plan",
                pros=["Faster execution", "Always available"],
                cons=["Less specific", "May miss policy requirements"],
                confidence=0.7
            ))
            
        else:
            # Fall back to default plan
            action_steps = self.default_plans[category].copy()
            
            reasoning_trace.append(ReasoningStep(
                step=1,
                action="policy_retrieval",
                rationale=f"Retrieved {len(policy_citations)} policy chunks but low relevance scores",
                confidence=0.5,
                policy_refs=[]
            ))
            
            reasoning_trace.append(ReasoningStep(
                step=2,
                action="fallback_plan",
                rationale=f"Using default {category.value} action plan due to insufficient policy guidance",
                confidence=0.6,
                policy_refs=[]
            ))
        
        # Adjust plan based on priority
        if priority == TicketPriority.CRITICAL:
            action_steps.insert(0, "URGENT: Implement immediate containment measures")
            action_steps.append("Notify management and stakeholders immediately")
        elif priority == TicketPriority.HIGH:
            action_steps.insert(0, "Prioritize this issue over lower priority tickets")
        
        # Add priority-based reasoning
        reasoning_trace.append(ReasoningStep(
            step=len(reasoning_trace) + 1,
            action="priority_adjustment",
            rationale=f"Adjusted plan for {priority.value} priority ticket",
            confidence=0.9,
            policy_refs=[]
        ))
        
        end_time = datetime.now()
        planning_time = int((end_time - start_time).total_seconds() * 1000)
        
        # Generate counterfactuals
        counterfactuals = self._generate_counterfactuals(category, priority)
        
        # Build explanation
        explanation = ExplanationObject(
            answer=f"Generated {len(action_steps)}-step action plan for {category.value} issue",
            decision=f"action_plan_steps={len(action_steps)}",
            confidence=0.8 if policy_citations else 0.6,
            reasoning_trace=reasoning_trace,
            policy_citations=policy_citations,
            missing_info=self._identify_missing_info(title, description),
            alternatives_considered=alternatives,
            counterfactuals=counterfactuals,
            telemetry=TelemetryData(
                latency_ms=planning_time,
                retrieval_k=len(policy_citations),
                triage_time_ms=0,
                planning_time_ms=planning_time,
                total_chunks_considered=len(policy_citations)
            ),
            timestamp=datetime.now(),
            model_version="1.0.0"
        )
        
        return action_steps, explanation
    
    def _extract_actions_from_policies(self, citations: List[PolicyCitation], category: TicketCategory) -> List[str]:
        """Extract actionable steps from policy citations"""
        actions = []
        
        # Look for action-oriented sentences in policy content
        for citation in citations:
            content = citation.chunk_content
            
            # Split into sentences and look for imperative statements
            sentences = content.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if self._is_actionable_sentence(sentence):
                    actions.append(sentence.capitalize())
        
        # If we have policy-derived actions, use them; otherwise fall back to defaults
        if len(actions) >= 3:
            return actions[:6]  # Limit to 6 steps
        else:
            # Merge with default plan
            default_actions = self.default_plans[category]
            combined = actions + default_actions
            return combined[:6]
    
    def _is_actionable_sentence(self, sentence: str) -> bool:
        """Check if a sentence contains actionable instructions"""
        action_indicators = [
            'verify', 'check', 'test', 'ensure', 'confirm', 'validate',
            'update', 'install', 'configure', 'reset', 'restart',
            'document', 'notify', 'escalate', 'follow', 'implement'
        ]
        
        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in action_indicators) and len(sentence) > 20
    
    def _identify_missing_info(self, title: str, description: str) -> List[str]:
        """Identify potentially missing information for better troubleshooting"""
        missing = []
        
        text = f"{title} {description}".lower()
        
        # Check for common missing details
        if 'error' in text and 'error message' not in text:
            missing.append("Specific error message or error code")
        
        if any(hw in text for hw in ['computer', 'laptop', 'device']) and 'model' not in text:
            missing.append("Device model and specifications")
        
        if 'software' in text and 'version' not in text:
            missing.append("Software version and build information")
        
        if 'network' in text and 'ip' not in text:
            missing.append("Network configuration details (IP, DNS, Gateway)")
        
        if len(description.split()) < 10:
            missing.append("More detailed description of the issue")
        
        return missing
    
    def _generate_counterfactuals(self, category: TicketCategory, priority: TicketPriority) -> List[Counterfactual]:
        """Generate counterfactual scenarios"""
        counterfactuals = []
        
        if priority != TicketPriority.CRITICAL:
            counterfactuals.append(Counterfactual(
                condition="If this were a critical priority issue",
                outcome="Immediate escalation and faster response time would be required",
                likelihood=0.3
            ))
        
        if category != TicketCategory.SECURITY:
            counterfactuals.append(Counterfactual(
                condition="If this were a security-related issue",
                outcome="Additional isolation and security scanning steps would be required",
                likelihood=0.2
            ))
        
        counterfactuals.append(Counterfactual(
            condition="If user provided more detailed information",
            outcome="More specific troubleshooting steps could be recommended",
            likelihood=0.6
        ))
        
        return counterfactuals
