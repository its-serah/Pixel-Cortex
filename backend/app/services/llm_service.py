import json
import re
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
import ollama
from app.models.models import TicketCategory, TicketPriority
from app.models.schemas import ExplanationObject, ReasoningStep, PolicyCitation, TelemetryData
from app.services.policy_retriever import PolicyRetriever
# from app.services.audit_service import AuditService  # not required at runtime

class LLMService:
    """
    Local LLM service with Chain-of-Thought reasoning, RAG, and strict IT support guardrails
    Uses lightweight models like Llama 3.2 3B or Phi-3 Mini for fast inference
    """
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name
        self.policy_retriever = PolicyRetriever()
        
        # IT Support domain restrictions
        self.allowed_domains = [
            "information technology", "computer support", "network issues",
            "software problems", "hardware troubleshooting", "system administration",
            "cybersecurity", "data recovery", "user access", "technical support",
            "it helpdesk", "system maintenance", "software installation",
            "network connectivity", "email issues", "password reset",
            "printer problems", "computer repair", "system updates"
        ]
        
        # Forbidden topics (strict guardrails)
        self.forbidden_patterns = [
            r"(?i)(sports|football|soccer|basketball)",
            r"(?i)(politics|government|voting|election)",
            r"(?i)(entertainment|movies|music|celebrities)",
            r"(?i)(personal life|dating|relationships)",
            r"(?i)(cooking|recipes|food|restaurant)",
            r"(?i)(travel|vacation|tourism|hotels)",
            r"(?i)(legal advice|medical advice|financial advice)",
            r"(?i)(religion|philosophy|ethics)"
        ]
        
        # Core system prompt for IT support focus
        self.system_prompt = """You are a specialized IT Support Assistant. Your ONLY role is to help with:
- Computer hardware and software issues
- Network connectivity problems  
- System administration tasks
- Cybersecurity concerns
- User access and authentication
- Technical troubleshooting
- IT policy compliance

STRICT RULES:
1. ONLY answer IT support questions
2. If asked about non-IT topics, politely decline and redirect to IT issues
3. Always search for relevant policies before answering
4. Provide step-by-step Chain-of-Thought reasoning
5. Ground all responses in official company policies when available
6. Be concise but thorough in technical explanations
7. Focus on practical, actionable solutions

If you don't know something about IT policies or procedures, say "I need to search our IT policies" and request policy lookup."""
    
    def __init_model(self):
        """Initialize the Ollama model if not already done"""
        try:
            # Check if model is available
            models = ollama.list()
            if not any(model['name'] == self.model_name for model in models.get('models', [])):
                print(f"Pulling {self.model_name} model...")
                ollama.pull(self.model_name)
            
            print(f"LLM service initialized with {self.model_name}")
            return True
        except Exception as e:
            print(f"Failed to initialize LLM: {e}")
            return False
    
    def generate_response(
        self, 
        query: str, 
        context: str = "",
        db: Session = None,
        include_cot: bool = True
    ) -> Tuple[str, ExplanationObject]:
        """
        Generate LLM response with Chain-of-Thought reasoning
        
        Args:
            query: User question/request
            context: Additional context (ticket details, etc.)
            db: Database session for policy retrieval
            include_cot: Whether to include detailed reasoning
            
        Returns:
            (response_text, explanation_object)
        """
        start_time = datetime.now()
        
        # 1. Validate domain (guardrails)
        if not self._is_valid_it_query(query):
            return self._handle_invalid_query(query)
        
        # 2. Initialize model if needed
        if not self.__init_model():
            return self._handle_model_error()
        
        # 3. Search for relevant policies (RAG)
        policy_citations = []
        if db:
            policy_citations = self.policy_retriever.retrieve_relevant_chunks(
                query, k=3, db=db
            )
        
        # 4. Build enhanced prompt with policies and CoT instruction
        enhanced_prompt = self._build_enhanced_prompt(query, context, policy_citations, include_cot)
        
        # 5. Generate LLM response
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": enhanced_prompt}
                ],
                options={
                    "temperature": 0.1,  # Low temperature for consistent, focused responses
                    "top_p": 0.9,
                    "max_tokens": 1000,
                    "stop": ["---END---", "\n\nUser:", "\n\nHuman:"]
                }
            )
            
            response_text = response['message']['content']
            
        except Exception as e:
            return self._handle_llm_error(str(e))
        
        end_time = datetime.now()
        processing_time = int((end_time - start_time).total_seconds() * 1000)
        
        # 6. Extract Chain-of-Thought reasoning (STORE IN LOGS ONLY)
        cot_reasoning, clean_response = self._extract_cot_reasoning(response_text)
        
        # 7. Build explanation object for audit logging
        explanation = self._build_llm_explanation(
            query, clean_response, cot_reasoning, policy_citations, processing_time
        )
        
        return clean_response, explanation
    
    def _is_valid_it_query(self, query: str) -> bool:
        """Check if query is within IT support domain"""
        
        # Check for forbidden topics
        for pattern in self.forbidden_patterns:
            if re.search(pattern, query):
                return False
        
        # Check for IT-related keywords
        it_keywords = [
            "computer", "software", "hardware", "network", "system", "server",
            "password", "login", "access", "email", "printer", "wifi", "internet",
            "application", "program", "install", "update", "error", "bug", "issue",
            "troubleshoot", "fix", "repair", "configure", "setup", "admin",
            "security", "firewall", "antivirus", "backup", "restore", "crash"
        ]
        
        query_lower = query.lower()
        has_it_keywords = any(keyword in query_lower for keyword in it_keywords)
        
        return has_it_keywords or len(query.split()) < 5  # Allow short queries
    
    def _build_enhanced_prompt(
        self, 
        query: str, 
        context: str, 
        policy_citations: List[PolicyCitation],
        include_cot: bool
    ) -> str:
        """Build enhanced prompt with policies and CoT instructions"""
        
        prompt_parts = []
        
        # Add context if provided
        if context:
            prompt_parts.append(f"CONTEXT: {context}")
        
        # Add policy information (RAG)
        if policy_citations:
            prompt_parts.append("RELEVANT IT POLICIES:")
            for i, citation in enumerate(policy_citations[:3], 1):
                prompt_parts.append(f"{i}. {citation.document_title}:")
                prompt_parts.append(f"   {citation.chunk_content[:300]}...")
        
        # Add the actual query
        prompt_parts.append(f"IT SUPPORT REQUEST: {query}")
        
        # Add Chain-of-Thought instruction
        if include_cot:
            prompt_parts.append("""
INSTRUCTIONS:
1. Think step-by-step using Chain-of-Thought reasoning
2. Start your response with <THINKING> tags for your reasoning process
3. Reference specific policies if they apply to this situation
4. End with <RESPONSE> tags containing your final answer
5. Keep focused ONLY on IT support - decline any off-topic requests

Format:
<THINKING>
Step 1: Analyze the IT issue...
Step 2: Check relevant policies...
Step 3: Consider technical solutions...
Step 4: Evaluate risks and requirements...
</THINKING>

<RESPONSE>
Your final IT support answer here
</RESPONSE>""")
        else:
            prompt_parts.append("\nProvide a direct, concise IT support response based on the policies above.")
        
        return "\n\n".join(prompt_parts)
    
    def _extract_cot_reasoning(self, response_text: str) -> Tuple[List[Dict], str]:
        """Extract Chain-of-Thought reasoning from LLM response"""
        
        cot_reasoning = []
        clean_response = response_text
        
        # Extract thinking section
        thinking_match = re.search(r'<THINKING>(.*?)</THINKING>', response_text, re.DOTALL)
        if thinking_match:
            thinking_content = thinking_match.group(1).strip()
            
            # Parse thinking steps
            steps = re.findall(r'Step (\d+):\s*(.*?)(?=Step \d+|$)', thinking_content, re.DOTALL)
            for step_num, step_content in steps:
                cot_reasoning.append({
                    "step": int(step_num),
                    "reasoning": step_content.strip(),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Extract clean response
        response_match = re.search(r'<RESPONSE>(.*?)</RESPONSE>', response_text, re.DOTALL)
        if response_match:
            clean_response = response_match.group(1).strip()
        else:
            # If no response tags, use full text but remove thinking
            clean_response = re.sub(r'<THINKING>.*?</THINKING>', '', response_text, flags=re.DOTALL).strip()
        
        return cot_reasoning, clean_response
    
    def _build_llm_explanation(
        self,
        query: str,
        response: str, 
        cot_reasoning: List[Dict],
        policy_citations: List[PolicyCitation],
        processing_time: int
    ) -> ExplanationObject:
        """Build comprehensive explanation object for LLM response"""
        
        reasoning_trace = []
        step = 1
        
        # Add query analysis step
        reasoning_trace.append(ReasoningStep(
            step=step,
            action="query_analysis",
            rationale=f"Analyzed IT support query: '{query[:100]}...'",
            confidence=0.9,
            policy_refs=[]
        ))
        step += 1
        
        # Add policy retrieval step
        if policy_citations:
            reasoning_trace.append(ReasoningStep(
                step=step,
                action="policy_retrieval",
                rationale=f"Retrieved {len(policy_citations)} relevant policy chunks for context",
                confidence=0.85,
                policy_refs=[c.chunk_id for c in policy_citations]
            ))
            step += 1
        
        # Add Chain-of-Thought reasoning steps
        for cot_step in cot_reasoning:
            reasoning_trace.append(ReasoningStep(
                step=step,
                action=f"llm_reasoning_step_{cot_step['step']}",
                rationale=cot_step["reasoning"][:200] + ("..." if len(cot_step["reasoning"]) > 200 else ""),
                confidence=0.8,
                policy_refs=[c.chunk_id for c in policy_citations if any(word in cot_step["reasoning"].lower() for word in c.document_title.lower().split())]
            ))
            step += 1
        
        # Add response generation step
        reasoning_trace.append(ReasoningStep(
            step=step,
            action="response_generation",
            rationale=f"Generated IT support response using {self.model_name} with policy grounding",
            confidence=0.85,
            policy_refs=[c.chunk_id for c in policy_citations]
        ))
        
        return ExplanationObject(
            answer=response,
            decision=f"llm_response=generated, policies_used={len(policy_citations)}, cot_steps={len(cot_reasoning)}",
            confidence=self._calculate_response_confidence(response, policy_citations, cot_reasoning),
            reasoning_trace=reasoning_trace,
            policy_citations=policy_citations,
            missing_info=self._identify_missing_context(query, response),
            alternatives_considered=self._generate_llm_alternatives(query, response),
            counterfactuals=[],
            telemetry=TelemetryData(
                latency_ms=processing_time,
                retrieval_k=len(policy_citations),
                triage_time_ms=0,
                planning_time_ms=processing_time,
                total_chunks_considered=len(policy_citations)
            ),
            timestamp=datetime.now(),
            model_version=self.model_name
        )
    
    def _calculate_response_confidence(
        self, 
        response: str, 
        policy_citations: List[PolicyCitation], 
        cot_reasoning: List[Dict]
    ) -> float:
        """Calculate confidence in LLM response"""
        
        confidence = 0.7  # Base confidence
        
        # Boost confidence with policy backing
        if policy_citations:
            avg_policy_relevance = sum(c.relevance_score for c in policy_citations) / len(policy_citations)
            confidence += avg_policy_relevance * 0.2
        
        # Boost confidence with detailed reasoning
        if len(cot_reasoning) >= 3:
            confidence += 0.1
        
        # Reduce confidence for uncertain language
        uncertain_phrases = ["might", "possibly", "maybe", "not sure", "unclear"]
        if any(phrase in response.lower() for phrase in uncertain_phrases):
            confidence -= 0.1
        
        return min(0.95, max(0.3, confidence))
    
    def _identify_missing_context(self, query: str, response: str) -> List[str]:
        """Identify missing information that could improve the response"""
        missing = []
        
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Check for common missing IT context
        if "error" in query_lower and "error code" not in query_lower:
            missing.append("Specific error code or message")
        
        if "network" in query_lower and "ip" not in query_lower:
            missing.append("Network configuration details (IP, subnet, gateway)")
        
        if "software" in query_lower and "version" not in query_lower:
            missing.append("Software version and operating system details")
        
        if "slow" in query_lower and "specifications" not in query_lower:
            missing.append("System specifications (RAM, CPU, storage)")
        
        if "need approval" in response_lower and "manager" not in query_lower:
            missing.append("Manager contact information for approval process")
        
        return missing
    
    def _generate_llm_alternatives(self, query: str, response: str) -> List[Dict[str, Any]]:
        """Generate alternative approaches for the IT issue"""
        alternatives = []
        
        query_lower = query.lower()
        
        # Hardware issues
        if any(hw in query_lower for hw in ["hardware", "computer", "laptop", "broken"]):
            alternatives.append({
                "option": "Remote diagnostic first",
                "pros": ["Faster resolution", "No physical access needed"],
                "cons": ["May not identify all issues", "Limited to software fixes"],
                "confidence": 0.8
            })
        
        # Software issues  
        if any(sw in query_lower for sw in ["software", "application", "program", "install"]):
            alternatives.append({
                "option": "Use virtualization or sandbox",
                "pros": ["Isolated testing", "Reduced security risk"],
                "cons": ["Additional setup required", "May not match production environment"],
                "confidence": 0.7
            })
        
        # Access issues
        if any(access in query_lower for access in ["password", "login", "access", "permission"]):
            alternatives.append({
                "option": "Temporary access with monitoring",
                "pros": ["Immediate access", "Maintains security"],
                "cons": ["Requires manual review", "Short-term solution"],
                "confidence": 0.75
            })
        
        return alternatives
    
    def _handle_invalid_query(self, query: str) -> Tuple[str, ExplanationObject]:
        """Handle queries outside IT support domain"""
        
        # Identify which forbidden topic was detected
        detected_topic = "non-IT topic"
        for pattern in self.forbidden_patterns:
            if re.search(pattern, query):
                match = re.search(pattern, query)
                detected_topic = match.group(0) if match else "non-IT topic"
                break
        
        response = f"""I'm specialized in IT support only. I noticed your question is about {detected_topic}, which is outside my expertise.

I can help you with:
- Computer hardware and software issues
- Network connectivity problems
- System administration tasks
- User access and authentication
- Technical troubleshooting

Please rephrase your question to focus on IT support needs."""
        
        # Create explanation for guardrail activation
        reasoning_trace = [
            ReasoningStep(
                step=1,
                action="domain_validation",
                rationale=f"Query '{query}' detected as non-IT topic: {detected_topic}",
                confidence=0.95,
                policy_refs=[]
            ),
            ReasoningStep(
                step=2,
                action="guardrail_activation",
                rationale="Activated domain restriction guardrails - declined non-IT query",
                confidence=1.0,
                policy_refs=[]
            )
        ]
        
        explanation = ExplanationObject(
            answer=response,
            decision="domain_restricted=true, topic_allowed=false",
            confidence=1.0,
            reasoning_trace=reasoning_trace,
            policy_citations=[],
            missing_info=["IT-related question or request"],
            alternatives_considered=[{
                "option": "Rephrase as IT support question",
                "pros": ["Get proper IT assistance"],
                "cons": ["Requires reformulating request"],
                "confidence": 1.0
            }],
            counterfactuals=[{
                "condition": "If this were an IT support question",
                "outcome": "Would provide detailed technical assistance",
                "likelihood": 1.0
            }],
            telemetry=TelemetryData(
                latency_ms=10,
                retrieval_k=0,
                triage_time_ms=10,
                planning_time_ms=0,
                total_chunks_considered=0
            ),
            timestamp=datetime.now(),
            model_version=self.model_name
        )
        
        return response, explanation
    
    def _handle_model_error(self) -> Tuple[str, ExplanationObject]:
        """Handle LLM model initialization errors"""
        response = "I'm currently unable to process your request due to a technical issue. Please try again later or contact an IT agent directly."
        
        explanation = ExplanationObject(
            answer=response,
            decision="llm_error=true, fallback_activated=true",
            confidence=0.0,
            reasoning_trace=[
                ReasoningStep(
                    step=1,
                    action="model_initialization_error",
                    rationale=f"Failed to initialize {self.model_name} model",
                    confidence=0.0,
                    policy_refs=[]
                )
            ],
            policy_citations=[],
            missing_info=["Functional LLM model"],
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
            model_version=self.model_name
        )
        
        return response, explanation
    
    def _handle_llm_error(self, error_msg: str) -> Tuple[str, ExplanationObject]:
        """Handle LLM generation errors"""
        response = f"I encountered a technical issue while processing your request. Please try rephrasing your question or contact IT support directly."
        
        explanation = ExplanationObject(
            answer=response,
            decision="llm_generation_error=true",
            confidence=0.0,
            reasoning_trace=[
                ReasoningStep(
                    step=1,
                    action="llm_generation_error",
                    rationale=f"LLM generation failed: {error_msg[:100]}",
                    confidence=0.0,
                    policy_refs=[]
                )
            ],
            policy_citations=[],
            missing_info=["Successful LLM inference"],
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
            model_version=self.model_name
        )
        
        return response, explanation
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            models = ollama.list()
            return [model['name'] for model in models.get('models', [])]
        except:
            return []
    
    def search_and_respond(
        self, 
        query: str, 
        context: str = "",
        db: Session = None
    ) -> Tuple[str, ExplanationObject]:
        """
        Generate response with automatic policy search when LLM indicates uncertainty
        """
        
        # First, try to generate response
        response, explanation = self.generate_response(query, context, db, include_cot=True)
        
        # Check if LLM indicates it needs more information
        uncertain_indicators = [
            "i need to search", "check our policies", "i don't know", 
            "need more information", "require additional details"
        ]
        
        if any(indicator in response.lower() for indicator in uncertain_indicators):
            # LLM requested policy search - do enhanced RAG
            enhanced_citations = self.policy_retriever.retrieve_relevant_chunks(
                query, k=5, db=db  # Get more policies
            )
            
            # Retry with more policy context
            response, explanation = self.generate_response(
                query, context, db, include_cot=True
            )
            
            # Update explanation to reflect enhanced search
            explanation.reasoning_trace.append(ReasoningStep(
                step=len(explanation.reasoning_trace) + 1,
                action="enhanced_policy_search",
                rationale=f"LLM requested additional policy search - retrieved {len(enhanced_citations)} additional chunks",
                confidence=0.8,
                policy_refs=[c.chunk_id for c in enhanced_citations]
            ))
        
        return response, explanation
