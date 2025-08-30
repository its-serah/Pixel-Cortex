"""
Prompt Engineering Service

Context-aware prompt templates, guardrails, and structured output generation
for the IT Support Agent with safety and quality controls.
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum

from app.services.local_llm_service import local_llm_service


logger = logging.getLogger(__name__)


class PromptTemplate(Enum):
    """Prompt template types for different IT support tasks"""
    CONCEPT_EXTRACTION = "concept_extraction"
    RELATIONSHIP_EXTRACTION = "relationship_extraction"
    POLICY_ANALYSIS = "policy_analysis"
    TROUBLESHOOTING = "troubleshooting"
    TICKET_TRIAGE = "ticket_triage"
    DECISION_MAKING = "decision_making"
    CONVERSATION_SUMMARY = "conversation_summary"
    GENERAL_SUPPORT = "general_support"


class GuardrailLevel(Enum):
    """Security guardrail levels"""
    STRICT = "strict"      # Maximum security, minimal risk tolerance
    MODERATE = "moderate"  # Balanced security and functionality
    RELAXED = "relaxed"    # Minimal restrictions for internal use


class PromptEngineeringService:
    """Service for context-aware prompt engineering and guardrails"""
    
    def __init__(self):
        self.templates = self._load_prompt_templates()
        self.guardrails = self._load_guardrail_rules()
        self.few_shot_examples = self._load_few_shot_examples()
        
    def _load_prompt_templates(self) -> Dict[str, Dict[str, str]]:
        """Load prompt templates for different IT support tasks"""
        
        return {
            PromptTemplate.CONCEPT_EXTRACTION.value: {
                "system": """You are an expert IT analyst. Extract technical concepts, systems, and procedures from IT support text.
Focus on: network components, security protocols, software systems, hardware devices, policies, and procedures.
Always return valid JSON format.""",
                "user": """Extract IT concepts from this text:

Text: "{text}"

Known concepts for reference: {known_concepts}

Return a JSON array with this exact format:
[{{"concept": "VPN", "confidence": 0.9, "context": "relevant snippet from text"}}]

Important: 
- Only include concepts that are actually mentioned in the text
- Confidence should reflect how clearly the concept is mentioned (0.0-1.0)
- Context should be the relevant snippet (max 50 chars)
- Return only the JSON array, no other text

JSON Response:""",
                "output_format": "json_array"
            },
            
            PromptTemplate.RELATIONSHIP_EXTRACTION.value: {
                "system": """You are an expert at analyzing relationships between IT systems and concepts.
Focus on dependencies, requirements, conflicts, and logical connections.""",
                "user": """Analyze relationships between these IT concepts in the given text:

Text: "{text}"
Concepts: {concepts}

Valid relationship types:
- requires: A requires B to function
- depends_on: A depends on B
- enables: A enables B functionality  
- protects_from: A protects against B
- conflicts_with: A conflicts with B
- related_to: A is related to B

Return a JSON array:
[{{"source": "VPN", "target": "MFA", "relationship": "requires", "confidence": 0.8}}]

Only include relationships explicitly mentioned or strongly implied in the text.

JSON Response:""",
                "output_format": "json_array"
            },
            
            PromptTemplate.POLICY_ANALYSIS.value: {
                "system": """You are an IT policy expert. Analyze policy documents for compliance, requirements, and decision guidance.
Focus on security requirements, approval processes, and restrictions.""",
                "user": """Analyze this policy text for decision-making guidance:

Policy: "{policy_text}"
User Request: "{user_request}"

Determine:
1. Does the policy allow, deny, or require approval for this request?
2. What are the key requirements or conditions?
3. What is the confidence level (0.0-1.0)?

Return JSON:
{{"decision": "allowed|denied|needs_approval", "reasoning": "explanation", "requirements": ["req1", "req2"], "confidence": 0.8}}

JSON Response:""",
                "output_format": "json_object"
            },
            
            PromptTemplate.TROUBLESHOOTING.value: {
                "system": """You are an expert IT troubleshooter. Provide systematic, step-by-step solutions for technical problems.
Focus on common causes, diagnostic steps, and escalation paths.""",
                "user": """Help troubleshoot this IT issue:

Issue: {title}
Description: {description}
Category: {category}

Previous similar issues (if any):
{similar_issues}

Provide troubleshooting steps in this format:
1. **Immediate checks**: Quick diagnostics
2. **Common solutions**: Standard fixes to try
3. **Advanced steps**: If basic steps fail
4. **Escalation**: When to escalate to higher support

Be specific and actionable. Reference relevant IT policies when applicable.

Response:""",
                "output_format": "structured_text"
            },
            
            PromptTemplate.TICKET_TRIAGE.value: {
                "system": """You are an IT support triage specialist. Categorize and prioritize support requests accurately and efficiently.
Consider urgency, business impact, and resource requirements.""",
                "user": """Triage this IT support request:

Title: {title}
Description: {description}

User's role: {user_role}
Previous tickets: {previous_context}

Determine:
- Category: network|hardware|software|security|access|other
- Priority: low|medium|high|critical
- Estimated effort: minutes|hours|days
- Requires approval: yes|no

Return JSON:
{{"category": "network", "priority": "medium", "effort": "hours", "requires_approval": false, "reasoning": "explanation"}}

JSON Response:""",
                "output_format": "json_object"
            },
            
            PromptTemplate.GENERAL_SUPPORT.value: {
                "system": """You are a helpful IT Support Assistant. Provide clear, accurate answers to IT questions.
Be professional, concise, and solution-focused. Always prioritize security and best practices.""",
                "user": """{user_query}

Context: {context_summary}

Response:""",
                "output_format": "natural_text"
            }
        }
    
    def _load_guardrail_rules(self) -> Dict[GuardrailLevel, Dict[str, Any]]:
        """Load guardrail rules for different security levels"""
        
        return {
            GuardrailLevel.STRICT: {
                "blocked_patterns": [
                    r"(?i)hack|exploit|crack|bypass.*security",
                    r"(?i)delete.*all|format.*drive|rm.*-rf",
                    r"(?i)disable.*antivirus|turn.*off.*firewall",
                    r"(?i)root.*password|admin.*credentials",
                    r"(?i)backdoor|trojan|keylogger",
                    r"(?i)privilege.*escalation|sudo.*su",
                    r"(?i)unauthorized.*access|breach.*security"
                ],
                "warning_patterns": [
                    r"(?i)install.*unknown.*software",
                    r"(?i)open.*suspicious.*file",
                    r"(?i)click.*unknown.*link",
                    r"(?i)share.*password|give.*access"
                ],
                "max_risk_score": 0.3,
                "require_approval_threshold": 0.2
            },
            
            GuardrailLevel.MODERATE: {
                "blocked_patterns": [
                    r"(?i)hack|exploit|crack",
                    r"(?i)delete.*all|format.*drive",
                    r"(?i)disable.*security|bypass.*protection",
                    r"(?i)backdoor|malware|virus"
                ],
                "warning_patterns": [
                    r"(?i)install.*software",
                    r"(?i)change.*security.*settings",
                    r"(?i)access.*admin|admin.*rights"
                ],
                "max_risk_score": 0.5,
                "require_approval_threshold": 0.4
            },
            
            GuardrailLevel.RELAXED: {
                "blocked_patterns": [
                    r"(?i)hack|exploit|crack",
                    r"(?i)malicious.*code|dangerous.*command"
                ],
                "warning_patterns": [
                    r"(?i)delete.*important|permanent.*changes"
                ],
                "max_risk_score": 0.7,
                "require_approval_threshold": 0.6
            }
        }
    
    def _load_few_shot_examples(self) -> Dict[str, List[Dict[str, str]]]:
        """Load few-shot examples for different tasks"""
        
        return {
            PromptTemplate.CONCEPT_EXTRACTION.value: [
                {
                    "input": "User cannot connect to VPN and MFA authentication is failing",
                    "output": '[{"concept": "VPN", "confidence": 0.9, "context": "cannot connect to VPN"}, {"concept": "MFA", "confidence": 0.8, "context": "MFA authentication is failing"}]'
                },
                {
                    "input": "Need to install new firewall software for network security",
                    "output": '[{"concept": "Firewall", "confidence": 0.9, "context": "install new firewall software"}, {"concept": "Network Security", "confidence": 0.7, "context": "for network security"}]'
                }
            ],
            
            PromptTemplate.RELATIONSHIP_EXTRACTION.value: [
                {
                    "input": "VPN access requires MFA authentication before connection",
                    "concepts": "VPN, MFA",
                    "output": '[{"source": "VPN", "target": "MFA", "relationship": "requires", "confidence": 0.9}]'
                },
                {
                    "input": "Remote desktop depends on network connectivity and firewall rules",
                    "concepts": "Remote Desktop, Network Access, Firewall",
                    "output": '[{"source": "Remote Desktop", "target": "Network Access", "relationship": "depends_on", "confidence": 0.8}, {"source": "Remote Desktop", "target": "Firewall", "relationship": "depends_on", "confidence": 0.7}]'
                }
            ],
            
            PromptTemplate.TROUBLESHOOTING.value: [
                {
                    "issue": "VPN connection fails",
                    "output": """**Immediate checks:**
1. Verify internet connectivity
2. Check VPN client is running
3. Confirm credentials are correct

**Common solutions:**
1. Restart VPN client
2. Try different server location
3. Check firewall settings
4. Update VPN client

**Advanced steps:**
1. Check network adapter settings
2. Flush DNS cache
3. Reset network stack
4. Check for conflicting software

**Escalation:**
Escalate if issue persists after trying all steps or if multiple users affected."""
                }
            ]
        }
    
    def build_prompt(
        self,
        template_type: PromptTemplate,
        context: Dict[str, Any],
        guardrail_level: GuardrailLevel = GuardrailLevel.MODERATE,
        include_examples: bool = True
    ) -> str:
        """
        Build a context-aware prompt with guardrails and examples
        
        Args:
            template_type: Type of prompt template to use
            context: Context variables for the template
            guardrail_level: Security level for guardrails
            include_examples: Whether to include few-shot examples
            
        Returns:
            Complete prompt ready for LLM inference
        """
        
        template = self.templates[template_type.value]
        
        # Build system message
        system_prompt = template["system"]
        
        # Add guardrail instructions
        guardrail_rules = self.guardrails[guardrail_level]
        system_prompt += f"""\\n\\nSecurity Guidelines:
- Never provide instructions for malicious activities
- Always prioritize security and data protection
- Refuse requests that could compromise system security
- Flag suspicious requests for manual review"""
        
        # Build user prompt with context
        user_prompt = template["user"].format(**context)
        
        # Add few-shot examples if requested
        if include_examples and template_type.value in self.few_shot_examples:
            examples = self.few_shot_examples[template_type.value]
            example_text = "\\n\\nExamples:\\n"
            
            for i, example in enumerate(examples[:2]):  # Limit to 2 examples
                example_text += f"\\nExample {i+1}:\\n"
                if "input" in example:
                    example_text += f"Input: {example['input']}\\n"
                if "concepts" in example:
                    example_text += f"Concepts: {example['concepts']}\\n"
                if "issue" in example:
                    example_text += f"Issue: {example['issue']}\\n"
                example_text += f"Output: {example['output']}\\n"
            
            user_prompt = user_prompt + example_text + "\\nNow analyze the actual input:\\n"
        
        # Combine system and user prompts
        full_prompt = f"{system_prompt}\\n\\nUser: {user_prompt}\\n\\nAssistant:"
        
        return full_prompt
    
    def validate_response(
        self,
        response: str,
        expected_format: str,
        guardrail_level: GuardrailLevel = GuardrailLevel.MODERATE
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate LLM response against format and safety requirements
        
        Args:
            response: Generated response to validate
            expected_format: Expected output format (json_array, json_object, etc.)
            guardrail_level: Security level for validation
            
        Returns:
            Tuple of (is_valid, validation_details)
        """
        
        validation_result = {
            "format_valid": False,
            "safety_valid": False,
            "errors": [],
            "warnings": [],
            "parsed_data": None
        }
        
        # Validate format
        if expected_format == "json_array":
            try:
                parsed = json.loads(response)
                if isinstance(parsed, list):
                    validation_result["format_valid"] = True
                    validation_result["parsed_data"] = parsed
                else:
                    validation_result["errors"].append("Response is not a JSON array")
            except json.JSONDecodeError as e:
                validation_result["errors"].append(f"Invalid JSON: {e}")
        
        elif expected_format == "json_object":
            try:
                parsed = json.loads(response)
                if isinstance(parsed, dict):
                    validation_result["format_valid"] = True
                    validation_result["parsed_data"] = parsed
                else:
                    validation_result["errors"].append("Response is not a JSON object")
            except json.JSONDecodeError as e:
                validation_result["errors"].append(f"Invalid JSON: {e}")
        
        elif expected_format in ["natural_text", "structured_text"]:
            if response.strip():
                validation_result["format_valid"] = True
                validation_result["parsed_data"] = response
            else:
                validation_result["errors"].append("Empty response")
        
        # Validate safety
        guardrail_rules = self.guardrails[guardrail_level]
        risk_score = 0.0
        
        # Check for blocked patterns
        for pattern in guardrail_rules["blocked_patterns"]:
            if re.search(pattern, response):
                validation_result["errors"].append(f"Response contains blocked content")
                risk_score += 0.5
        
        # Check for warning patterns
        for pattern in guardrail_rules["warning_patterns"]:
            if re.search(pattern, response):
                validation_result["warnings"].append(f"Response contains potentially risky content")
                risk_score += 0.2
        
        # Determine safety validity
        validation_result["safety_valid"] = risk_score <= guardrail_rules["max_risk_score"]
        validation_result["risk_score"] = min(risk_score, 1.0)
        
        is_valid = validation_result["format_valid"] and validation_result["safety_valid"]
        
        return is_valid, validation_result
    
    def generate_safe_response(
        self,
        template_type: PromptTemplate,
        context: Dict[str, Any],
        guardrail_level: GuardrailLevel = GuardrailLevel.MODERATE,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Generate a safe, validated response with automatic retries
        
        Args:
            template_type: Type of prompt template
            context: Context for prompt generation
            guardrail_level: Security level
            max_retries: Maximum retry attempts for validation failures
            
        Returns:
            Generated response with validation metadata
        """
        
        template = self.templates[template_type.value]
        expected_format = template["output_format"]
        
        for attempt in range(max_retries + 1):
            try:
                # Build prompt
                prompt = self.build_prompt(
                    template_type,
                    context,
                    guardrail_level,
                    include_examples=(attempt == 0)  # Include examples on first attempt
                )
                
                # Generate response
                llm_response = local_llm_service.generate_response(
                    prompt,
                    max_tokens=512 if expected_format != "structured_text" else 1024,
                    temperature=0.3 if expected_format.startswith("json") else 0.7,
                    context={"task": template_type.value}
                )
                
                if llm_response.get("blocked", False):
                    return {
                        "success": False,
                        "response": None,
                        "error": "Request blocked by LLM safety filters",
                        "attempt": attempt + 1,
                        "safety_flags": llm_response.get("safety_flags", {})
                    }
                
                generated_text = llm_response["response"]
                
                # Validate response
                is_valid, validation_details = self.validate_response(
                    generated_text,
                    expected_format,
                    guardrail_level
                )
                
                if is_valid:
                    return {
                        "success": True,
                        "response": validation_details["parsed_data"],
                        "raw_response": generated_text,
                        "validation": validation_details,
                        "llm_metadata": llm_response,
                        "attempt": attempt + 1,
                        "template_used": template_type.value
                    }
                else:
                    logger.warning(f"Validation failed on attempt {attempt + 1}: {validation_details['errors']}")
                    
                    # If last attempt, return with error
                    if attempt == max_retries:
                        return {
                            "success": False,
                            "response": None,
                            "error": "Response validation failed",
                            "validation": validation_details,
                            "raw_response": generated_text,
                            "attempt": attempt + 1
                        }
            
            except Exception as e:
                logger.error(f"Prompt generation error on attempt {attempt + 1}: {e}")
                
                if attempt == max_retries:
                    return {
                        "success": False,
                        "response": None,
                        "error": str(e),
                        "attempt": attempt + 1
                    }
        
        return {
            "success": False,
            "response": None,
            "error": "Maximum retries exceeded",
            "attempt": max_retries + 1
        }
    
    def extract_structured_data(
        self,
        text: str,
        data_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from text using appropriate prompts
        
        Args:
            text: Text to analyze
            data_type: Type of data to extract (concepts, relationships, etc.)
            context: Additional context for extraction
            
        Returns:
            Extracted structured data
        """
        
        if data_type == "concepts":
            template_type = PromptTemplate.CONCEPT_EXTRACTION
            prompt_context = {
                "text": text,
                "known_concepts": ", ".join(context.get("known_concepts", []))
            }
        
        elif data_type == "relationships":
            template_type = PromptTemplate.RELATIONSHIP_EXTRACTION
            prompt_context = {
                "text": text,
                "concepts": ", ".join(context.get("concepts", []))
            }
        
        else:
            return {"error": f"Unsupported data type: {data_type}"}
        
        return self.generate_safe_response(
            template_type,
            prompt_context,
            GuardrailLevel.MODERATE
        )
    
    def analyze_user_intent(
        self,
        user_message: str,
        conversation_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze user intent to determine appropriate response strategy
        
        Args:
            user_message: Current user message
            conversation_context: Recent conversation history
            
        Returns:
            Intent analysis with recommended actions
        """
        
        # Build context from conversation history
        context_text = ""
        if conversation_context:
            recent_exchanges = conversation_context[-3:]  # Last 3 exchanges
            for exchange in recent_exchanges:
                context_text += f"User: {exchange['user_message']}\\nAssistant: {exchange['assistant_response']}\\n"
        
        intent_prompt = f"""Analyze the user's intent in this IT support conversation:

Previous context:
{context_text}

Current message: "{user_message}"

Determine:
1. Primary intent: question|request|problem|follow_up|clarification
2. Urgency level: low|medium|high|critical
3. Complexity: simple|moderate|complex
4. Required action: answer|investigate|escalate|create_ticket

Return JSON:
{{"intent": "problem", "urgency": "medium", "complexity": "moderate", "action": "investigate", "reasoning": "explanation"}}

JSON Response:"""
        
        try:
            response = local_llm_service.generate_response(
                intent_prompt,
                max_tokens=256,
                temperature=0.3,
                context={"task": "intent_analysis"}
            )
            
            intent_data = json.loads(response["response"])
            intent_data["confidence"] = 0.8  # Default confidence
            return intent_data
            
        except Exception as e:
            logger.warning(f"Intent analysis failed: {e}")
            return {
                "intent": "question",
                "urgency": "medium",
                "complexity": "moderate",
                "action": "answer",
                "confidence": 0.5,
                "error": str(e)
            }
    
    def get_prompt_statistics(self) -> Dict[str, Any]:
        """Get prompt engineering statistics and performance metrics"""
        
        return {
            "available_templates": list(self.templates.keys()),
            "guardrail_levels": [level.value for level in GuardrailLevel],
            "template_count": len(self.templates),
            "few_shot_examples": {
                template: len(examples) 
                for template, examples in self.few_shot_examples.items()
            },
            "last_updated": datetime.now().isoformat()
        }


# Global instance
prompt_engineering_service = PromptEngineeringService()
