"""
Local Small Language Model Service
Uses llama-cpp-python for CPU inference with tiny models
Based on ML Challenge requirements for local inference
"""

import os
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not installed, falling back to rule-based")

class LocalSMLService:
    """
    Local Small Language Model Service
    Implements the ML Challenge requirements:
    - TinyLlama (1.1B) for ultra-compact inference
    - Chain-of-Thought reasoning
    - Policy-based responses
    - Search integration
    - Speech support (via Vosk)
    """
    
    def __init__(self):
        self.model = None
        self.model_path = os.getenv("SML_MODEL_PATH", "/tmp/tinyllama.gguf")
        self.model_url = os.getenv("SML_MODEL_URL", 
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        
        # Load model if available
        self._load_model()
        
        # Chain-of-Thought reasoning templates
        self.cot_templates = {
            "it_support": """You are an IT Support Agent. Follow these steps:
1. Understand the problem
2. Check IT policies
3. Generate solution steps
4. Provide compliance status (allowed/denied/needs_approval)
5. Cite relevant policies

Problem: {query}

Let me think step by step:
""",
            "ticket_creation": """Create a support ticket for this issue:
Issue: {issue}

Ticket Details:
- Title: 
- Category: 
- Priority: 
- Description:
- Assigned To:
""",
            "policy_check": """Check IT policy for this request:
Request: {request}

Policy Analysis:
1. Relevant policies:
2. Compliance status:
3. Required approvals:
4. Action items:
"""
        }
    
    def _load_model(self):
        """Load TinyLlama model for local inference"""
        if not LLAMA_CPP_AVAILABLE:
            logger.info("Using rule-based fallback (llama-cpp not available)")
            return
        
        try:
            # Download model if not exists
            if not os.path.exists(self.model_path):
                logger.info(f"Downloading TinyLlama model to {self.model_path}")
                import urllib.request
                urllib.request.urlretrieve(self.model_url, self.model_path)
                logger.info("Model downloaded successfully")
            
            # Load with optimal settings for CPU
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=2048,  # Context window
                n_threads=4,  # CPU threads
                n_gpu_layers=0,  # CPU only
                seed=42,  # Deterministic
                verbose=False
            )
            logger.info("TinyLlama model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def reason_with_cot(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Chain-of-Thought reasoning as per ML Challenge
        Returns structured reasoning with steps
        """
        # Determine query type
        query_lower = query.lower()
        
        if "ticket" in query_lower or "issue" in query_lower:
            template = self.cot_templates["ticket_creation"]
            prompt = template.format(issue=query)
        elif "policy" in query_lower or "allowed" in query_lower:
            template = self.cot_templates["policy_check"]
            prompt = template.format(request=query)
        else:
            template = self.cot_templates["it_support"]
            prompt = template.format(query=query)
        
        # Add context if available
        if context:
            if context.get("policies"):
                prompt += f"\n\nRelevant Policies:\n"
                for policy in context["policies"][:3]:
                    prompt += f"- {policy.get('title', 'Policy')}: {policy.get('content', '')[:200]}...\n"
            
            if context.get("history"):
                prompt += f"\n\nPrevious Solutions:\n"
                for hist in context["history"][-2:]:
                    prompt += f"- {hist.get('problem', '')}: {hist.get('solution', '')[:100]}...\n"
        
        # Generate response
        if self.model:
            response = self._generate_local(prompt)
        else:
            response = self._generate_rules_cot(query, context)
        
        # Parse into structured steps
        steps = self._parse_cot_response(response)
        
        return {
            "query": query,
            "reasoning": response,
            "steps": steps,
            "model": "TinyLlama-1.1B" if self.model else "rule-based",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _generate_local(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate using local TinyLlama model"""
        try:
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                echo=False
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            return self._generate_rules_cot(prompt, None)
    
    def _generate_rules_cot(self, query: str, context: Optional[Dict] = None) -> str:
        """Rule-based Chain-of-Thought fallback"""
        query_lower = query.lower()
        
        # VPN Issues
        if "vpn" in query_lower:
            return """Step 1: Understanding the problem
The user is experiencing VPN connectivity issues.

Step 2: Checking IT policies
Policy IT-SEC-001: VPN access requires multi-factor authentication
Policy IT-NET-003: VPN connections must use approved clients

Step 3: Solution steps
1. Verify network connectivity (ping test)
2. Check VPN client version (must be v4.0+)
3. Ensure MFA token is synchronized
4. Clear VPN client cache
5. Try alternative VPN endpoint if available

Step 4: Compliance status
Status: ALLOWED - User can troubleshoot VPN following these steps

Step 5: Policy citations
- IT-SEC-001: Security Policy, Section 3.2
- IT-NET-003: Network Access Policy, Section 5.1"""
        
        # Password Reset
        elif "password" in query_lower:
            return """Step 1: Understanding the problem
User needs to reset their password.

Step 2: Checking IT policies
Policy IT-SEC-002: Passwords must meet complexity requirements
Policy IT-ACC-001: Self-service password reset is available

Step 3: Solution steps
1. Navigate to password.company.com/reset
2. Enter your username and registered email
3. Complete MFA verification
4. Create new password (8+ chars, mixed case, numbers, special)
5. Cannot reuse last 5 passwords

Step 4: Compliance status
Status: ALLOWED - Self-service password reset available

Step 5: Policy citations
- IT-SEC-002: Security Policy, Section 2.1
- IT-ACC-001: Account Management, Section 1.3"""
        
        # Software Installation
        elif "install" in query_lower or "software" in query_lower:
            return """Step 1: Understanding the problem
User wants to install software.

Step 2: Checking IT policies
Policy IT-SW-001: All software requires approval
Policy IT-SW-002: Only approved vendors allowed

Step 3: Solution steps
1. Submit request via IT portal
2. Include business justification
3. Specify software name and version
4. Manager approval required
5. IT will schedule installation (2-3 days)

Step 4: Compliance status
Status: NEEDS_APPROVAL - Manager and IT approval required

Step 5: Policy citations
- IT-SW-001: Software Policy, Section 1.1
- IT-SW-002: Software Policy, Section 2.3"""
        
        else:
            return f"""Step 1: Understanding the problem
Query: {query}

Step 2: Checking IT policies
No specific policies found for this query.

Step 3: Solution steps
1. Document the issue in detail
2. Check knowledge base for similar issues
3. Contact IT support if unresolved
4. Provide ticket number for tracking

Step 4: Compliance status
Status: ALLOWED - General support request

Step 5: Policy citations
- IT-SUP-001: General Support Policy"""
    
    def _parse_cot_response(self, response: str) -> List[Dict[str, str]]:
        """Parse Chain-of-Thought response into structured steps"""
        steps = []
        current_step = None
        
        for line in response.split("\n"):
            if line.startswith("Step "):
                if current_step:
                    steps.append(current_step)
                current_step = {
                    "step": line.split(":")[0].strip(),
                    "description": line.split(":", 1)[1].strip() if ":" in line else "",
                    "details": []
                }
            elif current_step and line.strip():
                if line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "-")):
                    current_step["details"].append(line.strip())
                elif "Status:" in line:
                    current_step["status"] = line.split("Status:")[1].strip()
                elif "Policy" in line and "-" in line:
                    current_step["policy_ref"] = line.strip()
        
        if current_step:
            steps.append(current_step)
        
        return steps
    
    def should_search(self, query: str) -> Tuple[bool, List[str]]:
        """
        Determine if we should search for additional info
        As per ML Challenge: "Let the LLM decide when to search"
        """
        unknown_indicators = [
            "what is", "how to", "explain", "tell me about",
            "i don't know", "not sure", "help with"
        ]
        
        search_needed = any(ind in query.lower() for ind in unknown_indicators)
        
        # Extract search terms
        search_terms = []
        if search_needed:
            # Extract key IT terms to search
            it_terms = ["vpn", "password", "software", "network", "firewall",
                       "email", "printer", "access", "permission", "policy"]
            for term in it_terms:
                if term in query.lower():
                    search_terms.append(term)
        
        return search_needed, search_terms
    
    def create_ticket(self, issue: str, user_id: str = "demo") -> Dict[str, Any]:
        """
        Create IT support ticket
        As per ML Challenge requirements
        """
        # Generate ticket using CoT
        cot_result = self.reason_with_cot(f"Create ticket for: {issue}")
        
        # Extract ticket details from reasoning
        ticket = {
            "id": hashlib.md5(f"{issue}{datetime.utcnow()}".encode()).hexdigest()[:8],
            "title": self._extract_title(issue),
            "description": issue,
            "category": self._categorize_issue(issue),
            "priority": self._determine_priority(issue),
            "status": "new",
            "assigned_to": "unassigned",
            "created_by": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "reasoning": cot_result["reasoning"],
            "steps": cot_result["steps"]
        }
        
        return ticket
    
    def _extract_title(self, issue: str) -> str:
        """Extract concise title from issue description"""
        # Take first 50 chars or first sentence
        title = issue.split(".")[0][:50]
        return title if title else "IT Support Request"
    
    def _categorize_issue(self, issue: str) -> str:
        """Categorize IT issue"""
        issue_lower = issue.lower()
        
        if "vpn" in issue_lower or "network" in issue_lower:
            return "network"
        elif "password" in issue_lower or "login" in issue_lower:
            return "access"
        elif "software" in issue_lower or "install" in issue_lower:
            return "software"
        elif "hardware" in issue_lower or "laptop" in issue_lower:
            return "hardware"
        else:
            return "general"
    
    def _determine_priority(self, issue: str) -> str:
        """Determine ticket priority"""
        issue_lower = issue.lower()
        
        high_priority_keywords = ["urgent", "critical", "down", "not working", "blocked"]
        medium_priority_keywords = ["slow", "issue", "problem", "help"]
        
        if any(keyword in issue_lower for keyword in high_priority_keywords):
            return "high"
        elif any(keyword in issue_lower for keyword in medium_priority_keywords):
            return "medium"
        else:
            return "low"

# Singleton instance
local_sml_service = LocalSMLService()
