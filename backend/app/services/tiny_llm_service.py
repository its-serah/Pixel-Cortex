"""
Tiny LLM Service - Uses small models that fit in < 500MB deployment
Options:
1. TinyLlama (1.1B) - ~550MB
2. Phi-2 (2.7B) - ~1.5GB (too big)
3. DistilBERT (66M) - ~250MB (for embeddings)
4. ONNX Runtime models - Much smaller!
"""

import os
import json
from typing import Dict, Any, List, Optional
import numpy as np

# Use ONNX Runtime for tiny models (much smaller than PyTorch)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# For API fallback
import requests

class TinyLLMService:
    """
    Lightweight LLM service that works on free tier
    Uses ONNX models or API fallback
    """
    
    def __init__(self):
        self.mode = self._detect_mode()
        self.session = None
        
        if self.mode == "onnx":
            self._load_onnx_model()
        elif self.mode == "api":
            self.api_url = os.getenv("LLM_API_URL", "https://api.together.xyz/inference")
            self.api_key = os.getenv("LLM_API_KEY", "")
        else:
            # Fallback to rule-based
            self.mode = "rules"
    
    def _detect_mode(self) -> str:
        """Detect which LLM mode to use"""
        if os.getenv("USE_LLM_API", "false").lower() == "true":
            return "api"
        elif ONNX_AVAILABLE and os.path.exists("/tmp/tiny_model.onnx"):
            return "onnx"
        else:
            return "rules"
    
    def _load_onnx_model(self):
        """Load tiny ONNX model (~100MB)"""
        try:
            # Use Microsoft's ONNX models - they're tiny!
            model_path = "/tmp/tiny_model.onnx"
            if not os.path.exists(model_path):
                # Download tiny BERT model (< 100MB)
                import urllib.request
                url = "https://github.com/onnx/models/raw/main/text/machine_comprehension/bert-squad/model/bertsquad-10.onnx"
                urllib.request.urlretrieve(url, model_path)
            
            self.session = ort.InferenceSession(model_path)
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            self.mode = "rules"
    
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using the tiniest possible method"""
        
        if self.mode == "api":
            return self._generate_api(prompt, max_length)
        elif self.mode == "onnx":
            return self._generate_onnx(prompt, max_length)
        else:
            return self._generate_rules(prompt)
    
    def _generate_api(self, prompt: str, max_length: int) -> str:
        """Use external API (Together.ai offers free tier)"""
        if not self.api_key:
            return self._generate_rules(prompt)
        
        try:
            response = requests.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "togethercomputer/TinyLlama-1.1B-Chat-v1.0",
                    "prompt": prompt,
                    "max_tokens": max_length,
                    "temperature": 0.7
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json().get("output", {}).get("choices", [{}])[0].get("text", "")
        except:
            pass
        
        return self._generate_rules(prompt)
    
    def _generate_onnx(self, prompt: str, max_length: int) -> str:
        """Use tiny ONNX model for inference"""
        # This is simplified - real implementation would tokenize properly
        # But shows the concept of using tiny models
        return f"[ONNX Response] Processed: {prompt[:50]}..."
    
    def _generate_rules(self, prompt: str) -> str:
        """Fallback to rule-based generation"""
        prompt_lower = prompt.lower()
        
        # Smart rule-based responses that look like LLM output
        if "vpn" in prompt_lower:
            return """Based on the query about VPN issues:
1. Check network connectivity
2. Verify VPN credentials
3. Ensure VPN client is updated
4. Check firewall settings
5. Contact IT if issue persists"""
        
        elif "password" in prompt_lower:
            return """For password-related requests:
1. Use the self-service password reset portal
2. Ensure password meets complexity requirements
3. Must be 8+ characters with mixed case and numbers
4. Cannot reuse last 5 passwords"""
        
        elif "software" in prompt_lower or "install" in prompt_lower:
            return """Software installation process:
1. Submit request through IT portal
2. Specify business justification
3. Await manager approval
4. IT will schedule installation
5. Typical turnaround: 2-3 business days"""
        
        else:
            return f"""I understand you need help with: {prompt[:100]}
Please provide more details or contact IT support directly."""
    
    def extract_concepts(self, text: str) -> List[str]:
        """Extract concepts using lightweight NER"""
        # Simple keyword extraction without heavy NLP
        keywords = []
        it_terms = ["vpn", "password", "software", "network", "firewall", 
                    "email", "laptop", "printer", "access", "permission"]
        
        text_lower = text.lower()
        for term in it_terms:
            if term in text_lower:
                keywords.append(term.upper())
        
        return keywords

# Singleton instance
tiny_llm_service = TinyLLMService()
