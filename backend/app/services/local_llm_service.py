"""
Local LLM Service

Fast, optimized local inference using Microsoft Phi-3-mini with quantization,
caching, and guardrails for IT support tasks.
"""

import json
import time
import logging
import hashlib
import threading
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, pipeline
)
from sentence_transformers import SentenceTransformer
import psutil
import redis
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.models import ConversationLog, KnowledgeGraphConcept


logger = logging.getLogger(__name__)


class LocalLLMService:
    """Optimized local LLM service for IT support tasks"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.embedding_model = None
        self.device = self._get_optimal_device()
        self.cache = self._init_cache()
        self.model_lock = threading.Lock()
        self.last_used = datetime.now()
        
        # Model configuration for Phi-3-mini (optimized for your hardware)
        self.model_name = "microsoft/Phi-3-mini-4k-instruct"
        self.max_length = 4096
        self.max_new_tokens = 512
        
        # Performance monitoring
        self.inference_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "avg_inference_time": 0.0,
            "cache_hits": 0
        }
        
        logger.info(f"Initializing LocalLLM on device: {self.device}")
    
    def _get_optimal_device(self) -> str:
        """Determine optimal device for inference"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():  # Apple Silicon
            return "mps"
        else:
            return "cpu"
    
    def _init_cache(self) -> Optional[redis.Redis]:
        """Initialize Redis cache for response caching"""
        try:
            cache = redis.Redis(
                host=getattr(settings, 'REDIS_HOST', 'localhost'),
                port=getattr(settings, 'REDIS_PORT', 6379),
                db=1,  # Use DB 1 for LLM cache
                decode_responses=True
            )
            cache.ping()
            return cache
        except Exception as e:
            logger.warning(f"Redis cache not available: {e}")
            return None
    
    def load_model(self) -> None:
        """Load and optimize the LLM model"""
        if self.model is not None:
            return
        
        with self.model_lock:
            if self.model is not None:  # Double-check after acquiring lock
                return
            
            logger.info(f"Loading {self.model_name} model...")
            start_time = time.time()
            
            # Quantization config for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Move to device if not using device_map
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            # Load embedding model for semantic search
            self.embedding_model = SentenceTransformer(
                'all-MiniLM-L6-v2',  # Fast and efficient
                device=self.device
            )
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s")
            
            # Warm up the model
            self._warmup_model()
    
    def _warmup_model(self) -> None:
        """Warm up the model with a simple inference"""
        try:
            warmup_prompt = "What is VPN?"
            self.generate_response(warmup_prompt, max_tokens=10, use_cache=False)
            logger.info("Model warmed up successfully")
        except Exception as e:
            logger.error(f"Model warmup failed: {e}")
    
    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key for prompt and parameters"""
        cache_data = {
            "prompt": prompt,
            "model": self.model_name,
            **kwargs
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _check_memory_usage(self) -> Dict[str, float]:
        """Monitor memory usage"""
        memory = psutil.virtual_memory()
        return {
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "memory_used_gb": memory.used / (1024**3)
        }
    
    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_cache: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate response from local LLM with caching and optimization
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            use_cache: Whether to use response caching
            context: Additional context for the conversation
            
        Returns:
            Dictionary with response, metadata, and performance stats
        """
        self.load_model()
        
        start_time = time.time()
        cache_key = self._get_cache_key(prompt, max_tokens=max_tokens, temperature=temperature)
        
        # Check cache first
        if use_cache and self.cache:
            try:
                cached_response = self.cache.get(f"llm_response:{cache_key}")
                if cached_response:
                    self.inference_stats["cache_hits"] += 1
                    response_data = json.loads(cached_response)
                    response_data["from_cache"] = True
                    response_data["inference_time_ms"] = 0
                    return response_data
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        # Apply guardrails to prompt
        safe_prompt, safety_flags = self._apply_guardrails(prompt)
        if safety_flags["blocked"]:
            return {
                "response": "I cannot process this request due to safety concerns.",
                "blocked": True,
                "safety_flags": safety_flags,
                "inference_time_ms": 0
            }
        
        # Build full prompt with context
        full_prompt = self._build_contextual_prompt(safe_prompt, context)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length - max_tokens
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    length_penalty=1.0
                )
            
            # Decode response
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Post-process response
            response_text = self._post_process_response(response_text)
            
            inference_time = (time.time() - start_time) * 1000
            
            # Update stats
            self.inference_stats["total_requests"] += 1
            self.inference_stats["total_tokens"] += len(generated_tokens)
            self.inference_stats["avg_inference_time"] = (
                (self.inference_stats["avg_inference_time"] * (self.inference_stats["total_requests"] - 1) + inference_time) 
                / self.inference_stats["total_requests"]
            )
            
            response_data = {
                "response": response_text,
                "blocked": False,
                "safety_flags": safety_flags,
                "inference_time_ms": inference_time,
                "tokens_generated": len(generated_tokens),
                "memory_usage": self._check_memory_usage(),
                "from_cache": False
            }
            
            # Cache the response
            if use_cache and self.cache:
                try:
                    self.cache.setex(
                        f"llm_response:{cache_key}",
                        3600,  # 1 hour TTL
                        json.dumps({k: v for k, v in response_data.items() if k != "memory_usage"})
                    )
                except Exception as e:
                    logger.warning(f"Cache write error: {e}")
            
            self.last_used = datetime.now()
            return response_data
            
        except Exception as e:
            logger.error(f"LLM inference error: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your request.",
                "blocked": False,
                "error": str(e),
                "inference_time_ms": (time.time() - start_time) * 1000
            }
    
    def _apply_guardrails(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Apply safety guardrails to the prompt"""
        safety_flags = {
            "blocked": False,
            "reasons": [],
            "risk_score": 0.0
        }
        
        # Check for obvious harmful patterns
        harmful_patterns = [
            r"(?i)hack|exploit|backdoor|malware|virus",
            r"(?i)delete all|rm -rf|format drive",
            r"(?i)password.*admin|default.*password",
            r"(?i)bypass.*security|disable.*firewall",
            r"(?i)unauthorized.*access|privilege.*escalation"
        ]
        
        import re
        risk_score = 0.0
        for pattern in harmful_patterns:
            if re.search(pattern, prompt):
                risk_score += 0.3
                safety_flags["reasons"].append(f"Potential security risk detected")
        
        # Check for prompt injection attempts
        injection_patterns = [
            r"(?i)ignore.*instructions|forget.*context",
            r"(?i)system.*prompt|override.*behavior",
            r"(?i)jailbreak|unlock.*limitations"
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, prompt):
                risk_score += 0.5
                safety_flags["reasons"].append("Potential prompt injection detected")
        
        safety_flags["risk_score"] = min(risk_score, 1.0)
        
        # Block if risk is too high
        if risk_score > 0.7:
            safety_flags["blocked"] = True
        
        return prompt, safety_flags
    
    def _build_contextual_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build a contextual prompt with IT support framing"""
        
        system_context = """You are an expert IT Support Assistant with deep knowledge of:
- Network infrastructure (VPN, firewall, remote access)
- Security protocols (MFA, authentication, incident response)
- Software and hardware management
- IT policies and compliance requirements

You provide accurate, helpful responses while following security best practices.
Always be concise but thorough in your explanations."""
        
        # Add conversation history if available
        history_context = ""
        if context and "conversation_history" in context:
            recent_history = context["conversation_history"][-3:]  # Last 3 exchanges
            for entry in recent_history:
                history_context += f"\nUser: {entry.get('user_message', '')}\nAssistant: {entry.get('assistant_response', '')}"
        
        # Add relevant policies if available
        policy_context = ""
        if context and "relevant_policies" in context:
            policies = context["relevant_policies"][:2]  # Top 2 most relevant
            for policy in policies:
                policy_context += f"\nRelevant Policy: {policy.get('title', 'Unknown')}\nContent: {policy.get('content', '')[:200]}..."
        
        # Add knowledge graph context if available
        kg_context = ""
        if context and "kg_concepts" in context:
            concepts = context["kg_concepts"][:5]  # Top 5 concepts
            kg_context += f"\nRelated IT Concepts: {', '.join(concepts)}"
        
        full_prompt = f"""{system_context}

{history_context}

{policy_context}

{kg_context}

User Query: {prompt}

Assistant: I'll help you with this IT support request."""
        
        return full_prompt.strip()
    
    def _post_process_response(self, response: str) -> str:
        """Clean and validate the generated response"""
        # Remove common generation artifacts
        response = response.replace("<|endoftext|>", "")
        response = response.replace("<|end|>", "")
        response = response.strip()
        
        # Ensure response isn't too long
        max_response_length = 1000
        if len(response) > max_response_length:
            response = response[:max_response_length] + "..."
        
        return response
    
    def extract_concepts_with_llm(self, text: str, db: Session) -> List[Tuple[str, float, str]]:
        """Extract IT concepts using LLM instead of regex patterns"""
        
        # Get existing concepts for context
        known_concepts = db.query(KnowledgeGraphConcept).all()
        concept_list = [c.name for c in known_concepts[:20]]  # Limit for prompt size
        
        prompt = f"""Extract IT support concepts from this text. Focus on technical terms, systems, and procedures.

Known IT Concepts: {', '.join(concept_list)}

Text to analyze: "{text}"

Return ONLY a JSON array of concepts found, with confidence scores (0.0-1.0):
[{{"concept": "VPN", "confidence": 0.9, "context": "relevant context snippet"}}, ...]

Response:"""
        
        response = self.generate_response(
            prompt,
            max_tokens=256,
            temperature=0.3,  # Lower temperature for structured output
            context={"task": "concept_extraction"}
        )
        
        try:
            # Parse JSON response
            concepts_data = json.loads(response["response"])
            return [(c["concept"], c["confidence"], c["context"]) for c in concepts_data]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse concept extraction response: {e}")
            # Fallback to simple text parsing
            return self._fallback_concept_extraction(text, concept_list)
    
    def extract_relationships_with_llm(self, text: str, concepts: List[str]) -> List[Tuple[str, str, str, float]]:
        """Extract relationships between concepts using LLM"""
        
        if len(concepts) < 2:
            return []
        
        prompt = f"""Analyze the relationships between IT concepts in this text.

Concepts found: {', '.join(concepts)}

Text: "{text}"

Return ONLY a JSON array of relationships:
[{{"source": "VPN", "target": "MFA", "relationship": "requires", "confidence": 0.8}}, ...]

Valid relationship types: requires, depends_on, enables, protects_from, conflicts_with, related_to

Response:"""
        
        response = self.generate_response(
            prompt,
            max_tokens=256,
            temperature=0.3,
            context={"task": "relationship_extraction"}
        )
        
        try:
            relationships_data = json.loads(response["response"])
            return [(r["source"], r["target"], r["relationship"], r["confidence"]) for r in relationships_data]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse relationship extraction response: {e}")
            return []
    
    def _fallback_concept_extraction(self, text: str, known_concepts: List[str]) -> List[Tuple[str, float, str]]:
        """Fallback concept extraction using simple matching"""
        found_concepts = []
        text_lower = text.lower()
        
        for concept in known_concepts:
            if concept.lower() in text_lower:
                # Extract context around the concept
                start = max(0, text_lower.find(concept.lower()) - 30)
                end = min(len(text), start + len(concept) + 60)
                context = text[start:end].strip()
                found_concepts.append((concept, 0.7, context))
        
        return found_concepts
    
    def search_conversation_history(self, query: str, db: Session, limit: int = 5) -> List[Dict[str, Any]]:
        """Search through conversation history using semantic similarity"""
        
        # Get recent conversations
        recent_logs = db.query(ConversationLog).order_by(
            ConversationLog.timestamp.desc()
        ).limit(100).all()
        
        if not recent_logs:
            return []
        
        # Use embedding model for semantic search
        query_embedding = self.embedding_model.encode([query])
        
        # Create embeddings for conversation logs
        conversation_texts = []
        for log in recent_logs:
            text = f"{log.user_message} {log.assistant_response or ''}"
            conversation_texts.append(text)
        
        if not conversation_texts:
            return []
        
        log_embeddings = self.embedding_model.encode(conversation_texts)
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, log_embeddings)[0]
        
        # Get top matches
        top_indices = similarities.argsort()[-limit:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Minimum similarity threshold
                log = recent_logs[idx]
                results.append({
                    "conversation_id": log.id,
                    "user_message": log.user_message,
                    "assistant_response": log.assistant_response,
                    "timestamp": log.timestamp,
                    "similarity_score": float(similarities[idx]),
                    "context_type": "conversation_history"
                })
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        memory_usage = self._check_memory_usage()
        
        return {
            "model_loaded": self.model is not None,
            "model_name": self.model_name,
            "device": self.device,
            "last_used": self.last_used.isoformat(),
            "inference_stats": self.inference_stats,
            "memory_usage": memory_usage,
            "cache_available": self.cache is not None
        }
    
    def unload_model(self) -> None:
        """Unload model to free memory"""
        with self.model_lock:
            if self.model is not None:
                del self.model
                del self.tokenizer
                if self.embedding_model is not None:
                    del self.embedding_model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                self.model = None
                self.tokenizer = None
                self.embedding_model = None
                logger.info("Model unloaded successfully")


# Global instance
local_llm_service = LocalLLMService()
