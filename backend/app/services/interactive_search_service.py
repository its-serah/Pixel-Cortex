"""
Interactive Search and History Service

Provides LLM-enhanced search capabilities across conversation logs,
ticket history, and knowledge base with context-aware retrieval.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, text

from app.core.database import get_db
from app.models.models import Ticket, User, Policy
from app.services.local_llm_service import local_llm_service
from app.services.conversation_memory_service import conversation_memory_service
from app.services.prompt_engineering_service import prompt_service
from app.services.kg_builder import KnowledgeGraphBuilder


logger = logging.getLogger(__name__)


class SearchType:
    """Types of search operations"""
    CONVERSATION = "conversation"
    TICKETS = "tickets"
    POLICIES = "policies"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    UNIFIED = "unified"


class InteractiveSearchService:
    """Service for LLM-enhanced interactive search and history retrieval"""
    
    def __init__(self):
        self.kg_builder = KnowledgeGraphBuilder()
        
    async def intelligent_search(
        self,
        query: str,
        search_types: List[str] = None,
        user_id: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Perform intelligent search across multiple data sources using LLM
        """
        
        if search_types is None:
            search_types = [SearchType.UNIFIED]
        
        # Enhance query using LLM
        enhanced_query = await self._enhance_search_query(query)
        
        search_results = {
            "original_query": query,
            "enhanced_query": enhanced_query,
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        # Perform search across requested types
        for search_type in search_types:
            try:
                if search_type == SearchType.CONVERSATION:
                    results = await self._search_conversations(enhanced_query, user_id, filters, limit)
                elif search_type == SearchType.TICKETS:
                    results = await self._search_tickets(enhanced_query, user_id, filters, limit)
                elif search_type == SearchType.POLICIES:
                    results = await self._search_policies(enhanced_query, filters, limit)
                elif search_type == SearchType.KNOWLEDGE_GRAPH:
                    results = await self._search_knowledge_graph(enhanced_query, filters, limit)
                elif search_type == SearchType.UNIFIED:
                    results = await self._unified_search(enhanced_query, user_id, filters, limit)
                else:
                    continue
                
                search_results["results"][search_type] = results
                
            except Exception as e:
                logger.error(f"Search failed for type {search_type}: {e}")
                search_results["results"][search_type] = {
                    "error": str(e),
                    "items": []
                }
        
        # Generate intelligent summary
        search_results["summary"] = await self._generate_search_summary(search_results)
        
        return search_results
    
    async def _enhance_search_query(self, query: str) -> Dict[str, Any]:
        """Enhance search query using LLM for better retrieval"""
        
        try:
            prompt = prompt_service.build_prompt(
                "search_enhancement",
                query=query,
                context={
                    "instruction": "Analyze this search query and suggest improvements for better retrieval"
                }
            )
            
            response = local_llm_service.generate_response(
                prompt,
                max_tokens=200,
                temperature=0.3
            )
            
            if response.get("error"):
                # Fallback to original query if enhancement fails
                return {
                    "original": query,
                    "enhanced": query,
                    "keywords": self._extract_keywords(query),
                    "intent": "unknown"
                }
            
            # Parse LLM response for enhanced query components
            enhanced_text = response.get("response", query)
            
            return {
                "original": query,
                "enhanced": enhanced_text,
                "keywords": self._extract_keywords(enhanced_text),
                "intent": self._detect_search_intent(query),
                "suggestions": self._generate_search_suggestions(query)
            }
            
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}")
            return {
                "original": query,
                "enhanced": query,
                "keywords": self._extract_keywords(query),
                "intent": "unknown"
            }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from search text"""
        
        # Simple keyword extraction - can be enhanced with NLP
        import re
        
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "through", "during",
            "before", "after", "above", "below", "between", "among", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may", "might", "must", "can"
        }
        
        # Extract words and filter stop words
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]  # Return top 10 keywords
    
    def _detect_search_intent(self, query: str) -> str:
        """Detect the intent of the search query"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["error", "issue", "problem", "bug", "fail"]):
            return "troubleshooting"
        elif any(word in query_lower for word in ["policy", "rule", "procedure", "compliance"]):
            return "policy_lookup"
        elif any(word in query_lower for word in ["how", "what", "why", "when", "where"]):
            return "information_request"
        elif any(word in query_lower for word in ["history", "past", "previous", "before"]):
            return "historical_lookup"
        else:
            return "general_search"
    
    def _generate_search_suggestions(self, query: str) -> List[str]:
        """Generate search suggestions based on query"""
        
        suggestions = []
        intent = self._detect_search_intent(query)
        
        if intent == "troubleshooting":
            suggestions.extend([
                f"{query} solution",
                f"{query} steps",
                f"how to fix {query}"
            ])
        elif intent == "policy_lookup":
            suggestions.extend([
                f"{query} documentation",
                f"{query} requirements",
                f"{query} compliance"
            ])
        elif intent == "information_request":
            suggestions.extend([
                f"{query} guide",
                f"{query} tutorial",
                f"{query} best practices"
            ])
        
        return suggestions[:5]
    
    async def _search_conversations(
        self,
        enhanced_query: Dict[str, Any],
        user_id: Optional[int],
        filters: Optional[Dict[str, Any]],
        limit: int
    ) -> Dict[str, Any]:
        """Search conversation history using semantic similarity"""
        
        try:
            # Use conversation memory service for semantic search
            query_text = enhanced_query.get("enhanced", enhanced_query.get("original", ""))
            
            search_results = await conversation_memory_service.search_conversations(
                query=query_text,
                user_id=user_id,
                limit=limit,
                filters=filters
            )
            
            # Enhance results with LLM analysis
            enhanced_results = []
            for result in search_results.get("conversations", []):
                # Add relevance scoring and highlighting
                relevance_score = await self._calculate_relevance_score(
                    query_text, 
                    result.get("content", "")
                )
                
                result["relevance_score"] = relevance_score
                result["highlights"] = self._extract_highlights(
                    query_text, 
                    result.get("content", "")
                )
                
                enhanced_results.append(result)
            
            # Sort by relevance
            enhanced_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            return {
                "total_found": len(enhanced_results),
                "items": enhanced_results[:limit],
                "search_metadata": {
                    "query_enhanced": True,
                    "semantic_search": True,
                    "relevance_scored": True
                }
            }
            
        except Exception as e:
            logger.error(f"Conversation search failed: {e}")
            return {"error": str(e), "items": []}
    
    async def _search_tickets(
        self,
        enhanced_query: Dict[str, Any],
        user_id: Optional[int],
        filters: Optional[Dict[str, Any]],
        limit: int
    ) -> Dict[str, Any]:
        """Search ticket history with LLM-enhanced filtering"""
        
        try:
            db = next(get_db())
            
            # Build base query
            query_builder = db.query(Ticket)
            
            # Apply user filter
            if user_id:
                query_builder = query_builder.filter(
                    or_(Ticket.assigned_to == user_id, Ticket.created_by == user_id)
                )
            
            # Apply date filters
            if filters:
                if "start_date" in filters:
                    query_builder = query_builder.filter(
                        Ticket.created_at >= datetime.fromisoformat(filters["start_date"])
                    )
                if "end_date" in filters:
                    query_builder = query_builder.filter(
                        Ticket.created_at <= datetime.fromisoformat(filters["end_date"])
                    )
                if "status" in filters:
                    query_builder = query_builder.filter(Ticket.status == filters["status"])
                if "priority" in filters:
                    query_builder = query_builder.filter(Ticket.priority == filters["priority"])
            
            # Get tickets for semantic search
            tickets = query_builder.order_by(desc(Ticket.created_at)).limit(limit * 3).all()
            
            # Perform semantic search on ticket content
            query_text = enhanced_query.get("enhanced", enhanced_query.get("original", ""))
            relevant_tickets = []
            
            for ticket in tickets:
                # Create searchable content from ticket
                ticket_content = f"{ticket.title} {ticket.description}"
                if ticket.resolution:
                    ticket_content += f" {ticket.resolution}"
                
                # Calculate relevance
                relevance_score = await self._calculate_relevance_score(query_text, ticket_content)
                
                if relevance_score > 0.3:  # Threshold for relevance
                    ticket_dict = {
                        "id": ticket.id,
                        "title": ticket.title,
                        "description": ticket.description,
                        "status": ticket.status,
                        "priority": ticket.priority,
                        "created_at": ticket.created_at.isoformat(),
                        "updated_at": ticket.updated_at.isoformat() if ticket.updated_at else None,
                        "resolution": ticket.resolution,
                        "relevance_score": relevance_score,
                        "highlights": self._extract_highlights(query_text, ticket_content)
                    }
                    relevant_tickets.append(ticket_dict)
            
            # Sort by relevance
            relevant_tickets.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return {
                "total_found": len(relevant_tickets),
                "items": relevant_tickets[:limit],
                "search_metadata": {
                    "semantic_search": True,
                    "relevance_threshold": 0.3,
                    "total_searched": len(tickets)
                }
            }
            
        except Exception as e:
            logger.error(f"Ticket search failed: {e}")
            return {"error": str(e), "items": []}
    
    async def _search_policies(
        self,
        enhanced_query: Dict[str, Any],
        filters: Optional[Dict[str, Any]],
        limit: int
    ) -> Dict[str, Any]:
        """Search policy documents with LLM analysis"""
        
        try:
            db = next(get_db())
            
            # Build policy query
            query_builder = db.query(Policy)
            
            # Apply filters
            if filters:
                if "category" in filters:
                    query_builder = query_builder.filter(Policy.category == filters["category"])
                if "is_active" in filters:
                    query_builder = query_builder.filter(Policy.is_active == filters["is_active"])
            
            policies = query_builder.all()
            
            # Perform semantic search on policies
            query_text = enhanced_query.get("enhanced", enhanced_query.get("original", ""))
            relevant_policies = []
            
            for policy in policies:
                # Create searchable content
                policy_content = f"{policy.title} {policy.content}"
                if policy.description:
                    policy_content += f" {policy.description}"
                
                # Calculate relevance
                relevance_score = await self._calculate_relevance_score(query_text, policy_content)
                
                if relevance_score > 0.2:  # Lower threshold for policies
                    policy_dict = {
                        "id": policy.id,
                        "title": policy.title,
                        "description": policy.description,
                        "category": policy.category,
                        "is_active": policy.is_active,
                        "created_at": policy.created_at.isoformat(),
                        "relevance_score": relevance_score,
                        "highlights": self._extract_highlights(query_text, policy_content),
                        "compliance_level": await self._analyze_policy_compliance(policy_content, query_text)
                    }
                    relevant_policies.append(policy_dict)
            
            # Sort by relevance
            relevant_policies.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return {
                "total_found": len(relevant_policies),
                "items": relevant_policies[:limit],
                "search_metadata": {
                    "semantic_search": True,
                    "compliance_analyzed": True,
                    "total_searched": len(policies)
                }
            }
            
        except Exception as e:
            logger.error(f"Policy search failed: {e}")
            return {"error": str(e), "items": []}
    
    async def _search_knowledge_graph(
        self,
        enhanced_query: Dict[str, Any],
        filters: Optional[Dict[str, Any]],
        limit: int
    ) -> Dict[str, Any]:
        """Search knowledge graph using LLM-enhanced graph traversal"""
        
        try:
            query_text = enhanced_query.get("enhanced", enhanced_query.get("original", ""))
            
            # Extract concepts from query using LLM
            concepts = await self._extract_search_concepts(query_text)
            
            # Search knowledge graph
            kg_results = self.kg_builder.search_related_concepts(
                concepts,
                max_depth=2,
                limit=limit
            )
            
            # Enhance with LLM reasoning
            enhanced_kg_results = []
            for result in kg_results:
                # Analyze concept relevance
                relevance_explanation = await self._explain_concept_relevance(
                    query_text,
                    result
                )
                
                result["relevance_explanation"] = relevance_explanation
                result["search_confidence"] = await self._calculate_search_confidence(
                    query_text,
                    result
                )
                
                enhanced_kg_results.append(result)
            
            return {
                "total_found": len(enhanced_kg_results),
                "items": enhanced_kg_results[:limit],
                "search_metadata": {
                    "concepts_extracted": concepts,
                    "graph_traversal": True,
                    "llm_enhanced": True
                }
            }
            
        except Exception as e:
            logger.error(f"Knowledge graph search failed: {e}")
            return {"error": str(e), "items": []}
    
    async def _unified_search(
        self,
        enhanced_query: Dict[str, Any],
        user_id: Optional[int],
        filters: Optional[Dict[str, Any]],
        limit: int
    ) -> Dict[str, Any]:
        """Perform unified search across all data sources"""
        
        # Search all sources
        conversations = await self._search_conversations(enhanced_query, user_id, filters, limit // 3)
        tickets = await self._search_tickets(enhanced_query, user_id, filters, limit // 3)
        policies = await self._search_policies(enhanced_query, filters, limit // 3)
        kg_results = await self._search_knowledge_graph(enhanced_query, filters, limit // 3)
        
        # Combine and rank results
        all_results = []
        
        # Add conversation results
        for item in conversations.get("items", []):
            item["source_type"] = "conversation"
            all_results.append(item)
        
        # Add ticket results
        for item in tickets.get("items", []):
            item["source_type"] = "ticket"
            all_results.append(item)
        
        # Add policy results
        for item in policies.get("items", []):
            item["source_type"] = "policy"
            all_results.append(item)
        
        # Add knowledge graph results
        for item in kg_results.get("items", []):
            item["source_type"] = "knowledge_graph"
            all_results.append(item)
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Generate unified insights
        unified_insights = await self._generate_unified_insights(
            enhanced_query.get("enhanced", enhanced_query.get("original", "")),
            all_results[:limit]
        )
        
        return {
            "total_found": len(all_results),
            "items": all_results[:limit],
            "source_breakdown": {
                "conversations": len(conversations.get("items", [])),
                "tickets": len(tickets.get("items", [])),
                "policies": len(policies.get("items", [])),
                "knowledge_graph": len(kg_results.get("items", []))
            },
            "unified_insights": unified_insights,
            "search_metadata": {
                "unified_search": True,
                "cross_source_ranking": True
            }
        }
    
    async def _calculate_relevance_score(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content using LLM"""
        
        try:
            prompt = prompt_service.build_prompt(
                "relevance_scoring",
                query=query,
                content=content[:1000],  # Limit content length
                context={
                    "instruction": "Score the relevance of this content to the query on a scale of 0.0 to 1.0"
                }
            )
            
            response = local_llm_service.generate_response(
                prompt,
                max_tokens=50,
                temperature=0.1
            )
            
            if response.get("error"):
                # Fallback to simple keyword matching
                return self._simple_relevance_score(query, content)
            
            # Extract score from response
            response_text = response.get("response", "0.0")
            try:
                # Look for decimal number in response
                import re
                score_match = re.search(r'(\d+\.?\d*)', response_text)
                if score_match:
                    score = float(score_match.group(1))
                    return min(1.0, max(0.0, score))  # Clamp between 0 and 1
            except:
                pass
            
            return self._simple_relevance_score(query, content)
            
        except Exception as e:
            logger.warning(f"LLM relevance scoring failed: {e}")
            return self._simple_relevance_score(query, content)
    
    def _simple_relevance_score(self, query: str, content: str) -> float:
        """Simple keyword-based relevance scoring as fallback"""
        
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        return len(intersection) / len(query_words)
    
    def _extract_highlights(self, query: str, content: str, max_highlights: int = 3) -> List[str]:
        """Extract relevant highlights from content based on query"""
        
        import re
        
        query_words = [word.lower() for word in query.split() if len(word) > 2]
        content_lower = content.lower()
        
        highlights = []
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            # Check if sentence contains query words
            sentence_lower = sentence.lower()
            matches = sum(1 for word in query_words if word in sentence_lower)
            
            if matches > 0:
                highlights.append({
                    "text": sentence,
                    "match_count": matches,
                    "confidence": matches / len(query_words)
                })
        
        # Sort by match count and return top highlights
        highlights.sort(key=lambda x: x["match_count"], reverse=True)
        return highlights[:max_highlights]
    
    async def _extract_search_concepts(self, query: str) -> List[str]:
        """Extract key concepts from search query for knowledge graph traversal"""
        
        try:
            prompt = prompt_service.build_prompt(
                "concept_extraction",
                text=query,
                context={
                    "instruction": "Extract key IT concepts and terms from this search query"
                }
            )
            
            response = local_llm_service.generate_response(
                prompt,
                max_tokens=100,
                temperature=0.2
            )
            
            if response.get("error"):
                # Fallback to keyword extraction
                return self._extract_keywords(query)
            
            # Parse concepts from response
            response_text = response.get("response", "")
            concepts = [
                concept.strip() 
                for concept in response_text.split(",")
                if concept.strip()
            ]
            
            return concepts[:10]
            
        except Exception as e:
            logger.warning(f"Concept extraction failed: {e}")
            return self._extract_keywords(query)
    
    async def _explain_concept_relevance(self, query: str, kg_result: Dict[str, Any]) -> str:
        """Generate explanation for why a knowledge graph concept is relevant"""
        
        try:
            prompt = prompt_service.build_prompt(
                "relevance_explanation",
                query=query,
                concept=kg_result.get("concept", ""),
                relationships=kg_result.get("relationships", []),
                context={
                    "instruction": "Explain why this concept is relevant to the search query"
                }
            )
            
            response = local_llm_service.generate_response(
                prompt,
                max_tokens=100,
                temperature=0.3
            )
            
            return response.get("response", "Potentially relevant concept")
            
        except Exception as e:
            logger.warning(f"Relevance explanation failed: {e}")
            return "Relevance analysis unavailable"
    
    async def _calculate_search_confidence(self, query: str, result: Dict[str, Any]) -> float:
        """Calculate confidence score for search result"""
        
        try:
            # Simple confidence based on multiple factors
            confidence = 0.0
            
            # Relevance score contribution
            relevance = result.get("relevance_score", 0)
            confidence += relevance * 0.6
            
            # Relationship count contribution (for KG results)
            relationships = result.get("relationships", [])
            if relationships:
                confidence += min(0.3, len(relationships) * 0.1)
            
            # Recency contribution
            if "created_at" in result:
                try:
                    created_at = datetime.fromisoformat(result["created_at"])
                    days_old = (datetime.now() - created_at).days
                    recency_score = max(0, 1 - (days_old / 365))  # Decay over a year
                    confidence += recency_score * 0.1
                except:
                    pass
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5
    
    async def _analyze_policy_compliance(self, policy_content: str, query: str) -> str:
        """Analyze policy compliance level related to query"""
        
        try:
            prompt = prompt_service.build_prompt(
                "compliance_analysis",
                policy_content=policy_content[:500],  # Limit content
                query=query,
                context={
                    "instruction": "Analyze if this policy addresses the query and determine compliance level"
                }
            )
            
            response = local_llm_service.generate_response(
                prompt,
                max_tokens=50,
                temperature=0.2
            )
            
            return response.get("response", "Compliance analysis unavailable")
            
        except Exception as e:
            logger.warning(f"Compliance analysis failed: {e}")
            return "Unknown"
    
    async def _generate_search_summary(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligent summary of search results"""
        
        try:
            # Count total results across all sources
            total_results = 0
            source_counts = {}
            
            for source_type, results in search_results.get("results", {}).items():
                if isinstance(results, dict) and "items" in results:
                    count = len(results["items"])
                    total_results += count
                    source_counts[source_type] = count
            
            # Generate insights using LLM
            if total_results > 0:
                prompt = prompt_service.build_prompt(
                    "search_summary",
                    query=search_results.get("original_query", ""),
                    total_results=total_results,
                    source_counts=source_counts,
                    context={
                        "instruction": "Summarize the search results and provide insights"
                    }
                )
                
                response = local_llm_service.generate_response(
                    prompt,
                    max_tokens=150,
                    temperature=0.4
                )
                
                summary_text = response.get("response", "Search completed successfully")
            else:
                summary_text = "No relevant results found for the search query"
            
            return {
                "total_results": total_results,
                "source_breakdown": source_counts,
                "summary": summary_text,
                "search_quality": "high" if total_results > 5 else "medium" if total_results > 0 else "low"
            }
            
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return {
                "total_results": 0,
                "summary": "Search summary unavailable",
                "error": str(e)
            }
    
    async def _generate_unified_insights(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights from unified search results"""
        
        try:
            # Analyze result patterns
            source_types = {}
            avg_relevance = 0
            
            for result in results:
                source_type = result.get("source_type", "unknown")
                source_types[source_type] = source_types.get(source_type, 0) + 1
                avg_relevance += result.get("relevance_score", 0)
            
            if results:
                avg_relevance /= len(results)
            
            # Generate insights with LLM
            prompt = prompt_service.build_prompt(
                "unified_insights",
                query=query,
                result_count=len(results),
                source_distribution=source_types,
                avg_relevance=avg_relevance,
                context={
                    "instruction": "Analyze these unified search results and provide actionable insights"
                }
            )
            
            response = local_llm_service.generate_response(
                prompt,
                max_tokens=200,
                temperature=0.3
            )
            
            insights_text = response.get("response", "Analysis completed")
            
            return {
                "insights": insights_text,
                "patterns": {
                    "dominant_source": max(source_types, key=source_types.get) if source_types else None,
                    "source_diversity": len(source_types),
                    "average_relevance": avg_relevance
                },
                "recommendations": await self._generate_search_recommendations(query, results)
            }
            
        except Exception as e:
            logger.warning(f"Unified insights generation failed: {e}")
            return {
                "insights": "Insights generation unavailable",
                "error": str(e)
            }
    
    async def _generate_search_recommendations(self, query: str, results: List[Dict[str, Any]]) -> List[str]:
        """Generate search recommendations based on results"""
        
        recommendations = []
        
        # Analyze result quality
        high_relevance_count = len([r for r in results if r.get("relevance_score", 0) > 0.7])
        
        if high_relevance_count == 0:
            recommendations.append("Try refining your search query with more specific terms")
            recommendations.append("Consider using different keywords or synonyms")
        
        # Check source distribution
        source_types = set(r.get("source_type") for r in results)
        
        if "conversation" not in source_types:
            recommendations.append("Check conversation history for related discussions")
        
        if "ticket" not in source_types:
            recommendations.append("Review previous tickets for similar issues")
        
        if "policy" not in source_types:
            recommendations.append("Check policy documents for relevant procedures")
        
        # Query-specific recommendations
        intent = self._detect_search_intent(query)
        
        if intent == "troubleshooting" and len(results) < 3:
            recommendations.append("Consider searching for error codes or symptoms")
            recommendations.append("Look for similar issues in knowledge base")
        
        return recommendations[:5]
    
    async def get_search_history(
        self, 
        user_id: int, 
        limit: int = 20
    ) -> Dict[str, Any]:
        """Get user's search history with usage analytics"""
        
        try:
            # Get search history from conversation memory
            search_history = await conversation_memory_service.get_user_interactions(
                user_id=user_id,
                interaction_type="search",
                limit=limit
            )
            
            # Analyze search patterns
            patterns = await self._analyze_search_patterns(search_history)
            
            return {
                "search_history": search_history,
                "patterns": patterns,
                "total_searches": len(search_history),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Search history retrieval failed: {e}")
            return {"error": str(e), "search_history": []}
    
    async def _analyze_search_patterns(self, search_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user search patterns for insights"""
        
        if not search_history:
            return {"message": "No search history available"}
        
        # Simple pattern analysis
        query_lengths = [len(item.get("query", "").split()) for item in search_history]
        avg_query_length = sum(query_lengths) / len(query_lengths)
        
        # Common search topics
        all_queries = " ".join([item.get("query", "") for item in search_history])
        common_keywords = self._extract_keywords(all_queries)
        
        # Search frequency
        recent_searches = [
            item for item in search_history
            if (datetime.now() - datetime.fromisoformat(item.get("timestamp", "2020-01-01"))).days <= 7
        ]
        
        return {
            "average_query_length": avg_query_length,
            "common_topics": common_keywords[:5],
            "recent_search_count": len(recent_searches),
            "search_frequency": "high" if len(recent_searches) > 10 else "medium" if len(recent_searches) > 3 else "low",
            "patterns": {
                "prefers_short_queries": avg_query_length < 3,
                "diverse_topics": len(set(common_keywords)) > 10,
                "active_searcher": len(recent_searches) > 5
            }
        }
    
    async def suggest_related_searches(self, query: str, limit: int = 5) -> List[str]:
        """Suggest related searches based on query and historical data"""
        
        try:
            # Get suggestions from LLM
            prompt = prompt_service.build_prompt(
                "search_suggestions",
                query=query,
                context={
                    "instruction": f"Suggest {limit} related search queries for IT support"
                }
            )
            
            response = local_llm_service.generate_response(
                prompt,
                max_tokens=150,
                temperature=0.5
            )
            
            if response.get("error"):
                return self._fallback_suggestions(query)
            
            # Parse suggestions from response
            suggestions_text = response.get("response", "")
            suggestions = [
                suggestion.strip().strip("\"'")
                for suggestion in suggestions_text.split("\n")
                if suggestion.strip() and len(suggestion.strip()) > 5
            ]
            
            return suggestions[:limit]
            
        except Exception as e:
            logger.warning(f"Search suggestions failed: {e}")
            return self._fallback_suggestions(query)
    
    def _fallback_suggestions(self, query: str) -> List[str]:
        """Generate fallback search suggestions"""
        
        intent = self._detect_search_intent(query)
        
        if intent == "troubleshooting":
            return [
                f"{query} solution",
                f"how to fix {query}",
                f"{query} troubleshooting steps",
                f"{query} common causes",
                f"{query} prevention"
            ]
        elif intent == "policy_lookup":
            return [
                f"{query} policy",
                f"{query} procedures",
                f"{query} compliance",
                f"{query} requirements",
                f"{query} guidelines"
            ]
        else:
            return [
                f"{query} guide",
                f"{query} documentation",
                f"{query} best practices",
                f"{query} examples",
                f"{query} tutorial"
            ]


# Global instance
interactive_search_service = InteractiveSearchService()
