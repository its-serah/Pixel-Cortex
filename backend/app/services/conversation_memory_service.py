"""
Conversation Memory Service

Manages persistent conversation storage, context retrieval, and intelligent
memory management for the IT Support Agent.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_

from app.models.models import ConversationLog, User, SupportTicket
from app.models.schemas import ConversationLogCreate
from app.services.local_llm_service import local_llm_service


logger = logging.getLogger(__name__)


class ConversationMemoryService:
    """Service for managing conversation memory and context"""
    
    def __init__(self):
        self.max_context_age_days = 30
        self.max_context_entries = 50
        self.similarity_threshold = 0.3
        
    def log_conversation(
        self,
        db: Session,
        user_id: int,
        user_message: str,
        assistant_response: str,
        ticket_id: Optional[int] = None,
        audio_metadata: Optional[Dict[str, Any]] = None,
        llm_metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> ConversationLog:
        """
        Log a conversation exchange with rich metadata
        
        Args:
            db: Database session
            user_id: User ID
            user_message: User's input message
            assistant_response: Assistant's response
            ticket_id: Associated ticket ID if any
            audio_metadata: Audio processing metadata if from voice input
            llm_metadata: LLM processing metadata
            session_id: Conversation session identifier
            
        Returns:
            Created conversation log entry
        """
        
        # Prepare conversation metadata
        conversation_metadata = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "input_method": "audio" if audio_metadata else "text",
            "response_time_ms": llm_metadata.get("inference_time_ms", 0) if llm_metadata else 0,
        }
        
        # Add audio metadata if available
        if audio_metadata:
            conversation_metadata["audio"] = {
                "duration": audio_metadata.get("duration", 0),
                "confidence": audio_metadata.get("confidence", 0),
                "language": audio_metadata.get("language", "en"),
                "processing_time_ms": audio_metadata.get("processing_time_ms", 0)
            }
        
        # Add LLM metadata if available
        if llm_metadata:
            conversation_metadata["llm"] = {
                "model_name": llm_metadata.get("model_name", "unknown"),
                "tokens_generated": llm_metadata.get("tokens_generated", 0),
                "from_cache": llm_metadata.get("from_cache", False),
                "safety_flags": llm_metadata.get("safety_flags", {}),
                "memory_usage": llm_metadata.get("memory_usage", {})
            }
        
        # Create conversation log entry
        log_entry = ConversationLog(
            user_id=user_id,
            user_message=user_message,
            assistant_response=assistant_response,
            ticket_id=ticket_id,
            metadata=conversation_metadata,
            timestamp=datetime.now()
        )
        
        db.add(log_entry)
        db.commit()
        
        logger.info(f"Conversation logged for user {user_id}, session {session_id}")
        return log_entry
    
    def get_conversation_context(
        self,
        db: Session,
        user_id: int,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversation context for a user
        
        Args:
            db: Database session
            user_id: User ID
            session_id: Optional session ID to filter by
            limit: Maximum number of entries to return
            
        Returns:
            List of conversation context entries
        """
        
        query = db.query(ConversationLog).filter(ConversationLog.user_id == user_id)
        
        # Filter by session if provided
        if session_id:
            query = query.filter(ConversationLog.metadata.contains({"session_id": session_id}))
        
        # Get recent conversations
        recent_logs = query.order_by(desc(ConversationLog.timestamp)).limit(limit).all()
        
        context_entries = []
        for log in recent_logs:
            context_entries.append({
                "user_message": log.user_message,
                "assistant_response": log.assistant_response,
                "timestamp": log.timestamp.isoformat(),
                "ticket_id": log.ticket_id,
                "input_method": log.metadata.get("input_method", "text"),
                "session_id": log.metadata.get("session_id")
            })
        
        return list(reversed(context_entries))  # Return in chronological order
    
    def search_conversation_history(
        self,
        db: Session,
        query: str,
        user_id: Optional[int] = None,
        days_back: int = 30,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search conversation history using LLM-powered semantic search
        
        Args:
            db: Database session
            query: Search query
            user_id: Optional user ID to filter by
            days_back: How many days back to search
            limit: Maximum results to return
            
        Returns:
            List of relevant conversation entries with similarity scores
        """
        
        # Build base query
        cutoff_date = datetime.now() - timedelta(days=days_back)
        db_query = db.query(ConversationLog).filter(ConversationLog.timestamp >= cutoff_date)
        
        if user_id:
            db_query = db_query.filter(ConversationLog.user_id == user_id)
        
        # Get conversations to search
        conversations = db_query.order_by(desc(ConversationLog.timestamp)).limit(200).all()
        
        if not conversations:
            return []
        
        # Use LLM service for semantic search
        search_results = local_llm_service.search_conversation_history(query, db, limit)
        
        # Enhance with conversation metadata
        enhanced_results = []
        for result in search_results:
            conversation = next((c for c in conversations if c.id == result["conversation_id"]), None)
            if conversation:
                enhanced_result = {
                    **result,
                    "user_id": conversation.user_id,
                    "ticket_id": conversation.ticket_id,
                    "input_method": conversation.metadata.get("input_method", "text"),
                    "session_id": conversation.metadata.get("session_id"),
                    "audio_duration": conversation.metadata.get("audio", {}).get("duration", 0)
                }
                enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def get_related_ticket_context(
        self,
        db: Session,
        current_query: str,
        user_id: Optional[int] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find related support tickets using LLM-powered similarity search
        
        Args:
            db: Database session
            current_query: Current user query
            user_id: Optional user ID to filter by
            limit: Maximum results to return
            
        Returns:
            List of related tickets with context
        """
        
        # Get recent tickets
        query = db.query(SupportTicket)
        if user_id:
            query = query.filter(SupportTicket.requester_id == user_id)
        
        recent_tickets = query.order_by(desc(SupportTicket.created_at)).limit(100).all()
        
        if not recent_tickets:
            return []
        
        # Use LLM for semantic similarity
        ticket_texts = []
        for ticket in recent_tickets:
            ticket_text = f"{ticket.title} {ticket.description}"
            ticket_texts.append(ticket_text)
        
        # Get embeddings and find similar tickets
        try:
            local_llm_service.load_model()
            query_embedding = local_llm_service.embedding_model.encode([current_query])
            ticket_embeddings = local_llm_service.embedding_model.encode(ticket_texts)
            
            # Calculate similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, ticket_embeddings)[0]
            
            # Get top matches
            top_indices = similarities.argsort()[-limit:][::-1]
            
            related_tickets = []
            for idx in top_indices:
                if similarities[idx] > self.similarity_threshold:
                    ticket = recent_tickets[idx]
                    related_tickets.append({
                        "ticket_id": ticket.id,
                        "title": ticket.title,
                        "description": ticket.description[:200] + "..." if len(ticket.description) > 200 else ticket.description,
                        "status": ticket.status.value,
                        "category": ticket.category.value,
                        "priority": ticket.priority.value,
                        "created_at": ticket.created_at.isoformat(),
                        "similarity_score": float(similarities[idx]),
                        "context_type": "related_ticket"
                    })
            
            return related_tickets
            
        except Exception as e:
            logger.error(f"Related ticket search error: {e}")
            return []
    
    def summarize_conversation_session(
        self,
        db: Session,
        session_id: str,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a summary of a conversation session using LLM
        
        Args:
            db: Database session
            session_id: Session to summarize
            user_id: Optional user ID filter
            
        Returns:
            Conversation session summary
        """
        
        # Get all conversations in session
        query = db.query(ConversationLog).filter(
            ConversationLog.metadata.contains({"session_id": session_id})
        )
        
        if user_id:
            query = query.filter(ConversationLog.user_id == user_id)
        
        session_logs = query.order_by(ConversationLog.timestamp).all()
        
        if not session_logs:
            return {"error": "Session not found", "summary": ""}
        
        # Build conversation text
        conversation_text = []
        for log in session_logs:
            conversation_text.append(f"User: {log.user_message}")
            if log.assistant_response:
                conversation_text.append(f"Assistant: {log.assistant_response}")
        
        full_conversation = "\\n".join(conversation_text)
        
        # Generate summary using LLM
        summary_prompt = f"""Summarize this IT support conversation session. Focus on:
- Main issues discussed
- Solutions provided
- Action items or follow-ups needed
- Key technical concepts involved

Conversation:
{full_conversation}

Provide a concise summary (max 200 words):"""
        
        try:
            response = local_llm_service.generate_response(
                summary_prompt,
                max_tokens=256,
                temperature=0.3,
                context={"task": "conversation_summary"}
            )
            
            summary = response["response"]
            
            # Extract session statistics
            session_stats = {
                "total_exchanges": len(session_logs),
                "duration_minutes": (session_logs[-1].timestamp - session_logs[0].timestamp).total_seconds() / 60,
                "audio_interactions": sum(1 for log in session_logs if log.metadata.get("input_method") == "audio"),
                "tickets_created": len(set(log.ticket_id for log in session_logs if log.ticket_id)),
                "session_start": session_logs[0].timestamp.isoformat(),
                "session_end": session_logs[-1].timestamp.isoformat()
            }
            
            return {
                "summary": summary,
                "session_stats": session_stats,
                "total_exchanges": len(session_logs),
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Session summary error: {e}")
            return {
                "error": str(e),
                "summary": "Failed to generate summary",
                "total_exchanges": len(session_logs)
            }
    
    def get_user_interaction_patterns(
        self,
        db: Session,
        user_id: int,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze user interaction patterns for personalized support
        
        Args:
            db: Database session
            user_id: User ID to analyze
            days_back: Analysis time window
            
        Returns:
            User interaction pattern analysis
        """
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        user_logs = db.query(ConversationLog).filter(
            and_(
                ConversationLog.user_id == user_id,
                ConversationLog.timestamp >= cutoff_date
            )
        ).all()
        
        if not user_logs:
            return {"error": "No conversation history found"}
        
        # Analyze patterns
        patterns = {
            "total_conversations": len(user_logs),
            "audio_usage_percent": (
                sum(1 for log in user_logs if log.metadata.get("input_method") == "audio") 
                / len(user_logs) * 100
            ),
            "avg_session_length": 0,
            "common_topics": [],
            "peak_usage_hours": [],
            "preferred_response_style": "detailed"  # Default
        }
        
        # Analyze usage times
        usage_hours = [log.timestamp.hour for log in user_logs]
        from collections import Counter
        hour_counts = Counter(usage_hours)
        patterns["peak_usage_hours"] = [hour for hour, count in hour_counts.most_common(3)]
        
        # Extract common topics using LLM
        recent_messages = [log.user_message for log in user_logs[-20:]]  # Last 20 messages
        if recent_messages:
            topic_prompt = f"""Analyze these IT support messages and identify the 5 most common topics or issues:

Messages:
{chr(10).join(recent_messages)}

Return ONLY a JSON array of topics:
["VPN Issues", "Password Reset", "Software Installation", ...]

Response:"""
            
            try:
                response = local_llm_service.generate_response(
                    topic_prompt,
                    max_tokens=128,
                    temperature=0.3,
                    context={"task": "topic_analysis"}
                )
                
                topics = json.loads(response["response"])
                patterns["common_topics"] = topics[:5]  # Top 5 topics
                
            except Exception as e:
                logger.warning(f"Topic analysis failed: {e}")
                patterns["common_topics"] = ["General IT Support"]
        
        return patterns
    
    def get_contextual_memory(
        self,
        db: Session,
        current_query: str,
        user_id: int,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get relevant contextual memory for current query
        
        Args:
            db: Database session
            current_query: Current user query
            user_id: User ID
            session_id: Current session ID
            
        Returns:
            Contextual memory including conversation history, related tickets, and patterns
        """
        
        context = {
            "conversation_history": [],
            "related_conversations": [],
            "related_tickets": [],
            "user_patterns": {},
            "context_summary": ""
        }
        
        # Get current session context
        if session_id:
            context["conversation_history"] = self.get_conversation_context(
                db, user_id, session_id, limit=5
            )
        
        # Search for related conversations
        context["related_conversations"] = self.search_conversation_history(
            db, current_query, user_id, days_back=7, limit=3
        )
        
        # Get related tickets
        context["related_tickets"] = self.get_related_ticket_context(
            db, current_query, user_id, limit=3
        )
        
        # Get user patterns
        context["user_patterns"] = self.get_user_interaction_patterns(
            db, user_id, days_back=14
        )
        
        # Generate context summary using LLM
        context["context_summary"] = self._generate_context_summary(context, current_query)
        
        return context
    
    def _generate_context_summary(self, context: Dict[str, Any], current_query: str) -> str:
        """Generate a concise context summary using LLM"""
        
        # Build context text
        context_parts = []
        
        if context["conversation_history"]:
            recent_topics = [entry["user_message"][:50] + "..." for entry in context["conversation_history"][-3:]]
            context_parts.append(f"Recent topics: {', '.join(recent_topics)}")
        
        if context["related_conversations"]:
            similar_issues = [conv["user_message"][:50] + "..." for conv in context["related_conversations"][:2]]
            context_parts.append(f"Similar past issues: {', '.join(similar_issues)}")
        
        if context["related_tickets"]:
            related_tickets = [ticket["title"] for ticket in context["related_tickets"][:2]]
            context_parts.append(f"Related tickets: {', '.join(related_tickets)}")
        
        if context["user_patterns"].get("common_topics"):
            topics = context["user_patterns"]["common_topics"][:3]
            context_parts.append(f"User's common topics: {', '.join(topics)}")
        
        if not context_parts:
            return "No relevant context available."
        
        context_text = ". ".join(context_parts)
        
        summary_prompt = f"""Based on this user's context and current query, provide a brief summary of relevant background:

Context: {context_text}
Current Query: {current_query}

Provide a 1-2 sentence summary of the most relevant context:"""
        
        try:
            response = local_llm_service.generate_response(
                summary_prompt,
                max_tokens=64,
                temperature=0.3,
                context={"task": "context_summary"}
            )
            
            return response["response"]
            
        except Exception as e:
            logger.warning(f"Context summary generation failed: {e}")
            return "Previous interactions available for reference."
    
    def cleanup_old_conversations(self, db: Session, days_to_keep: int = 90) -> Dict[str, int]:
        """
        Clean up old conversation logs to manage database size
        
        Args:
            db: Database session
            days_to_keep: Number of days of history to keep
            
        Returns:
            Cleanup statistics
        """
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Count conversations to delete
        old_conversations = db.query(ConversationLog).filter(
            ConversationLog.timestamp < cutoff_date
        )
        
        count_to_delete = old_conversations.count()
        
        # Perform deletion
        deleted_count = old_conversations.delete()
        db.commit()
        
        logger.info(f"Cleaned up {deleted_count} old conversation logs")
        
        return {
            "conversations_deleted": deleted_count,
            "cutoff_date": cutoff_date.isoformat(),
            "days_kept": days_to_keep
        }
    
    def export_conversation_data(
        self,
        db: Session,
        user_id: Optional[int] = None,
        session_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Export conversation data for analysis or backup
        
        Args:
            db: Database session
            user_id: Optional user ID filter
            session_id: Optional session ID filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Exported conversation data
        """
        
        query = db.query(ConversationLog)
        
        # Apply filters
        if user_id:
            query = query.filter(ConversationLog.user_id == user_id)
        
        if session_id:
            query = query.filter(ConversationLog.metadata.contains({"session_id": session_id}))
        
        if start_date:
            query = query.filter(ConversationLog.timestamp >= start_date)
        
        if end_date:
            query = query.filter(ConversationLog.timestamp <= end_date)
        
        conversations = query.order_by(ConversationLog.timestamp).all()
        
        # Export data
        exported_data = {
            "export_metadata": {
                "total_conversations": len(conversations),
                "export_date": datetime.now().isoformat(),
                "filters": {
                    "user_id": user_id,
                    "session_id": session_id,
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None
                }
            },
            "conversations": []
        }
        
        for log in conversations:
            conversation_data = {
                "id": log.id,
                "user_id": log.user_id,
                "user_message": log.user_message,
                "assistant_response": log.assistant_response,
                "ticket_id": log.ticket_id,
                "timestamp": log.timestamp.isoformat(),
                "metadata": log.metadata
            }
            exported_data["conversations"].append(conversation_data)
        
        return exported_data
    
    def get_memory_statistics(self, db: Session) -> Dict[str, Any]:
        """Get conversation memory statistics"""
        
        # Total conversations
        total_conversations = db.query(ConversationLog).count()
        
        # Recent activity (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        recent_conversations = db.query(ConversationLog).filter(
            ConversationLog.timestamp >= week_ago
        ).count()
        
        # Audio vs text usage
        audio_conversations = db.query(ConversationLog).filter(
            ConversationLog.metadata.contains({"input_method": "audio"})
        ).count()
        
        # Average session length
        session_lengths = []
        sessions = db.query(ConversationLog).all()
        session_groups = {}
        
        for log in sessions:
            session_id = log.metadata.get("session_id", "unknown")
            if session_id not in session_groups:
                session_groups[session_id] = []
            session_groups[session_id].append(log)
        
        for session_logs in session_groups.values():
            if len(session_logs) > 1:
                duration = (session_logs[-1].timestamp - session_logs[0].timestamp).total_seconds() / 60
                session_lengths.append(duration)
        
        avg_session_length = sum(session_lengths) / len(session_lengths) if session_lengths else 0
        
        return {
            "total_conversations": total_conversations,
            "recent_conversations_7d": recent_conversations,
            "audio_usage_percent": (audio_conversations / total_conversations * 100) if total_conversations > 0 else 0,
            "avg_session_length_minutes": avg_session_length,
            "total_sessions": len(session_groups),
            "memory_age_days": (datetime.now() - sessions[0].timestamp).days if sessions else 0
        }


# Global instance
conversation_memory_service = ConversationMemoryService()
