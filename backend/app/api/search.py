"""
Interactive Search API Router

Provides REST endpoints for LLM-enhanced search across multiple data sources.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

from app.services.interactive_search_service import interactive_search_service, SearchType
from app.core.auth import get_current_user
from app.models.models import User


router = APIRouter()


class SearchRequest(BaseModel):
    """Search request model"""
    query: str
    search_types: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    limit: int = 10


@router.post("/search", summary="Perform intelligent search")
async def intelligent_search(
    request: SearchRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Perform intelligent search across multiple data sources including:
    - Conversation history
    - Ticket history  
    - Policy documents
    - Knowledge graph
    - Unified search across all sources
    
    The search uses LLM enhancement for better query understanding and relevance scoring.
    """
    try:
        # Validate search types
        valid_types = [SearchType.CONVERSATION, SearchType.TICKETS, SearchType.POLICIES, 
                      SearchType.KNOWLEDGE_GRAPH, SearchType.UNIFIED]
        
        if request.search_types:
            invalid_types = [t for t in request.search_types if t not in valid_types]
            if invalid_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid search types: {invalid_types}. Valid types: {valid_types}"
                )
        
        # Perform search
        results = await interactive_search_service.intelligent_search(
            query=request.query,
            search_types=request.search_types,
            user_id=current_user.id,
            filters=request.filters,
            limit=request.limit
        )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/conversations", summary="Search conversation history")
async def search_conversations(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results to return"),
    start_date: Optional[str] = Query(None, description="Filter from date (ISO format)"),
    end_date: Optional[str] = Query(None, description="Filter to date (ISO format)"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Search conversation history using semantic similarity.
    Returns conversations ranked by relevance with highlights.
    """
    try:
        filters = {}
        if start_date:
            filters["start_date"] = start_date
        if end_date:
            filters["end_date"] = end_date
        
        # Enhance query
        enhanced_query = await interactive_search_service._enhance_search_query(query)
        
        # Search conversations
        results = await interactive_search_service._search_conversations(
            enhanced_query, current_user.id, filters, limit
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversation search failed: {str(e)}")


@router.get("/tickets", summary="Search ticket history")
async def search_tickets(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results to return"),
    status: Optional[str] = Query(None, description="Filter by ticket status"),
    priority: Optional[str] = Query(None, description="Filter by ticket priority"),
    start_date: Optional[str] = Query(None, description="Filter from date (ISO format)"),
    end_date: Optional[str] = Query(None, description="Filter to date (ISO format)"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Search ticket history with LLM-enhanced relevance scoring.
    Returns tickets ranked by relevance with highlighted matches.
    """
    try:
        filters = {}
        if status:
            filters["status"] = status
        if priority:
            filters["priority"] = priority
        if start_date:
            filters["start_date"] = start_date
        if end_date:
            filters["end_date"] = end_date
        
        # Enhance query
        enhanced_query = await interactive_search_service._enhance_search_query(query)
        
        # Search tickets
        results = await interactive_search_service._search_tickets(
            enhanced_query, current_user.id, filters, limit
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ticket search failed: {str(e)}")


@router.get("/policies", summary="Search policy documents")
async def search_policies(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results to return"),
    category: Optional[str] = Query(None, description="Filter by policy category"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Search policy documents with compliance analysis.
    Returns policies ranked by relevance with compliance insights.
    """
    try:
        filters = {}
        if category:
            filters["category"] = category
        if is_active is not None:
            filters["is_active"] = is_active
        
        # Enhance query
        enhanced_query = await interactive_search_service._enhance_search_query(query)
        
        # Search policies
        results = await interactive_search_service._search_policies(
            enhanced_query, filters, limit
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Policy search failed: {str(e)}")


@router.get("/knowledge-graph", summary="Search knowledge graph")
async def search_knowledge_graph(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=30, description="Maximum results to return"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Search knowledge graph using concept extraction and graph traversal.
    Returns related concepts with explanations of relevance.
    """
    try:
        # Enhance query
        enhanced_query = await interactive_search_service._enhance_search_query(query)
        
        # Search knowledge graph
        results = await interactive_search_service._search_knowledge_graph(
            enhanced_query, None, limit
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Knowledge graph search failed: {str(e)}")


@router.get("/suggestions", summary="Get search suggestions")
async def get_search_suggestions(
    query: str = Query(..., description="Partial or complete search query"),
    limit: int = Query(5, ge=1, le=10, description="Maximum suggestions to return")
) -> List[str]:
    """
    Get intelligent search suggestions based on query.
    Uses LLM to generate contextually relevant suggestions.
    """
    try:
        suggestions = await interactive_search_service.suggest_related_searches(query, limit)
        return suggestions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")


@router.get("/history", summary="Get user search history")
async def get_search_history(
    limit: int = Query(20, ge=1, le=100, description="Maximum history items to return"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get user's search history with usage analytics and patterns.
    Includes insights about search behavior and common topics.
    """
    try:
        history = await interactive_search_service.get_search_history(
            user_id=current_user.id,
            limit=limit
        )
        
        return history
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get search history: {str(e)}")


@router.get("/analytics", summary="Get search analytics")
async def get_search_analytics(
    days_back: int = Query(30, ge=1, le=365, description="Days to analyze"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get search analytics and insights for the specified time period.
    """
    try:
        # Get extended search history
        search_history = await interactive_search_service.get_search_history(
            user_id=current_user.id,
            limit=1000  # Get more data for analytics
        )
        
        # Filter by date range
        cutoff_date = datetime.now() - timedelta(days=days_back)
        filtered_history = [
            item for item in search_history.get("search_history", [])
            if datetime.fromisoformat(item.get("timestamp", "2020-01-01")) >= cutoff_date
        ]
        
        # Generate analytics
        analytics = {
            "period": {
                "days_back": days_back,
                "start_date": cutoff_date.isoformat(),
                "end_date": datetime.now().isoformat()
            },
            "search_volume": {
                "total_searches": len(filtered_history),
                "daily_average": len(filtered_history) / max(1, days_back),
                "peak_day": None  # Could be calculated with more detailed tracking
            },
            "query_analysis": await interactive_search_service._analyze_search_patterns(filtered_history),
            "success_metrics": {
                "searches_with_results": len([s for s in filtered_history if s.get("result_count", 0) > 0]),
                "average_results_per_search": sum(s.get("result_count", 0) for s in filtered_history) / max(1, len(filtered_history))
            }
        }
        
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


@router.get("/trending", summary="Get trending search topics")
async def get_trending_topics(
    days_back: int = Query(7, ge=1, le=30, description="Days to analyze for trends"),
    limit: int = Query(10, ge=1, le=20, description="Maximum trending topics to return")
) -> Dict[str, Any]:
    """
    Get trending search topics across all users for the specified period.
    """
    try:
        # This would require tracking search queries across all users
        # For now, return a placeholder response
        trending_topics = {
            "period": {
                "days_back": days_back,
                "end_date": datetime.now().isoformat()
            },
            "trending_topics": [
                {"topic": "VPN issues", "search_count": 45, "trend": "up"},
                {"topic": "Password reset", "search_count": 38, "trend": "stable"},
                {"topic": "Email configuration", "search_count": 32, "trend": "down"},
                {"topic": "Network connectivity", "search_count": 28, "trend": "up"},
                {"topic": "Software installation", "search_count": 25, "trend": "stable"}
            ][:limit],
            "total_searches": 168,
            "unique_queries": 89,
            "note": "Trending topics feature requires implementation of cross-user analytics"
        }
        
        return trending_topics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trending topics: {str(e)}")


@router.get("/quick-access", summary="Get quick access searches")
async def get_quick_access_searches(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get personalized quick access searches based on user patterns.
    """
    try:
        # Get user's search patterns
        search_history = await interactive_search_service.get_search_history(
            user_id=current_user.id,
            limit=50
        )
        
        patterns = search_history.get("patterns", {})
        common_topics = patterns.get("common_topics", [])
        
        # Generate quick access suggestions
        quick_access = {
            "personalized_searches": [],
            "common_it_searches": [
                "Password reset procedure",
                "VPN connection issues",
                "Email setup guide",
                "Network troubleshooting",
                "Software installation policy"
            ],
            "recent_topics": common_topics[:5],
            "suggested_searches": []
        }
        
        # Add personalized searches based on common topics
        for topic in common_topics[:3]:
            suggestions = await interactive_search_service.suggest_related_searches(topic, 2)
            quick_access["personalized_searches"].extend(suggestions)
        
        # Add general IT support suggestions
        general_suggestions = await interactive_search_service.suggest_related_searches(
            "IT support help", 3
        )
        quick_access["suggested_searches"] = general_suggestions
        
        return quick_access
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quick access: {str(e)}")


@router.get("/search-types", summary="Get available search types")
async def get_search_types() -> Dict[str, Any]:
    """
    Get information about available search types and their capabilities.
    """
    return {
        "search_types": {
            SearchType.CONVERSATION: {
                "description": "Search conversation history using semantic similarity",
                "features": ["semantic_search", "relevance_scoring", "highlighting"]
            },
            SearchType.TICKETS: {
                "description": "Search ticket history with LLM-enhanced filtering",
                "features": ["semantic_search", "status_filtering", "priority_filtering", "date_filtering"]
            },
            SearchType.POLICIES: {
                "description": "Search policy documents with compliance analysis",
                "features": ["semantic_search", "compliance_analysis", "category_filtering"]
            },
            SearchType.KNOWLEDGE_GRAPH: {
                "description": "Search knowledge graph using concept extraction",
                "features": ["concept_extraction", "graph_traversal", "relationship_analysis"]
            },
            SearchType.UNIFIED: {
                "description": "Search across all data sources with unified ranking",
                "features": ["cross_source_search", "unified_ranking", "insights_generation"]
            }
        },
        "default_type": SearchType.UNIFIED,
        "recommended_limits": {
            "conversation": 20,
            "tickets": 15,
            "policies": 10,
            "knowledge_graph": 10,
            "unified": 10
        }
    }


@router.post("/save-search", summary="Save search query")
async def save_search(
    query: str,
    result_count: int = 0,
    search_type: str = SearchType.UNIFIED,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Save a search query to user's history for analytics and personalization.
    """
    try:
        # Save search to conversation memory
        search_data = {
            "query": query,
            "search_type": search_type,
            "result_count": result_count,
            "timestamp": datetime.now().isoformat()
        }
        
        # Note: This requires implementing store_interaction in conversation_memory_service
        # For now, return a success response
        
        return {
            "saved": True,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "note": "Search saving requires implementation in conversation memory service"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save search: {str(e)}")


@router.get("/suggestions", summary="Get search suggestions")
async def get_search_suggestions(
    query: str = Query(..., description="Partial or complete search query"),
    limit: int = Query(5, ge=1, le=10, description="Maximum suggestions to return")
) -> List[str]:
    """
    Get intelligent search suggestions based on query.
    Uses LLM to generate contextually relevant suggestions.
    """
    try:
        suggestions = await interactive_search_service.suggest_related_searches(query, limit)
        return suggestions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")


@router.get("/history", summary="Get user search history")
async def get_search_history(
    limit: int = Query(20, ge=1, le=100, description="Maximum history items to return"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get user's search history with usage analytics and patterns.
    Includes insights about search behavior and common topics.
    """
    try:
        history = await interactive_search_service.get_search_history(
            user_id=current_user.id,
            limit=limit
        )
        
        return history
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get search history: {str(e)}")


@router.get("/analytics", summary="Get search analytics")
async def get_search_analytics(
    days_back: int = Query(30, ge=1, le=365, description="Days to analyze"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get search analytics and insights for the specified time period.
    """
    try:
        # Get extended search history
        search_history = await interactive_search_service.get_search_history(
            user_id=current_user.id,
            limit=1000  # Get more data for analytics
        )
        
        # Filter by date range
        cutoff_date = datetime.now() - timedelta(days=days_back)
        filtered_history = [
            item for item in search_history.get("search_history", [])
            if datetime.fromisoformat(item.get("timestamp", "2020-01-01")) >= cutoff_date
        ]
        
        # Generate analytics
        analytics = {
            "period": {
                "days_back": days_back,
                "start_date": cutoff_date.isoformat(),
                "end_date": datetime.now().isoformat()
            },
            "search_volume": {
                "total_searches": len(filtered_history),
                "daily_average": len(filtered_history) / max(1, days_back),
                "peak_day": None  # Could be calculated with more detailed tracking
            },
            "query_analysis": await interactive_search_service._analyze_search_patterns(filtered_history),
            "success_metrics": {
                "searches_with_results": len([s for s in filtered_history if s.get("result_count", 0) > 0]),
                "average_results_per_search": sum(s.get("result_count", 0) for s in filtered_history) / max(1, len(filtered_history))
            }
        }
        
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


@router.get("/stats", summary="Get search statistics")
async def get_search_stats(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get comprehensive search statistics for the current user.
    """
    try:
        # Get search history
        search_history = await interactive_search_service.get_search_history(
            user_id=current_user.id,
            limit=1000
        )
        
        history_items = search_history.get("search_history", [])
        patterns = search_history.get("patterns", {})
        
        # Calculate additional stats
        total_searches = len(history_items)
        
        # Recent activity (last 7 days)
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_searches = [
            item for item in history_items
            if datetime.fromisoformat(item.get("timestamp", "2020-01-01")) >= recent_cutoff
        ]
        
        # Search intent distribution
        intent_counts = {}
        for item in history_items:
            query = item.get("query", "")
            intent = interactive_search_service._detect_search_intent(query)
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        return {
            "user_id": current_user.id,
            "total_searches": total_searches,
            "recent_activity": {
                "searches_last_7_days": len(recent_searches),
                "average_per_day": len(recent_searches) / 7
            },
            "search_patterns": patterns,
            "intent_distribution": intent_counts,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get search stats: {str(e)}")
