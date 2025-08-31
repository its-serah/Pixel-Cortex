from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.models.models import UserRole, TicketStatus, TicketPriority, TicketCategory

# User schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    role: UserRole = UserRole.USER
    is_active: bool = True

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None

class UserResponse(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Authentication schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

# Ticket schemas
class TicketBase(BaseModel):
    title: str
    description: str
    category: Optional[TicketCategory] = None
    priority: TicketPriority = TicketPriority.MEDIUM

class TicketCreate(TicketBase):
    pass

class TicketUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[TicketCategory] = None
    priority: Optional[TicketPriority] = None
    status: Optional[TicketStatus] = None
    assigned_agent_id: Optional[int] = None

class TicketResponse(TicketBase):
    id: int
    status: TicketStatus
    category: TicketCategory
    requester_id: int
    assigned_agent_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    due_date: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    triage_confidence: Optional[float] = None
    triage_reasoning: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True

# XAI Explanation schemas
class PolicyCitation(BaseModel):
    document_id: int
    document_title: str
    chunk_id: int
    chunk_content: str
    relevance_score: float

class ReasoningStep(BaseModel):
    step: int
    action: str
    rationale: str
    confidence: float
    policy_refs: List[int] = Field(default_factory=list)  # Referenced chunk IDs

class AlternativeOption(BaseModel):
    option: str
    pros: List[str]
    cons: List[str]
    confidence: float

class Counterfactual(BaseModel):
    condition: str
    outcome: str
    likelihood: float

class TelemetryData(BaseModel):
    latency_ms: int
    retrieval_k: int
    triage_time_ms: int
    planning_time_ms: int
    total_chunks_considered: int
    
class ExplanationObject(BaseModel):
    # Allow field name model_version without warnings
    model_config = ConfigDict(protected_namespaces=())

    # Core decision
    answer: str
    decision: str
    confidence: float
    
    # Reasoning chain
    reasoning_trace: List[ReasoningStep]
    
    # Evidence and citations
    policy_citations: List[PolicyCitation]
    
    # Uncertainty and alternatives
    missing_info: List[str] = Field(default_factory=list)
    alternatives_considered: List[AlternativeOption] = Field(default_factory=list)
    counterfactuals: List[Counterfactual] = Field(default_factory=list)
    
    # Performance metrics
    telemetry: TelemetryData
    
    # Metadata
    timestamp: datetime
    model_version: str = "1.0.0"

# Ticket Event schemas
class TicketEventCreate(BaseModel):
    event_type: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    comment: Optional[str] = None

class TicketEventResponse(BaseModel):
    id: int
    ticket_id: int
    user_id: int
    event_type: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    comment: Optional[str] = None
    explanation_data: Optional[ExplanationObject] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

# Triage request/response
class TriageRequest(BaseModel):
    title: str
    description: str

class TriageResponse(BaseModel):
    category: TicketCategory
    priority: TicketPriority
    confidence: float
    explanation: ExplanationObject

# Knowledge Graph schemas
class ConceptRelationship(BaseModel):
    source_concept_id: int
    target_concept_id: int
    relationship_type: str  # requires, depends_on, overrides, related_to
    weight: float = 1.0
    metadata: Optional[Dict[str, Any]] = None

class KnowledgeGraphConceptBase(BaseModel):
    name: str
    concept_type: str  # technology, policy, procedure, requirement
    description: Optional[str] = None
    aliases: Optional[List[str]] = None
    importance_score: float = 1.0

class KnowledgeGraphConceptCreate(KnowledgeGraphConceptBase):
    policy_chunks: Optional[List[int]] = None

class KnowledgeGraphConceptResponse(KnowledgeGraphConceptBase):
    id: int
    policy_chunks: Optional[List[int]] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class GraphHop(BaseModel):
    from_concept: str
    to_concept: str
    relationship_type: str
    hop_number: int
    reasoning: str

class KGEnhancedPolicyCitation(PolicyCitation):
    """Extended policy citation that includes graph reasoning"""
    graph_path: List[GraphHop] = Field(default_factory=list)  # How we reached this policy through the graph
    semantic_score: float  # Original semantic search score
    graph_boost_score: float  # Additional relevance from graph connections
    combined_score: float  # Final relevance score

class KnowledgeGraphQuery(BaseModel):
    query_text: str
    initial_concepts: List[int]  # Concept IDs found in semantic search
    graph_hops: List[GraphHop]  # Path taken through the graph
    retrieved_chunks: List[int]  # Final set of policy chunk IDs
    processing_time_ms: int
    graph_traversal_depth: int
    total_concepts_visited: int

class KGEnhancedExplanationObject(ExplanationObject):
    """Enhanced explanation object that includes knowledge graph reasoning"""
    kg_policy_citations: List[KGEnhancedPolicyCitation] = Field(default_factory=list)  # Graph-enhanced citations
    graph_reasoning: List[GraphHop] = Field(default_factory=list)  # Graph traversal path
    concepts_discovered: List[str] = Field(default_factory=list)  # New concepts found via graph
    graph_coverage_score: float = 0.0  # How well the graph covered the query

class GraphVisualization(BaseModel):
    """Data structure for graph visualization"""
    nodes: List[Dict[str, Any]]  # Graph nodes with properties
    edges: List[Dict[str, Any]]  # Graph edges with properties
    highlighted_path: Optional[List[int]] = None  # Highlighted reasoning path
    query_concepts: List[int] = Field(default_factory=list)  # Concepts that matched the original query

class PolicyConceptExtractionResponse(BaseModel):
    chunk_id: int
    concept_id: int
    concept_name: str
    confidence_score: float
    context_window: Optional[str] = None
    extraction_method: str
    created_at: datetime
    
    class Config:
        from_attributes = True
