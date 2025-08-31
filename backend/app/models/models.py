from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Float, Table, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    AGENT = "agent"
    USER = "user"

class TicketStatus(str, enum.Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    WAITING_FOR_USER = "waiting_for_user"
    RESOLVED = "resolved"
    CLOSED = "closed"

class TicketPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TicketCategory(str, enum.Enum):
    HARDWARE = "hardware"
    SOFTWARE = "software"
    NETWORK = "network"
    ACCESS = "access"
    SECURITY = "security"
    OTHER = "other"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    role = Column(SQLEnum(UserRole), default=UserRole.USER)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)  # Added for test compatibility
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tickets = relationship("Ticket", back_populates="requester", foreign_keys="Ticket.requester_id")
    assigned_tickets = relationship("Ticket", back_populates="assigned_agent", foreign_keys="Ticket.assigned_agent_id")

class Ticket(Base):
    __tablename__ = "tickets"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    category = Column(SQLEnum(TicketCategory), nullable=False)
    priority = Column(SQLEnum(TicketPriority), default=TicketPriority.MEDIUM)
    status = Column(SQLEnum(TicketStatus), default=TicketStatus.OPEN)
    
    # User relationships
    requester_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    assigned_agent_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    due_date = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    
    # Triage and XAI data
    triage_confidence = Column(Float, nullable=True)
    triage_reasoning = Column(JSON, nullable=True)
    
    # Relationships
    requester = relationship("User", back_populates="tickets", foreign_keys=[requester_id])
    assigned_agent = relationship("User", back_populates="assigned_tickets", foreign_keys=[assigned_agent_id])
    events = relationship("TicketEvent", back_populates="ticket")

class TicketEvent(Base):
    __tablename__ = "ticket_events"
    
    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    event_type = Column(String, nullable=False)  # created, updated, status_changed, assigned, commented
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    comment = Column(Text, nullable=True)
    explanation_data = Column(JSON, nullable=True)  # XAI explanation object
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    ticket = relationship("Ticket", back_populates="events")
    user = relationship("User")

class SLAConfig(Base):
    __tablename__ = "sla_config"
    
    id = Column(Integer, primary_key=True, index=True)
    category = Column(SQLEnum(TicketCategory), nullable=False)
    priority = Column(SQLEnum(TicketPriority), nullable=False)
    response_time_hours = Column(Integer, nullable=False)  # Hours to first response
    resolution_time_hours = Column(Integer, nullable=False)  # Hours to resolution
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_log"
    
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String, unique=True, nullable=False)  # UUID for this event
    previous_hash = Column(String, nullable=True)  # Hash of previous audit entry
    current_hash = Column(String, nullable=False)  # Hash of this entry
    
    # Event data
    action = Column(String, nullable=False)
    resource_type = Column(String, nullable=False)
    resource_id = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Detailed event data
    event_data = Column(JSON, nullable=False)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    
    # Relationships
    user = relationship("User")

class PolicyDocument(Base):
    __tablename__ = "policy_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False, unique=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String, nullable=False)  # Hash of content for change detection
    file_type = Column(String, nullable=False)  # md, pdf, txt
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    chunks = relationship("PolicyChunk", back_populates="document")

class PolicyChunk(Base):
    __tablename__ = "policy_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("policy_documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String, nullable=False)
    
    # BM25 and vector embeddings (stored as JSON for simplicity)
    bm25_features = Column(JSON, nullable=True)
    tfidf_vector = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("PolicyDocument", back_populates="chunks")

# Knowledge Graph Models

# Association table for many-to-many relationship between concepts
concept_relationships = Table(
    'concept_relationships',
    Base.metadata,
    Column('source_concept_id', Integer, ForeignKey('kg_concepts.id'), primary_key=True),
    Column('target_concept_id', Integer, ForeignKey('kg_concepts.id'), primary_key=True),
    Column('relationship_type', String, nullable=False),  # requires, depends_on, overrides, related_to
    Column('weight', Float, default=1.0),  # Relationship strength
    Column('created_at', DateTime, default=datetime.utcnow),
    Column('metadata', JSON, nullable=True)  # Additional relationship data
)

class KnowledgeGraphConcept(Base):
    __tablename__ = "kg_concepts"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)  # e.g., "VPN", "MFA", "Remote Access"
    concept_type = Column(String, nullable=False)  # technology, policy, procedure, requirement
    description = Column(Text, nullable=True)
    aliases = Column(JSON, nullable=True)  # Alternative names/terms
    
    # Policy grounding - which policy chunks mention this concept
    policy_chunks = Column(JSON, nullable=True)  # List of chunk IDs that mention this concept
    
    # Metadata
    importance_score = Column(Float, default=1.0)  # How critical this concept is
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Many-to-many relationships with other concepts
    related_concepts_out = relationship(
        "KnowledgeGraphConcept",
        secondary=concept_relationships,
        primaryjoin=id == concept_relationships.c.source_concept_id,
        secondaryjoin=id == concept_relationships.c.target_concept_id,
        back_populates="related_concepts_in"
    )
    related_concepts_in = relationship(
        "KnowledgeGraphConcept",
        secondary=concept_relationships,
        primaryjoin=id == concept_relationships.c.target_concept_id,
        secondaryjoin=id == concept_relationships.c.source_concept_id,
        back_populates="related_concepts_out"
    )

class KnowledgeGraphQuery(Base):
    __tablename__ = "kg_queries"
    
    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(String, nullable=False)
    initial_concepts = Column(JSON, nullable=False)  # List of concept IDs found in initial search
    graph_hops = Column(JSON, nullable=False)  # Path taken through the graph
    retrieved_chunks = Column(JSON, nullable=False)  # Final set of policy chunk IDs
    
    # Query metrics
    processing_time_ms = Column(Integer, nullable=False)
    graph_traversal_depth = Column(Integer, nullable=False)
    total_concepts_visited = Column(Integer, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
class PolicyConceptExtraction(Base):
    __tablename__ = "policy_concept_extractions"
    
    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(Integer, ForeignKey("policy_chunks.id"), nullable=False)
    concept_id = Column(Integer, ForeignKey("kg_concepts.id"), nullable=False)
    
    # Extraction confidence and context
    confidence_score = Column(Float, nullable=False)  # How confident we are this concept is in the chunk
    context_window = Column(Text, nullable=True)  # Text around where concept was found
    extraction_method = Column(String, nullable=False)  # spacy, regex, manual
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    chunk = relationship("PolicyChunk")
    concept = relationship("KnowledgeGraphConcept")

class ConversationLog(Base):
    __tablename__ = "conversation_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Message data
    user_message = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    
    # Metadata
    response_time_ms = Column(Integer, nullable=True)
    model_used = Column(String, nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
