import pytest
from unittest.mock import Mock
from app.services.decision_service import DecisionService, DecisionType
from app.models.models import TicketCategory, TicketPriority
from app.models.schemas import PolicyCitation

def test_decision_service_security_violation():
    """Test automatic denial for security violations"""
    service = DecisionService()
    
    # Mock database session
    mock_db = Mock()
    
    # Test case: unauthorized software installation
    title = "Install gaming software"
    description = "Need to install personal gaming software on work computer"
    category = TicketCategory.SOFTWARE
    priority = TicketPriority.LOW
    
    decision, explanation = service.make_decision(title, description, category, priority, mock_db)
    
    assert decision == DecisionType.DENIED
    assert "security policy violations" in explanation.reasoning_trace[0].rationale
    assert explanation.confidence > 0.9


def test_decision_service_requires_approval():
    """Test approval requirement for high-cost items"""
    service = DecisionService()
    
    mock_db = Mock()
    
    # Test case: expensive hardware purchase
    title = "Purchase new server"
    description = "Need expensive hardware for production database server"
    category = TicketCategory.HARDWARE
    priority = TicketPriority.HIGH
    
    decision, explanation = service.make_decision(title, description, category, priority, mock_db)
    
    assert decision == DecisionType.NEEDS_APPROVAL
    assert "high cost" in explanation.reasoning_trace[0].rationale or "critical system" in explanation.reasoning_trace[0].rationale
    assert explanation.confidence >= 0.8


def test_decision_service_allowed():
    """Test automatic approval for standard requests"""
    service = DecisionService()
    
    mock_db = Mock()
    
    # Test case: routine password reset
    title = "Password reset request"
    description = "Standard password reset for user account"
    category = TicketCategory.ACCESS
    priority = TicketPriority.MEDIUM
    
    decision, explanation = service.make_decision(title, description, category, priority, mock_db)
    
    assert decision == DecisionType.ALLOWED
    assert explanation.confidence > 0.6
    assert len(explanation.reasoning_trace) > 0


def test_decision_service_policy_grounding():
    """Test that decisions are grounded in policy citations"""
    service = DecisionService()
    
    # Mock policy retriever to return citations
    mock_citations = [
        PolicyCitation(
            chunk_id="policy_123",
            document_name="Security Policy",
            chunk_content="Unauthorized software installation is prohibited and will be denied",
            page_number=1,
            relevance_score=0.95
        )
    ]
    
    service.policy_retriever.retrieve_relevant_chunks = Mock(return_value=mock_citations)
    
    mock_db = Mock()
    
    title = "Install unauthorized software"
    description = "Need to bypass security policies"
    category = TicketCategory.SOFTWARE
    priority = TicketPriority.LOW
    
    decision, explanation = service.make_decision(title, description, category, priority, mock_db)
    
    assert decision == DecisionType.DENIED
    assert len(explanation.policy_citations) > 0
    assert explanation.policy_citations[0].chunk_id == "policy_123"
    assert "policy" in explanation.reasoning_trace[-1].action


def test_decision_confidence_calculation():
    """Test that confidence is calculated based on policy support"""
    service = DecisionService()
    
    # High relevance policy citations should increase confidence
    high_relevance_citations = [
        PolicyCitation(
            chunk_id="policy_456",
            document_name="IT Policy",
            chunk_content="Password resets are automatically approved for standard users",
            page_number=1,
            relevance_score=0.95
        )
    ]
    
    service.policy_retriever.retrieve_relevant_chunks = Mock(return_value=high_relevance_citations)
    
    mock_db = Mock()
    
    decision, explanation = service.make_decision(
        "Password reset", "Standard password reset", TicketCategory.ACCESS, TicketPriority.LOW, mock_db
    )
    
    # Should have high confidence with strong policy backing
    assert explanation.confidence > 0.8


def test_decision_missing_information_identification():
    """Test identification of missing information that could affect decisions"""
    service = DecisionService()
    mock_db = Mock()
    
    # Software request without cost information
    title = "Install new software"
    description = "Need new software for project work"
    category = TicketCategory.SOFTWARE
    priority = TicketPriority.MEDIUM
    
    decision, explanation = service.make_decision(title, description, category, priority, mock_db)
    
    # Should identify missing cost information
    assert any("cost" in info.lower() for info in explanation.missing_info)


def test_decision_alternatives_and_counterfactuals():
    """Test generation of alternatives and counterfactuals"""
    service = DecisionService()
    mock_db = Mock()
    
    title = "Install unauthorized software"
    description = "Need gaming software on work computer"
    category = TicketCategory.SOFTWARE
    priority = TicketPriority.LOW
    
    decision, explanation = service.make_decision(title, description, category, priority, mock_db)
    
    # Denied decisions should include alternative options
    if decision == DecisionType.DENIED:
        assert len(explanation.alternatives_considered) > 0
        assert "modifications" in explanation.alternatives_considered[0]["option"].lower()
    
    # Should include counterfactual scenarios
    assert len(explanation.counterfactuals) > 0


def test_critical_priority_escalation():
    """Test that critical priority requests get escalated for approval"""
    service = DecisionService()
    mock_db = Mock()
    
    title = "Critical security issue"
    description = "Critical network security incident requires immediate attention"
    category = TicketCategory.SECURITY
    priority = TicketPriority.CRITICAL
    
    decision, explanation = service.make_decision(title, description, category, priority, mock_db)
    
    # Critical security issues should require approval
    assert decision == DecisionType.NEEDS_APPROVAL
    assert any("critical" in step.rationale.lower() for step in explanation.reasoning_trace)


def test_explanation_structure():
    """Test that explanation object contains all required components"""
    service = DecisionService()
    mock_db = Mock()
    
    decision, explanation = service.make_decision(
        "Test request", "Test description", TicketCategory.OTHER, TicketPriority.MEDIUM, mock_db
    )
    
    # Verify explanation structure
    assert explanation.answer is not None
    assert explanation.decision is not None
    assert 0 <= explanation.confidence <= 1
    assert len(explanation.reasoning_trace) > 0
    assert isinstance(explanation.missing_info, list)
    assert isinstance(explanation.alternatives_considered, list)
    assert isinstance(explanation.counterfactuals, list)
    assert explanation.telemetry is not None
    assert explanation.timestamp is not None
    assert explanation.model_version == "1.0.0"


if __name__ == "__main__":
    # Run basic tests
    test_decision_service_security_violation()
    test_decision_service_requires_approval()
    test_decision_service_allowed()
    print("✅ All decision service tests passed!")
