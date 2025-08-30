import pytest
from datetime import datetime
from app.models.schemas import ExplanationObject, ReasoningStep, PolicyCitation, TelemetryData
from app.services.xai_service import XAIService

def test_explanation_object_schema():
    """Test that ExplanationObject schema validation works correctly"""
    
    # Valid explanation object
    valid_explanation = ExplanationObject(
        answer="Test answer",
        decision="test_decision=value",
        confidence=0.8,
        reasoning_trace=[
            ReasoningStep(
                step=1,
                action="test_action",
                rationale="Test rationale",
                confidence=0.9,
                policy_refs=[]
            )
        ],
        policy_citations=[],
        missing_info=[],
        alternatives_considered=[],
        counterfactuals=[],
        telemetry=TelemetryData(
            latency_ms=100,
            retrieval_k=5,
            triage_time_ms=50,
            planning_time_ms=50,
            total_chunks_considered=10
        ),
        timestamp=datetime.now(),
        model_version="1.0.0"
    )
    
    # Should not raise any validation errors
    assert valid_explanation.answer == "Test answer"
    assert valid_explanation.confidence == 0.8
    assert len(valid_explanation.reasoning_trace) == 1

def test_xai_service_schema_validation():
    """Test XAI service schema validation functionality"""
    
    xai_service = XAIService()
    
    # Valid explanation
    valid_explanation = ExplanationObject(
        answer="Valid explanation",
        decision="category=hardware",
        confidence=0.75,
        reasoning_trace=[
            ReasoningStep(
                step=1,
                action="classification",
                rationale="Classified based on keywords",
                confidence=0.8,
                policy_refs=[]
            )
        ],
        policy_citations=[],
        missing_info=[],
        alternatives_considered=[],
        counterfactuals=[],
        telemetry=TelemetryData(
            latency_ms=150,
            retrieval_k=3,
            triage_time_ms=100,
            planning_time_ms=50,
            total_chunks_considered=5
        ),
        timestamp=datetime.now()
    )
    
    validation = xai_service.validate_explanation_schema(valid_explanation)
    assert validation["is_valid"] == True
    assert len(validation["errors"]) == 0

def test_xai_service_schema_validation_errors():
    """Test XAI service detects schema validation errors"""
    
    xai_service = XAIService()
    
    # Invalid explanation (empty answer)
    invalid_explanation = ExplanationObject(
        answer="",  # Empty answer should trigger error
        decision="category=hardware",
        confidence=1.5,  # Invalid confidence > 1
        reasoning_trace=[],
        policy_citations=[],
        missing_info=[],
        alternatives_considered=[],
        counterfactuals=[],
        telemetry=TelemetryData(
            latency_ms=150,
            retrieval_k=3,
            triage_time_ms=100,
            planning_time_ms=50,
            total_chunks_considered=5
        ),
        timestamp=datetime.now()
    )
    
    validation = xai_service.validate_explanation_schema(invalid_explanation)
    assert validation["is_valid"] == False
    assert "Missing answer field" in validation["errors"]
    assert "Confidence must be between 0 and 1" in validation["errors"]

def test_reasoning_trace_sequence_validation():
    """Test that reasoning trace step numbering is validated"""
    
    xai_service = XAIService()
    
    # Out of sequence reasoning steps
    explanation = ExplanationObject(
        answer="Test answer",
        decision="test_decision",
        confidence=0.8,
        reasoning_trace=[
            ReasoningStep(step=1, action="step1", rationale="First step", confidence=0.9),
            ReasoningStep(step=3, action="step3", rationale="Third step", confidence=0.8),  # Missing step 2
            ReasoningStep(step=2, action="step2", rationale="Second step", confidence=0.7)   # Out of order
        ],
        policy_citations=[],
        missing_info=[],
        alternatives_considered=[],
        counterfactuals=[],
        telemetry=TelemetryData(
            latency_ms=100,
            retrieval_k=0,
            triage_time_ms=100,
            planning_time_ms=0,
            total_chunks_considered=0
        ),
        timestamp=datetime.now()
    )
    
    validation = xai_service.validate_explanation_schema(explanation)
    assert validation["is_valid"] == True  # Warnings don't make it invalid
    assert any("out of sequence" in warning for warning in validation["warnings"])

def test_policy_citation_schema():
    """Test PolicyCitation schema validation"""
    
    citation = PolicyCitation(
        document_id=1,
        document_title="Test Policy",
        chunk_id=1,
        chunk_content="Test policy content explaining procedures",
        relevance_score=0.85
    )
    
    assert citation.document_id == 1
    assert citation.relevance_score == 0.85
    assert "procedures" in citation.chunk_content

def test_telemetry_data_schema():
    """Test TelemetryData schema validation"""
    
    telemetry = TelemetryData(
        latency_ms=250,
        retrieval_k=5,
        triage_time_ms=150,
        planning_time_ms=100,
        total_chunks_considered=8
    )
    
    assert telemetry.latency_ms == 250
    assert telemetry.retrieval_k == 5
    assert telemetry.total_chunks_considered == 8

if __name__ == "__main__":
    pytest.main([__file__])
