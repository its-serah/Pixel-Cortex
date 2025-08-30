import pytest
from app.services.triage_service import TriageService
from app.services.xai_service import XAIService
from app.utils.privacy import DeterminismUtils, ReproducibilityValidator
from app.models.models import TicketCategory, TicketPriority

def test_triage_determinism():
    """Test that triage produces deterministic results"""
    triage_service = TriageService()
    
    title = "Computer won't start"
    description = "My laptop computer will not turn on when I press the power button"
    
    # Run triage multiple times
    result1 = triage_service.triage_ticket(title, description)
    result2 = triage_service.triage_ticket(title, description)
    result3 = triage_service.triage_ticket(title, description)
    
    # Results should be identical
    assert result1[0] == result2[0] == result3[0]  # Category
    assert result1[1] == result2[1] == result3[1]  # Priority
    assert result1[2] == result2[2] == result3[2]  # Confidence
    
    # Explanations should have same structure
    assert result1[3].decision == result2[3].decision == result3[3].decision
    assert len(result1[3].reasoning_trace) == len(result2[3].reasoning_trace) == len(result3[3].reasoning_trace)

def test_determinism_utils_hash_consistency():
    """Test determinism utilities produce consistent hashes"""
    utils = DeterminismUtils()
    
    data1 = {"key1": "value1", "key2": {"nested": "value"}}
    data2 = {"key2": {"nested": "value"}, "key1": "value1"}  # Different order
    data3 = {"key1": "value1", "key2": {"nested": "different"}}
    
    hash1 = utils.calculate_content_hash(utils.normalize_dict_for_hashing(data1))
    hash2 = utils.calculate_content_hash(utils.normalize_dict_for_hashing(data2))
    hash3 = utils.calculate_content_hash(utils.normalize_dict_for_hashing(data3))
    
    assert hash1 == hash2  # Same data, different order should produce same hash
    assert hash1 != hash3  # Different data should produce different hash

def test_determinism_utils_seed_generation():
    """Test deterministic seed generation"""
    utils = DeterminismUtils()
    
    input1 = "test input string"
    input2 = "test input string"
    input3 = "different input string"
    
    seed1 = utils.generate_deterministic_seed(input1)
    seed2 = utils.generate_deterministic_seed(input2)
    seed3 = utils.generate_deterministic_seed(input3)
    
    assert seed1 == seed2  # Same input should produce same seed
    assert seed1 != seed3  # Different input should produce different seed
    assert isinstance(seed1, int)

def test_deterministic_list_sorting():
    """Test deterministic list sorting"""
    utils = DeterminismUtils()
    
    items1 = [{"id": 3, "name": "c"}, {"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    items2 = [{"id": 2, "name": "b"}, {"id": 3, "name": "c"}, {"id": 1, "name": "a"}]  # Different order
    
    sorted1 = utils.ensure_deterministic_ordering(items1)
    sorted2 = utils.ensure_deterministic_ordering(items2)
    
    assert sorted1 == sorted2  # Should produce same ordering regardless of input order
    assert sorted1[0]["id"] == 1  # Should be sorted by id
    assert sorted1[1]["id"] == 2
    assert sorted1[2]["id"] == 3

def test_reproducibility_validator():
    """Test reproducibility validation"""
    validator = ReproducibilityValidator()
    
    input_data = {"title": "Test ticket", "description": "Test description"}
    output_data1 = {"category": "hardware", "priority": "high", "confidence": 0.8}
    output_data2 = {"category": "hardware", "priority": "high", "confidence": 0.8}
    output_data3 = {"category": "software", "priority": "medium", "confidence": 0.7}
    
    # Same input + output should validate as deterministic
    validation1 = validator.validate_deterministic_output(input_data, output_data1, output_data2)
    assert validation1["is_deterministic"] == True
    
    # Same input, different output should fail validation
    validation2 = validator.validate_deterministic_output(input_data, output_data1, output_data3)
    assert validation2["is_deterministic"] == False
    assert len(validation2["errors"]) > 0

def test_replay_signature_generation():
    """Test replay signature generation for determinism validation"""
    validator = ReproducibilityValidator()
    
    input_data = {"title": "Test", "description": "Test desc"}
    output_data = {"result": "test_result", "timestamp": "2024-01-01T00:00:00"}
    
    signature1 = validator.create_replay_signature(input_data, output_data)
    signature2 = validator.create_replay_signature(input_data, output_data)
    
    assert signature1 == signature2  # Same data should produce same signature
    assert len(signature1) == 64  # SHA-256 hex string length

def test_explanation_determinism_integration(db_session):
    """Test end-to-end determinism in XAI explanations"""
    xai_service = XAIService()
    
    title = "Printer not working"
    description = "Office printer shows error message and won't print documents"
    
    # Generate explanations multiple times
    result1 = xai_service.explain_ticket_processing(title, description, db_session, include_planning=False)
    result2 = xai_service.explain_ticket_processing(title, description, db_session, include_planning=False)
    
    # Core decisions should be deterministic
    assert result1["category"] == result2["category"]
    assert result1["priority"] == result2["priority"]
    assert result1["confidence"] == result2["confidence"]
    
    # Explanation structure should be consistent
    exp1 = result1["explanation"]
    exp2 = result2["explanation"]
    
    assert exp1.decision == exp2.decision
    assert len(exp1.reasoning_trace) == len(exp2.reasoning_trace)
    
    # Individual reasoning steps should have same actions and rationales
    for step1, step2 in zip(exp1.reasoning_trace, exp2.reasoning_trace):
        assert step1.action == step2.action
        assert step1.step == step2.step
        # Note: rationales might contain timestamps, so we check key components
        assert step1.confidence == step2.confidence

if __name__ == "__main__":
    pytest.main([__file__])
