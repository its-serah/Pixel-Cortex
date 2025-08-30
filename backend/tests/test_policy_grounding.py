import pytest
from app.services.policy_retriever import PolicyRetriever
from app.services.policy_indexer import PolicyIndexer
from app.models.models import PolicyDocument, PolicyChunk
from app.models.schemas import PolicyCitation

def test_policy_retriever_initialization(db_session):
    """Test policy retriever can be initialized"""
    retriever = PolicyRetriever()
    assert retriever.stop_words is not None
    assert not retriever.is_initialized

def test_policy_indexer_text_chunking():
    """Test policy indexer chunks text correctly"""
    indexer = PolicyIndexer(chunk_size=100)
    
    text = "This is a test document. It has multiple sentences. Each sentence should be preserved. The chunking should be sentence-aware and not break in the middle of sentences."
    
    chunks = indexer.chunk_text(text, chunk_size=50)
    
    assert len(chunks) > 1
    # Each chunk should end with a period or be the last chunk
    for chunk in chunks[:-1]:
        assert chunk.endswith('.')

def test_policy_indexer_hash_calculation():
    """Test content hashing is deterministic"""
    indexer = PolicyIndexer()
    
    content1 = "This is test content"
    content2 = "This is test content"
    content3 = "This is different content"
    
    hash1 = indexer.calculate_hash(content1)
    hash2 = indexer.calculate_hash(content2)
    hash3 = indexer.calculate_hash(content3)
    
    assert hash1 == hash2  # Same content should produce same hash
    assert hash1 != hash3  # Different content should produce different hash

def test_policy_document_indexing(db_session):
    """Test policy document is properly indexed in database"""
    
    # Create a sample policy document
    doc = PolicyDocument(
        filename="test_policy.md",
        title="Test Policy",
        content="This is a test policy document. It contains important procedures. Follow these steps carefully.",
        content_hash="test_hash",
        file_type="md"
    )
    
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    
    # Create chunks
    chunk1 = PolicyChunk(
        document_id=doc.id,
        chunk_index=0,
        content="This is a test policy document. It contains important procedures.",
        content_hash="chunk1_hash"
    )
    
    chunk2 = PolicyChunk(
        document_id=doc.id,
        chunk_index=1,
        content="Follow these steps carefully.",
        content_hash="chunk2_hash"
    )
    
    db_session.add(chunk1)
    db_session.add(chunk2)
    db_session.commit()
    
    # Verify indexing
    retrieved_doc = db_session.query(PolicyDocument).filter(PolicyDocument.id == doc.id).first()
    assert retrieved_doc is not None
    assert retrieved_doc.title == "Test Policy"
    assert len(retrieved_doc.chunks) == 2

def test_policy_retriever_with_documents(db_session):
    """Test policy retrieval with actual documents"""
    
    # Create test documents and chunks
    doc = PolicyDocument(
        filename="hardware_test.md",
        title="Hardware Support Policy",
        content="Computer troubleshooting procedures for desktop and laptop issues",
        content_hash="hardware_hash",
        file_type="md"
    )
    
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    
    chunk = PolicyChunk(
        document_id=doc.id,
        chunk_index=0,
        content="When computer won't turn on, check power connections first. Then test with different power cable.",
        content_hash="chunk_hash"
    )
    
    db_session.add(chunk)
    db_session.commit()
    
    # Test retrieval
    retriever = PolicyRetriever()
    retriever.initialize_retrievers(db_session)
    
    citations = retriever.retrieve_relevant_chunks("computer power issue", k=1, db=db_session)
    
    assert len(citations) >= 1
    assert citations[0].document_title == "Hardware Support Policy"
    assert "power" in citations[0].chunk_content.lower()

def test_policy_grounding_faithfulness(db_session):
    """Test that policy citations are faithful to source content"""
    
    # Create test policy with specific content
    doc = PolicyDocument(
        filename="specific_policy.md",
        title="Specific IT Policy",
        content="For hardware issues: Step 1 is power check. Step 2 is cable verification. Step 3 is component testing.",
        content_hash="specific_hash",
        file_type="md"
    )
    
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    
    chunk = PolicyChunk(
        document_id=doc.id,
        chunk_index=0,
        content="For hardware issues: Step 1 is power check. Step 2 is cable verification. Step 3 is component testing.",
        content_hash="specific_chunk_hash"
    )
    
    db_session.add(chunk)
    db_session.commit()
    
    # Test retrieval and verify faithfulness
    retriever = PolicyRetriever()
    retriever.initialize_retrievers(db_session)
    
    citations = retriever.retrieve_relevant_chunks("hardware troubleshooting steps", k=1, db=db_session)
    
    assert len(citations) >= 1
    citation = citations[0]
    
    # Verify the citation content matches exactly what's in the database
    db_chunk = db_session.query(PolicyChunk).filter(PolicyChunk.id == citation.chunk_id).first()
    assert db_chunk is not None
    assert citation.chunk_content == db_chunk.content
    
    # Verify no hallucination - citation should contain actual policy steps
    assert "Step 1 is power check" in citation.chunk_content
    assert "Step 2 is cable verification" in citation.chunk_content

if __name__ == "__main__":
    pytest.main([__file__])
