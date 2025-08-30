import os
import hashlib
import json
from typing import List, Dict, Tuple, Set, Optional, Any
from datetime import datetime
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sqlalchemy.orm import Session
from app.models.models import PolicyDocument, PolicyChunk, KnowledgeGraphConcept
from app.models.schemas import PolicyCitation, KGEnhancedPolicyCitation, GraphHop
from app.core.database import get_db
from app.services.knowledge_graph_builder import PolicyKnowledgeGraphBuilder
from app.services.knowledge_graph_query import KnowledgeGraphQueryService

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class PolicyRetriever:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.bm25 = None
        self.chunks_data = []
        self.is_initialized = False
        # Initialize KG services
        self.kg_builder = PolicyKnowledgeGraphBuilder()
        self.kg_query = KnowledgeGraphQueryService()
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 and TF-IDF"""
        # Tokenize and remove stopwords
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        return tokens
    
    def initialize_retrievers(self, db: Session):
        """Initialize BM25 and TF-IDF models with policy chunks from database"""
        chunks = db.query(PolicyChunk).join(PolicyDocument).all()
        
        if not chunks:
            print("No policy chunks found in database")
            return
        
        # Prepare data
        self.chunks_data = []
        corpus = []
        
        for chunk in chunks:
            chunk_data = {
                'id': chunk.id,
                'document_id': chunk.document_id,
                'document_title': chunk.document.title,
                'content': chunk.content,
                'chunk_index': chunk.chunk_index
            }
            self.chunks_data.append(chunk_data)
            
            # Preprocess for BM25
            tokens = self.preprocess_text(chunk.content)
            corpus.append(tokens)
        
        # Initialize BM25
        if corpus:
            self.bm25 = BM25Okapi(corpus)
            
            # Initialize TF-IDF
            text_corpus = [chunk['content'] for chunk in self.chunks_data]
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_corpus)
            
            self.is_initialized = True
            print(f"Initialized retrievers with {len(chunks)} policy chunks")
    
    def retrieve_relevant_chunks(
        self, 
        query: str, 
        k: int = 5, 
        alpha: float = 0.6,
        db: Session = None
    ) -> List[PolicyCitation]:
        """
        Retrieve most relevant policy chunks using hybrid BM25 + TF-IDF approach
        
        Args:
            query: Search query
            k: Number of top chunks to retrieve
            alpha: Weight for BM25 vs TF-IDF (alpha=1.0 means pure BM25, alpha=0.0 means pure TF-IDF)
            db: Database session
        
        Returns:
            List of PolicyCitation objects with relevance scores
        """
        if not self.is_initialized:
            if db:
                self.initialize_retrievers(db)
            else:
                return []
        
        if not self.chunks_data:
            return []
        
        start_time = datetime.now()
        
        # BM25 scoring
        query_tokens = self.preprocess_text(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # TF-IDF scoring
        query_vector = self.tfidf_vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Hybrid scoring (normalize and combine)
        bm25_normalized = bm25_scores / (np.max(bm25_scores) + 1e-6)
        tfidf_normalized = tfidf_scores / (np.max(tfidf_scores) + 1e-6)
        
        hybrid_scores = alpha * bm25_normalized + (1 - alpha) * tfidf_normalized
        
        # Get top k results with deterministic ordering (sort by score desc, then by chunk id)
        scored_chunks = list(zip(hybrid_scores, range(len(self.chunks_data))))
        scored_chunks.sort(key=lambda x: (-x[0], self.chunks_data[x[1]]['id']))
        top_chunks = scored_chunks[:k]
        
        # Build citations
        citations = []
        for score, idx in top_chunks:
            chunk_data = self.chunks_data[idx]
            citation = PolicyCitation(
                document_id=chunk_data['document_id'],
                document_title=chunk_data['document_title'],
                chunk_id=chunk_data['id'],
                chunk_content=chunk_data['content'],
                relevance_score=float(score)
            )
            citations.append(citation)
        
        end_time = datetime.now()
        retrieval_time = int((end_time - start_time).total_seconds() * 1000)
        
        return citations
    
    def get_chunk_by_id(self, chunk_id: int, db: Session) -> Dict:
        """Get specific chunk by ID"""
        chunk = db.query(PolicyChunk).filter(PolicyChunk.id == chunk_id).first()
        if chunk:
            return {
                'id': chunk.id,
                'document_id': chunk.document_id,
                'document_title': chunk.document.title,
                'content': chunk.content,
                'chunk_index': chunk.chunk_index
            }
        return None
    
    def search_policies_by_category(
        self, 
        category: str, 
        query: str = "", 
        k: int = 3,
        db: Session = None
    ) -> List[PolicyCitation]:
        """
        Search for policies relevant to a specific ticket category
        """
        category_query = f"{category} {query}".strip()
        return self.retrieve_relevant_chunks(category_query, k, db=db)
    
    def kg_enhanced_retrieve(
        self,
        query: str,
        k: int = 7,  # Get more initial results to allow for graph expansion
        alpha: float = 0.6,
        enable_kg: bool = True,
        max_graph_hops: int = 2,
        min_relationship_weight: float = 0.5,
        db: Session = None
    ) -> Tuple[List[KGEnhancedPolicyCitation], List[GraphHop], Dict[str, Any]]:
        """
        KG-Enhanced RAG: Semantic search + Knowledge Graph traversal
        
        This is the main method for the KG-Enhanced RAG system. It:
        1. Performs semantic search in the vector store
        2. Identifies IT concepts in the query
        3. Traverses the knowledge graph to find related concepts
        4. Retrieves additional relevant policies based on graph connections
        5. Combines and ranks all results
        
        Args:
            query: User query text
            k: Number of initial semantic search results
            alpha: BM25 vs TF-IDF weight
            enable_kg: Whether to use knowledge graph enhancement
            max_graph_hops: Maximum graph traversal depth
            min_relationship_weight: Minimum weight for graph relationships
            db: Database session
            
        Returns:
            Tuple of (enhanced_citations, graph_hops, metadata)
        """
        if not db:
            return [], [], {}
        
        start_time = datetime.now()
        
        # Step 1: Perform semantic search in vector store
        semantic_citations = self.retrieve_relevant_chunks(query, k, alpha, db)
        
        if not enable_kg or not semantic_citations:
            # Fall back to semantic-only results
            enhanced_citations = [
                KGEnhancedPolicyCitation(
                    document_id=c.document_id,
                    document_title=c.document_title,
                    chunk_id=c.chunk_id,
                    chunk_content=c.chunk_content,
                    relevance_score=c.relevance_score,
                    graph_path=[],
                    semantic_score=c.relevance_score,
                    graph_boost_score=0.0,
                    combined_score=c.relevance_score
                ) for c in semantic_citations
            ]
            
            return enhanced_citations, [], {
                "processing_time_ms": int((datetime.now() - start_time).total_seconds() * 1000),
                "kg_enabled": False,
                "semantic_only": True
            }
        
        # Step 2: Identify IT concepts in the query
        initial_concepts_with_scores = self.kg_builder.find_concepts_in_text(query)
        initial_concepts = [concept for concept, score in initial_concepts_with_scores if score > 0.5]
        
        if not initial_concepts:
            # No concepts found, return semantic results only
            enhanced_citations = [
                KGEnhancedPolicyCitation(
                    document_id=c.document_id,
                    document_title=c.document_title,
                    chunk_id=c.chunk_id,
                    chunk_content=c.chunk_content,
                    relevance_score=c.relevance_score,
                    graph_path=[],
                    semantic_score=c.relevance_score,
                    graph_boost_score=0.0,
                    combined_score=c.relevance_score
                ) for c in semantic_citations
            ]
            
            return enhanced_citations, [], {
                "processing_time_ms": int((datetime.now() - start_time).total_seconds() * 1000),
                "kg_enabled": True,
                "concepts_found": False
            }
        
        # Step 3: Traverse knowledge graph to find related concepts and policies
        enhanced_citations, graph_hops, kg_metadata = self.kg_query.query_graph_for_related_policies(
            db=db,
            initial_concepts=initial_concepts,
            semantic_citations=semantic_citations,
            max_hops=max_graph_hops,
            min_weight=min_relationship_weight
        )
        
        # Step 4: Limit final results to reasonable number
        final_k = min(k + 3, len(enhanced_citations))  # Allow a few extra from graph
        enhanced_citations = enhanced_citations[:final_k]
        
        end_time = datetime.now()
        total_time = int((end_time - start_time).total_seconds() * 1000)
        
        # Combine metadata
        combined_metadata = {
            "total_processing_time_ms": total_time,
            "kg_enabled": True,
            "initial_concepts_found": initial_concepts,
            "semantic_citations_count": len(semantic_citations),
            "final_citations_count": len(enhanced_citations),
            "graph_expansion_ratio": len(enhanced_citations) / max(len(semantic_citations), 1),
            **kg_metadata
        }
        
        return enhanced_citations, graph_hops, combined_metadata
    
    def kg_enhanced_search_by_category(
        self,
        category: str,
        query: str = "",
        k: int = 5,
        enable_kg: bool = True,
        db: Session = None
    ) -> Tuple[List[KGEnhancedPolicyCitation], List[GraphHop], Dict[str, Any]]:
        """
        Category-specific search with KG enhancement
        """
        category_query = f"{category} {query}".strip()
        return self.kg_enhanced_retrieve(
            query=category_query,
            k=k,
            enable_kg=enable_kg,
            db=db
        )
    
    def explain_retrieval_reasoning(
        self,
        query: str,
        enhanced_citations: List[KGEnhancedPolicyCitation],
        graph_hops: List[GraphHop],
        metadata: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable explanation of the KG-Enhanced RAG retrieval process
        """
        explanation = f"KG-Enhanced RAG Retrieval for: '{query}'\n\n"
        
        # Semantic search summary
        semantic_count = sum(1 for c in enhanced_citations if c.semantic_score > 0)
        graph_count = len(enhanced_citations) - semantic_count
        
        explanation += f"📊 Retrieval Summary:\n"
        explanation += f"   • Semantic search found: {semantic_count} relevant policies\n"
        explanation += f"   • Graph traversal added: {graph_count} related policies\n"
        explanation += f"   • Total processing time: {metadata.get('total_processing_time_ms', 0)}ms\n\n"
        
        # Concepts found
        if 'initial_concepts_found' in metadata:
            explanation += f"🔍 IT Concepts Identified: {', '.join(metadata['initial_concepts_found'])}\n\n"
        
        # Graph traversal explanation
        if graph_hops:
            explanation += "🕸️ Knowledge Graph Traversal:\n"
            hop_num = 1
            for hop in graph_hops[:5]:  # Limit to first 5 hops for readability
                explanation += f"   {hop_num}. {hop.from_concept} → {hop.to_concept} "
                explanation += f"(via '{hop.relationship_type}')\n"
                hop_num += 1
            
            if len(graph_hops) > 5:
                explanation += f"   ... and {len(graph_hops) - 5} more graph connections\n"
            explanation += "\n"
        
        # Top citations with reasoning
        explanation += "📄 Top Policy Citations:\n"
        for i, citation in enumerate(enhanced_citations[:3], 1):
            explanation += f"   {i}. {citation.document_title} (Score: {citation.combined_score:.3f})\n"
            if citation.graph_path:
                explanation += f"      → Found via graph: {citation.graph_path[0].reasoning}\n"
            else:
                explanation += f"      → Found via semantic search\n"
        
        return explanation
    
    def get_related_concepts_for_query(self, query: str, db: Session) -> List[Tuple[str, float]]:
        """
        Get IT concepts mentioned in a query along with confidence scores
        """
        return self.kg_builder.find_concepts_in_text(query)
