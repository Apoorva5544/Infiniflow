"""
Tests for Advanced RAG Engine
"""
import pytest
from ai_engine.advanced_rag import AdvancedRAGEngine, RAGEvaluator
from unittest.mock import Mock, MagicMock


class TestAdvancedRAGEngine:
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store"""
        mock_vs = Mock()
        mock_vs.similarity_search = Mock(return_value=[
            Mock(page_content="Test content 1", metadata={"source": "doc1.pdf"}),
            Mock(page_content="Test content 2", metadata={"source": "doc2.pdf"})
        ])
        return mock_vs
    
    @pytest.fixture
    def rag_engine(self, mock_vector_store):
        """Create RAG engine instance"""
        return AdvancedRAGEngine(mock_vector_store)
    
    def test_multi_query_generation(self, rag_engine):
        """Test multi-query generation"""
        query = "What is machine learning?"
        queries = rag_engine.generate_multi_queries(query, num_queries=3)
        
        assert isinstance(queries, list)
        assert len(queries) >= 1
    
    def test_hypothetical_document_generation(self, rag_engine):
        """Test HyDE"""
        query = "Explain neural networks"
        hyp_doc = rag_engine.generate_hypothetical_document(query)
        
        assert isinstance(hyp_doc, str)
        assert len(hyp_doc) > 0
    
    def test_query_decomposition(self, rag_engine):
        """Test query decomposition"""
        complex_query = "Compare machine learning and deep learning, and explain their applications"
        sub_queries = rag_engine.decompose_query(complex_query)
        
        assert isinstance(sub_queries, list)
        assert len(sub_queries) >= 1
    
    def test_adaptive_retrieval_multi_query(self, rag_engine, mock_vector_store):
        """Test adaptive retrieval with multi-query strategy"""
        query = "What is AI?"
        docs = rag_engine.adaptive_retrieval(query, strategy="multi_query")
        
        assert isinstance(docs, list)
        assert len(docs) > 0
    
    def test_adaptive_retrieval_hyde(self, rag_engine, mock_vector_store):
        """Test adaptive retrieval with HyDE strategy"""
        query = "What is AI?"
        docs = rag_engine.adaptive_retrieval(query, strategy="hyde")
        
        assert isinstance(docs, list)
        assert len(docs) > 0
    
    def test_query_routing(self, rag_engine):
        """Test query routing"""
        simple_query = "What is the capital of France?"
        strategy = rag_engine.query_routing(simple_query)
        
        assert strategy in ["simple", "multi_query", "hyde", "decompose"]


class TestRAGEvaluator:
    
    def test_relevance_score_calculation(self):
        """Test relevance score calculation"""
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="8"))
        
        query = "What is AI?"
        documents = [
            Mock(page_content="AI is artificial intelligence"),
            Mock(page_content="Machine learning is a subset of AI")
        ]
        
        score = RAGEvaluator.calculate_relevance_score(query, documents, mock_llm)
        
        assert isinstance(score, float)
        assert 0 <= score <= 10
    
    def test_answer_quality_evaluation(self):
        """Test answer quality evaluation"""
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="7"))
        
        query = "What is AI?"
        answer = "AI is artificial intelligence"
        
        metrics = RAGEvaluator.calculate_answer_quality(query, answer, None, mock_llm)
        
        assert isinstance(metrics, dict)
        assert "relevance" in metrics
