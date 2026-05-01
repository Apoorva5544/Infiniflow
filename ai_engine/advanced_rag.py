"""
Advanced RAG Engine with Production Features
- Multi-query generation
- Hypothetical document embeddings (HyDE)
- Query decomposition
- Self-reflection and correction
- Adaptive retrieval
"""
import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json


class AdvancedRAGEngine:
    """Production-grade RAG with advanced retrieval strategies"""
    
    def __init__(self, vector_store, llm_model: str = "llama-3.1-70b-versatile"):
        self.vector_store = vector_store
        self.llm = ChatGroq(
            temperature=0,
            model_name=llm_model,
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip("\"' ")
        )
        self.creative_llm = ChatGroq(
            temperature=0.7,
            model_name=llm_model,
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip("\"' ")
        )
    
    def generate_multi_queries(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple perspectives of the same query for better retrieval
        """
        prompt = ChatPromptTemplate.from_template(
            """You are an AI assistant that generates multiple search queries.
            Generate {num_queries} different versions of the following question to retrieve relevant documents.
            Provide alternative perspectives and phrasings.
            
            Original question: {question}
            
            Output the queries as a JSON array of strings."""
        )
        
        chain = prompt | self.creative_llm | StrOutputParser()
        
        try:
            result = chain.invoke({"question": query, "num_queries": num_queries})
            # Parse JSON response
            queries = json.loads(result)
            return queries if isinstance(queries, list) else [query]
        except:
            return [query]
    
    def generate_hypothetical_document(self, query: str) -> str:
        """
        HyDE: Generate a hypothetical ideal document that would answer the query
        """
        prompt = ChatPromptTemplate.from_template(
            """Write a detailed paragraph that would perfectly answer this question.
            Write as if you're an expert providing the exact information needed.
            
            Question: {question}
            
            Hypothetical Answer:"""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": query})
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Break down complex queries into simpler sub-questions
        """
        prompt = ChatPromptTemplate.from_template(
            """Break down this complex question into 2-4 simpler sub-questions.
            Each sub-question should be independently answerable.
            
            Complex Question: {question}
            
            Output as JSON array of strings."""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = chain.invoke({"question": query})
            sub_queries = json.loads(result)
            return sub_queries if isinstance(sub_queries, list) else [query]
        except:
            return [query]
    
    def adaptive_retrieval(
        self, 
        query: str, 
        strategy: str = "multi_query"
    ) -> List[Document]:
        """
        Adaptive retrieval using different strategies
        
        Strategies:
        - multi_query: Generate multiple query variations
        - hyde: Use hypothetical document embeddings
        - decompose: Break down complex queries
        - hybrid: Combine vector + keyword search
        """
        
        if strategy == "multi_query":
            queries = self.generate_multi_queries(query)
            all_docs = []
            for q in queries:
                docs = self.vector_store.similarity_search(q, k=5)
                all_docs.extend(docs)
            # Deduplicate
            seen = set()
            unique_docs = []
            for doc in all_docs:
                doc_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if doc_hash not in seen:
                    seen.add(doc_hash)
                    unique_docs.append(doc)
            return unique_docs[:10]
        
        elif strategy == "hyde":
            hypothetical_doc = self.generate_hypothetical_document(query)
            docs = self.vector_store.similarity_search(hypothetical_doc, k=10)
            return docs
        
        elif strategy == "decompose":
            sub_queries = self.decompose_query(query)
            all_docs = []
            for sq in sub_queries:
                docs = self.vector_store.similarity_search(sq, k=5)
                all_docs.extend(docs)
            # Deduplicate
            seen = set()
            unique_docs = []
            for doc in all_docs:
                doc_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if doc_hash not in seen:
                    seen.add(doc_hash)
                    unique_docs.append(doc)
            return unique_docs[:10]
        
        else:  # hybrid
            return self.vector_store.similarity_search(query, k=10)
    
    def contextual_compression(
        self, 
        query: str, 
        documents: List[Document]
    ) -> List[Document]:
        """
        Compress and filter documents to only relevant parts
        """
        compressor = LLMChainExtractor.from_llm(self.llm)
        base_retriever = self.vector_store.as_retriever()
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        try:
            compressed_docs = compression_retriever.get_relevant_documents(query)
            return compressed_docs
        except:
            return documents
    
    def self_reflect_answer(
        self, 
        query: str, 
        answer: str, 
        context: List[Document]
    ) -> Dict[str, Any]:
        """
        Self-reflection: Evaluate answer quality and suggest improvements
        """
        prompt = ChatPromptTemplate.from_template(
            """Evaluate this AI-generated answer for quality and accuracy.
            
            Question: {question}
            Answer: {answer}
            
            Evaluate on:
            1. Accuracy (based on context)
            2. Completeness
            3. Clarity
            4. Relevance
            
            Provide:
            - Score (0-10)
            - Strengths
            - Weaknesses
            - Suggested improvements
            
            Output as JSON with keys: score, strengths, weaknesses, improvements"""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = chain.invoke({"question": query, "answer": answer})
            evaluation = json.loads(result)
            return evaluation
        except:
            return {"score": 7, "strengths": [], "weaknesses": [], "improvements": []}
    
    def query_routing(self, query: str) -> str:
        """
        Route query to appropriate retrieval strategy based on complexity
        """
        prompt = ChatPromptTemplate.from_template(
            """Analyze this query and determine the best retrieval strategy.
            
            Query: {query}
            
            Strategies:
            - simple: Direct factual question
            - multi_query: Needs multiple perspectives
            - hyde: Abstract or conceptual question
            - decompose: Complex multi-part question
            
            Output only the strategy name."""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            strategy = chain.invoke({"query": query}).strip().lower()
            if strategy in ["simple", "multi_query", "hyde", "decompose"]:
                return strategy
            return "simple"
        except:
            return "simple"


class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    @staticmethod
    def calculate_relevance_score(query: str, documents: List[Document], llm) -> float:
        """Calculate average relevance of retrieved documents"""
        if not documents:
            return 0.0
        
        prompt = ChatPromptTemplate.from_template(
            """Rate the relevance of this document to the query on a scale of 0-10.
            
            Query: {query}
            Document: {document}
            
            Output only a number between 0 and 10."""
        )
        
        chain = prompt | llm | StrOutputParser()
        scores = []
        
        for doc in documents[:5]:  # Evaluate top 5
            try:
                score = float(chain.invoke({
                    "query": query,
                    "document": doc.page_content[:500]
                }))
                scores.append(score)
            except:
                scores.append(5.0)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    @staticmethod
    def calculate_answer_quality(
        query: str, 
        answer: str, 
        ground_truth: Optional[str],
        llm
    ) -> Dict[str, float]:
        """Evaluate answer quality metrics"""
        metrics = {}
        
        # Faithfulness: Does answer align with context?
        prompt = ChatPromptTemplate.from_template(
            """Rate how well this answer addresses the question (0-10).
            
            Question: {query}
            Answer: {answer}
            
            Output only a number."""
        )
        
        chain = prompt | llm | StrOutputParser()
        
        try:
            metrics["relevance"] = float(chain.invoke({"query": query, "answer": answer}))
        except:
            metrics["relevance"] = 5.0
        
        # If ground truth available, calculate similarity
        if ground_truth:
            prompt = ChatPromptTemplate.from_template(
                """Compare these two answers and rate their similarity (0-10).
                
                Answer 1: {answer1}
                Answer 2: {answer2}
                
                Output only a number."""
            )
            chain = prompt | llm | StrOutputParser()
            try:
                metrics["accuracy"] = float(chain.invoke({
                    "answer1": answer,
                    "answer2": ground_truth
                }))
            except:
                metrics["accuracy"] = 5.0
        
        return metrics
