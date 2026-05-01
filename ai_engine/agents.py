"""
AI Agents for Autonomous Task Execution
- Research Agent
- Data Analysis Agent
- Code Generation Agent
- Multi-Agent Orchestration
"""
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory
import os


class ResearchAgent:
    """Autonomous research agent for document analysis"""
    
    def __init__(self, vector_store, model: str = "llama-3.1-70b-versatile"):
        self.vector_store = vector_store
        self.llm = ChatGroq(
            temperature=0.7,
            model_name=model,
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip("\"' ")
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
    def create_tools(self) -> List[Tool]:
        """Create tools for the agent"""
        
        def search_documents(query: str) -> str:
            """Search through documents"""
            docs = self.vector_store.similarity_search(query, k=5)
            if not docs:
                return "No relevant documents found."
            
            result = "Found relevant information:\n\n"
            for i, doc in enumerate(docs, 1):
                result += f"{i}. {doc.page_content[:200]}...\n"
                result += f"   Source: {doc.metadata.get('source', 'Unknown')}\n\n"
            return result
        
        def summarize_topic(topic: str) -> str:
            """Summarize information about a topic"""
            docs = self.vector_store.similarity_search(topic, k=10)
            if not docs:
                return "No information found on this topic."
            
            combined_text = "\n".join([doc.page_content for doc in docs[:5]])
            
            prompt = f"""Summarize the following information about {topic}:
            
            {combined_text[:2000]}
            
            Provide a comprehensive summary."""
            
            response = self.llm.invoke(prompt)
            return response.content
        
        def compare_concepts(concepts: str) -> str:
            """Compare multiple concepts"""
            concept_list = [c.strip() for c in concepts.split(",")]
            if len(concept_list) < 2:
                return "Please provide at least 2 concepts separated by commas."
            
            results = {}
            for concept in concept_list:
                docs = self.vector_store.similarity_search(concept, k=3)
                results[concept] = "\n".join([doc.page_content[:300] for doc in docs])
            
            comparison_text = "\n\n".join([
                f"**{concept}**:\n{text}" 
                for concept, text in results.items()
            ])
            
            prompt = f"""Compare and contrast these concepts:
            
            {comparison_text[:2000]}
            
            Provide a detailed comparison."""
            
            response = self.llm.invoke(prompt)
            return response.content
        
        return [
            Tool(
                name="SearchDocuments",
                func=search_documents,
                description="Search through documents for specific information. Input should be a search query."
            ),
            Tool(
                name="SummarizeTopic",
                func=summarize_topic,
                description="Get a comprehensive summary of a topic from the documents. Input should be a topic name."
            ),
            Tool(
                name="CompareConcepts",
                func=compare_concepts,
                description="Compare multiple concepts. Input should be comma-separated concept names."
            )
        ]
    
    def run(self, task: str) -> Dict[str, Any]:
        """Execute research task"""
        tools = self.create_tools()
        
        prompt = PromptTemplate.from_template(
            """You are an expert research assistant. Use the available tools to complete the task.
            
            Available tools:
            {tools}
            
            Tool Names: {tool_names}
            
            Task: {input}
            
            Think step by step:
            {agent_scratchpad}
            
            What should I do next?"""
        )
        
        agent = create_react_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5,
            memory=self.memory
        )
        
        try:
            result = agent_executor.invoke({"input": task})
            return {
                "success": True,
                "output": result.get("output", ""),
                "steps": result.get("intermediate_steps", [])
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": ""
            }


class DataAnalysisAgent:
    """Agent for analyzing data patterns in documents"""
    
    def __init__(self, vector_store, model: str = "llama-3.1-70b-versatile"):
        self.vector_store = vector_store
        self.llm = ChatGroq(
            temperature=0.3,
            model_name=model,
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip("\"' ")
        )
    
    def analyze_patterns(self, query: str) -> Dict[str, Any]:
        """Analyze patterns in document data"""
        docs = self.vector_store.similarity_search(query, k=20)
        
        if not docs:
            return {"patterns": [], "insights": "No data found."}
        
        # Extract metadata patterns
        sources = {}
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            sources[source] = sources.get(source, 0) + 1
        
        # Analyze content
        combined_text = "\n".join([doc.page_content for doc in docs[:10]])
        
        prompt = f"""Analyze the following data and identify key patterns, trends, and insights:
        
        {combined_text[:3000]}
        
        Provide:
        1. Key patterns identified
        2. Important trends
        3. Actionable insights
        4. Data quality assessment
        
        Format as structured analysis."""
        
        response = self.llm.invoke(prompt)
        
        return {
            "patterns": list(sources.keys()),
            "source_distribution": sources,
            "insights": response.content,
            "total_documents": len(docs)
        }
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text"""
        prompt = f"""Extract all important entities (people, organizations, locations, concepts) from this text:
        
        {text[:2000]}
        
        Return as a JSON array of strings."""
        
        response = self.llm.invoke(prompt)
        try:
            import json
            entities = json.loads(response.content)
            return entities if isinstance(entities, list) else []
        except:
            return []


class MultiAgentOrchestrator:
    """Orchestrate multiple AI agents for complex tasks"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.research_agent = ResearchAgent(vector_store)
        self.analysis_agent = DataAnalysisAgent(vector_store)
    
    def route_task(self, task: str, task_type: str = "auto") -> Dict[str, Any]:
        """Route task to appropriate agent"""
        
        if task_type == "auto":
            # Auto-detect task type
            if any(word in task.lower() for word in ["analyze", "pattern", "trend", "insight"]):
                task_type = "analysis"
            else:
                task_type = "research"
        
        if task_type == "research":
            return self.research_agent.run(task)
        elif task_type == "analysis":
            return self.analysis_agent.analyze_patterns(task)
        else:
            return {"error": "Unknown task type"}
    
    def collaborative_task(self, task: str) -> Dict[str, Any]:
        """Execute task using multiple agents collaboratively"""
        
        # Step 1: Research agent gathers information
        research_result = self.research_agent.run(f"Research: {task}")
        
        # Step 2: Analysis agent analyzes the findings
        if research_result.get("success"):
            analysis_result = self.analysis_agent.analyze_patterns(task)
            
            return {
                "research": research_result.get("output", ""),
                "analysis": analysis_result.get("insights", ""),
                "patterns": analysis_result.get("patterns", []),
                "collaborative": True
            }
        
        return research_result
