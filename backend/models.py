from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text, Float, Boolean, JSON, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import enum

Base = declarative_base()


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    VIEWER = "viewer"


class QueryStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    role = Column(String, default=UserRole.USER.value)
    is_active = Column(Boolean, default=True)
    api_key = Column(String, unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Usage tracking
    total_queries = Column(Integer, default=0)
    total_documents = Column(Integer, default=0)

    workspaces = relationship("Workspace", back_populates="owner")
    queries = relationship("QueryLog", back_populates="user")
    api_usage = relationship("APIUsage", back_populates="user")


class Workspace(Base):
    __tablename__ = "workspaces"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # AI Configuration
    llm_model = Column(String, default="llama-3.1-70b-versatile")
    temperature = Column(Float, default=0.0)
    max_tokens = Column(Integer, default=2048)
    
    # Analytics
    total_documents = Column(Integer, default=0)
    total_queries = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)

    owner = relationship("User", back_populates="workspaces")
    documents = relationship("Document", back_populates="workspace")
    queries = relationship("QueryLog", back_populates="workspace")


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    file_path = Column(String)
    file_type = Column(String)
    file_size = Column(Integer)  # in bytes
    workspace_id = Column(Integer, ForeignKey("workspaces.id"))
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    # Processing metadata
    chunk_count = Column(Integer, default=0)
    processing_time = Column(Float, nullable=True)
    status = Column(String, default="processed")
    
    # Content metadata
    doc_metadata = Column(JSON, nullable=True)

    workspace = relationship("Workspace", back_populates="documents")


class QueryLog(Base):
    __tablename__ = "query_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    workspace_id = Column(Integer, ForeignKey("workspaces.id"))
    
    # Query details
    query_text = Column(Text)
    response_text = Column(Text)
    status = Column(String, default=QueryStatus.COMPLETED.value)
    
    # Performance metrics
    latency_ms = Column(Float)
    tokens_used = Column(Integer, default=0)
    documents_retrieved = Column(Integer, default=0)
    
    # AI metadata
    model_used = Column(String)
    retrieval_score = Column(Float, nullable=True)
    
    # Context
    sources = Column(JSON, nullable=True)
    query_metadata = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="queries")
    workspace = relationship("Workspace", back_populates="queries")


class APIUsage(Base):
    __tablename__ = "api_usage"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    endpoint = Column(String)
    method = Column(String)
    status_code = Column(Integer)
    response_time_ms = Column(Float)
    
    # Request details
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="api_usage")


class AIAgent(Base):
    __tablename__ = "ai_agents"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text)
    
    # Agent configuration
    system_prompt = Column(Text)
    tools = Column(JSON)  # List of available tools
    model = Column(String, default="llama-3.1-70b-versatile")
    
    # Behavior
    max_iterations = Column(Integer, default=10)
    temperature = Column(Float, default=0.7)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SemanticCache(Base):
    __tablename__ = "semantic_cache"
    id = Column(Integer, primary_key=True, index=True)
    query_hash = Column(String, unique=True, index=True)
    query_text = Column(Text)
    response = Column(Text)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"))
    
    # Cache metadata
    hit_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
