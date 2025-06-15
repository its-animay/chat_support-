import os
from pydantic_settings import BaseSettings
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv


load_dotenv()

class MilvusConfig(BaseModel):
    """Configuration for Milvus vector database"""
    host: str = Field(default=os.getenv("MILVUS_HOST", "localhost"))
    port: str = Field(default=os.getenv("MILVUS_PORT", "19530"))
    user: Optional[str] = Field(default=os.getenv("MILVUS_USER", None))
    password: Optional[str] = Field(default=os.getenv("MILVUS_PASSWORD", None))
    secure: bool = Field(default=os.getenv("MILVUS_SECURE", "False").lower() == "true")
    collection_prefix: str = Field(default="teacher_")
    embedding_dim: int = Field(default=3072)  # Gemini embedding dimension

class GeminiConfig(BaseModel):
    """Configuration for Google Gemini API"""
    api_key: str = Field(default=os.getenv("GOOGLE_API_KEY", ""))
    embedding_model: str = Field(default=os.getenv("GEMINI_EMBEDDING_MODEL", "embedding-gecko-001"))
    generation_model: str = Field(default=os.getenv("GEMINI_GENERATION_MODEL", "gemini-1.5-pro"))
    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.95)
    top_k: int = Field(default=40)

class RagConfig(BaseModel):
    """Configuration for RAG system"""
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    default_top_k: int = Field(default=5)
    similarity_threshold: float = Field(default=0.75)
    max_context_length: int = Field(default=4000)  # Characters to include in context
    include_metadata: bool = Field(default=True)
    rerank_results: bool = Field(default=True)
    multi_hop_max_iterations: int = Field(default=3)


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    openai_api_key: Optional[str] = None
    jwt_secret_key: str = "your-secret-key"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    redis_fallback_enabled: bool = False
    redis_fallback_dir: str = "./fallback_storage"
    redis_max_concurrent_ops: int = 100
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    
    model_config = ConfigDict(extra='allow')

config = Settings()