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
    collection_prefix: str = Field(default=os.getenv("MILVUS_COLLECTION_PREFIX", "teacher_"))
    embedding_dim: int = Field(default=int(os.getenv("EMBEDDING_DIM", "384")))  # Default for all-MiniLM-L6-v2

class GeminiConfig(BaseModel):
    """Configuration for Google Gemini API"""
    api_key: str = Field(default=os.getenv("GOOGLE_API_KEY", ""))
    embedding_model: str = Field(default=os.getenv("GEMINI_EMBEDDING_MODEL", "embedding-gecko-001"))
    generation_model: str = Field(default=os.getenv("GEMINI_GENERATION_MODEL", "gemini-1.5-pro"))
    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.95)
    top_k: int = Field(default=40)


class FileProcessingConfig(BaseModel):
    """Configuration for file processing"""
    max_file_size_mb: int = Field(default=int(os.getenv("MAX_FILE_SIZE_MB", "20")))
    allowed_extensions_str: str = Field(default=os.getenv("ALLOWED_EXTENSIONS", ".txt,.pdf,.docx,.md,.html,.csv,.json,.xlsx"))
    chunk_size: int = Field(default=int(os.getenv("FILE_CHUNK_SIZE", "1000")))
    chunk_overlap: int = Field(default=int(os.getenv("FILE_CHUNK_OVERLAP", "100")))
    
    @property
    def allowed_extensions(self) -> List[str]:
        """Get list of allowed extensions"""
        extensions = self.allowed_extensions_str.split(",")
        return [ext.strip() for ext in extensions]
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes"""
        return self.max_file_size_mb * 1024 * 1024

class RagConfig(BaseModel):
    """Configuration for RAG system"""
    # Embedding settings
    embedding_model: str = Field(default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    
    # Reranker settings
    reranker_model: str = Field(default=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    
    # Retrieval settings
    default_top_k: int = Field(default=int(os.getenv("RAG_TOP_K", "10")))
    default_top_n: int = Field(default=int(os.getenv("RAG_TOP_N", "3")))
    similarity_threshold: float = Field(default=float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.75")))
    
    # Document processing
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    max_context_length: int = Field(default=4000)  # Characters to include in context
    
    # Caching
    enable_cache: bool = Field(default=os.getenv("ENABLE_CACHE", "True").lower() == "true")
    cache_ttl: int = Field(default=int(os.getenv("CACHE_TTL", "3600")))  # 1 hour default
    
    # Additional features
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
    file_processing: FileProcessingConfig = Field(default_factory=FileProcessingConfig)

    
    model_config = ConfigDict(extra='allow')

config = Settings()