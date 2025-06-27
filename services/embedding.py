from typing import List, Optional, Dict, Any, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from core.config import Settings
from core.logger import logger
import asyncio
from functools import lru_cache

settings = Settings()

class EmbeddingService:
    """Service for generating text embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = None):
        """Initialize the embedding service with specified model or default from settings"""
        self.model_name = model_name or settings.rag.embedding_model
        self._model = None
        self._init_lock = asyncio.Lock()
        
    async def _load_model(self):
        """Lazy-load the embedding model when first needed"""
        if self._model is None:
            async with self._init_lock:
                if self._model is None:  # Double-check after acquiring lock
                    logger.info(f"Loading embedding model: {self.model_name}")
                    # Run model loading in a thread pool to avoid blocking the event loop
                    self._model = await asyncio.to_thread(SentenceTransformer, self.model_name)
                    logger.info(f"Embedding model loaded successfully")
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text input"""
        await self._load_model()
        
        try:
            # Run embedding in a thread pool to avoid blocking the event loop
            embedding = await asyncio.to_thread(self._model.encode, text, 
                                              convert_to_numpy=True, 
                                              normalize_embeddings=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}", exc_info=True)
            # Return zero vector as fallback with correct dimensionality
            return np.zeros(self._model.get_sentence_embedding_dimension())
    
    async def batch_embed_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a batch of texts with specified batch size"""
        await self._load_model()
        
        if not texts:
            return np.array([])
        
        try:
            # Process in batches to manage memory usage
            all_embeddings = []
            
            # Process in chunks to avoid memory issues
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                # Run embedding in a thread pool to avoid blocking the event loop
                batch_embeddings = await asyncio.to_thread(
                    self._model.encode,
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                all_embeddings.append(batch_embeddings)
            
            return np.vstack(all_embeddings)
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}", exc_info=True)
            # Return empty array as fallback
            return np.array([])
    
    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        if self._model is None:
            # Load model synchronously if dimension is needed before async init
            self._model = SentenceTransformer(self.model_name)
        return self._model.get_sentence_embedding_dimension()


@lru_cache(maxsize=2)
def get_embedding_service(model_name: Optional[str] = None) -> EmbeddingService:
    """Get a cached instance of the embedding service"""
    return EmbeddingService(model_name)