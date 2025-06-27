from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import asyncio
from sentence_transformers import CrossEncoder
from core.config import Settings
from core.logger import logger
from functools import lru_cache

settings = Settings()

class RerankerService:
    """Service for reranking retrieved documents using a cross-encoder model"""
    
    def __init__(self, model_name: str = None):
        """Initialize the reranker service with specified model or default from settings"""
        self.model_name = model_name or settings.rag.reranker_model
        self._model = None
        self._init_lock = asyncio.Lock()
    
    async def _load_model(self):
        """Lazy-load the reranker model when first needed"""
        if self._model is None:
            async with self._init_lock:
                if self._model is None:  # Double-check after acquiring lock
                    logger.info(f"Loading reranker model: {self.model_name}")
                    # Run model loading in a thread pool to avoid blocking the event loop
                    self._model = await asyncio.to_thread(CrossEncoder, self.model_name)
                    logger.info(f"Reranker model loaded successfully")
    
    async def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to the query
        
        Args:
            query: The user query
            documents: List of document dicts from first-stage retrieval
            top_n: Number of top documents to return after reranking
            
        Returns:
            List of documents sorted by relevance with rerank scores added
        """
        if not documents:
            return []
            
        await self._load_model()
        
        try:
            # Create document-query pairs for reranking
            pairs = [(query, doc.get('content', '')) for doc in documents]
            
            # Get reranking scores
            scores = await asyncio.to_thread(self._model.predict, pairs)
            
            # Add scores to documents
            for i, score in enumerate(scores):
                documents[i]['rerank_score'] = float(score)
            
            # Sort by reranking score (descending)
            reranked_docs = sorted(documents, key=lambda x: x.get('rerank_score', 0), reverse=True)
            
            # Return top_n documents
            return reranked_docs[:top_n]
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}", exc_info=True)
            # Fall back to original ordering
            return documents[:top_n]


@lru_cache(maxsize=1)
def get_reranker_service(model_name: Optional[str] = None) -> RerankerService:
    """Get a cached instance of the reranker service"""
    return RerankerService(model_name)