# embedder.py (Quota-Resilient Version)
import google.generativeai as genai
from typing import List, Optional, Dict, Any, Union
import numpy as np
import hashlib
import os
import json
import time
from functools import lru_cache
import random
from pathlib import Path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from core.config import config
from core.logger import logger

class GeminiEmbedder:
    """Handles document embedding using Google's Gemini models with quota resilience"""
    
    def __init__(self, cache_dir="cache/embeddings"):
        self.api_key = config.gemini.api_key
        self.model_name = config.gemini.embedding_model
        self.embedding_dim = config.milvus.embedding_dim
        self.cache_dir = cache_dir
        self._setup_api()
        self._setup_cache()
        
    def _setup_api(self):
        """Initialize the Gemini API client"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini API initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}", exc_info=True)
            raise
    
    def _setup_cache(self):
        """Set up the embedding cache directory"""
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.debug(f"Embedding cache directory set up at: {self.cache_dir}")
        except Exception as e:
            logger.error(f"Failed to set up cache directory: {e}", exc_info=True)
            # Fallback - continue without caching
            self.cache_dir = None
    
    def _get_cache_key(self, text: str, task_type: str = "retrieval_document") -> str:
        """Generate a deterministic cache key from text and task type"""
        # Create a hash of the text and task type
        text_bytes = text.encode('utf-8')
        task_bytes = task_type.encode('utf-8')
        combined = text_bytes + b":" + task_bytes
        return hashlib.md5(combined).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get the file path for a cache key"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _cache_embedding(self, cache_key: str, embedding: List[float]):
        """Cache an embedding to disk"""
        if not self.cache_dir:
            return
            
        try:
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'w') as f:
                json.dump({"embedding": embedding}, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    def _get_cached_embedding(self, cache_key: str) -> Optional[List[float]]:
        """Retrieve a cached embedding if it exists"""
        if not self.cache_dir:
            return None
            
        try:
            cache_path = self._get_cache_path(cache_key)
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    return data.get("embedding")
            return None
        except Exception as e:
            logger.warning(f"Failed to retrieve cached embedding: {e}")
            return None
    
    def _generate_deterministic_embedding(self, text: str) -> List[float]:
        """Generate a deterministic but random-looking embedding based on text hash"""
        # This is a fallback when API calls fail and no cache exists
        text_hash = hashlib.sha256(text.encode('utf-8')).digest()
        
        # Use the hash to seed a random number generator
        random.seed(text_hash)
        
        # Generate a random embedding vector
        embedding = [random.uniform(-1, 1) for _ in range(self.embedding_dim)]
        
        # Normalize the vector to unit length
        magnitude = sum(x**2 for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x/magnitude for x in embedding]
            
        logger.warning(f"Using deterministic fallback embedding for text: {text[:50]}...")
        return embedding
    
    @retry(
        retry=retry_if_exception_type((
            genai.types.generation_types.BlockedPromptException,
            genai.types.generation_types.StopCandidateException
        )),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _call_embedding_api(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """Call the Gemini embedding API with retries for specific errors"""
        try:
            embedding_result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type=task_type
            )
            
            # Extract the embedding values
            embedding = embedding_result['embedding']
            return embedding
        except Exception as e:
            logger.error(f"API call failed for embedding: {e}", exc_info=True)
            raise
    
    async def get_embedding(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """Generate embedding for a single text with caching and fallbacks"""
        try:
            if not text or text.isspace():
                logger.warning("Empty text provided for embedding")
                # Return zero vector for empty text
                return [0.0] * self.embedding_dim
            
            # Truncate text if it's too long (Gemini has a token limit)
            if len(text) > 10000:  # Approximate limit
                logger.warning(f"Text too long ({len(text)} chars), truncating for embedding")
                text = text[:10000]
            
            # Check cache first
            cache_key = self._get_cache_key(text, task_type)
            cached_embedding = self._get_cached_embedding(cache_key)
            
            if cached_embedding:
                logger.debug(f"Using cached embedding for text: {text[:50]}...")
                return cached_embedding
            
            try:
                # Try to get embedding from API
                embedding = await self._call_embedding_api(text, task_type)
                
                # Cache the successful result
                self._cache_embedding(cache_key, embedding)
                
                # Ensure correct dimensionality
                if len(embedding) != self.embedding_dim:
                    logger.warning(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")
                
                return embedding
            except Exception as e:
                # Check for quota exceeded error
                if "429" in str(e) or "quota" in str(e).lower() or "resource exhausted" in str(e).lower():
                    logger.warning(f"API quota exceeded, using deterministic fallback embedding")
                    # Generate deterministic fallback embedding
                    fallback_embedding = self._generate_deterministic_embedding(text)
                    
                    # Cache the fallback result too (optional)
                    self._cache_embedding(cache_key, fallback_embedding)
                    
                    return fallback_embedding
                else:
                    # For other errors, also use fallback
                    logger.error(f"Failed to generate embedding: {e}", exc_info=True)
                    return self._generate_deterministic_embedding(text)
                
        except Exception as e:
            logger.error(f"Unexpected error in get_embedding: {e}", exc_info=True)
            # Return deterministic fallback on any error
            return self._generate_deterministic_embedding(text)
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with rate limiting"""
        embeddings = []
        for i, text in enumerate(texts):
            # Add delay every few requests to avoid hitting rate limits
            if i > 0 and i % 5 == 0:
                await asyncio.sleep(1)  # Sleep for 1 second every 5 requests
                
            embedding = await self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    async def get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a search query"""
        try:
            return await self.get_embedding(query, task_type="retrieval_query")
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}", exc_info=True)
            # Return deterministic fallback on error
            return self._generate_deterministic_embedding(query)