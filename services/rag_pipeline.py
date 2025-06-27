from typing import List, Dict, Any, Optional, Union
import asyncio
import time
from core.config import Settings
from core.logger import logger
from services.milvus_client import milvus_client
from services.reranker import get_reranker_service
from services.llm_client import llm_service
from services.redis_client import redis_client

settings = Settings()

class RAGPipeline:
    """Complete Retrieval-Augmented Generation pipeline"""
    
    def __init__(self):
        """Initialize the RAG pipeline"""
        self.milvus_client = milvus_client
        self.reranker = get_reranker_service()
        self.llm_service = llm_service
    
    async def process_query(
        self,
        query: str,
        top_k: int = 10,
        top_n: int = 3,
        temperature: float = 0.7,
        teacher_id: Optional[str] = None,
        filter_expr: Optional[str] = None,
        use_cache: bool = True,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline
        
        Args:
            query: The user's query
            top_k: Number of initial documents to retrieve
            top_n: Number of documents to keep after reranking
            temperature: Temperature for LLM generation
            teacher_id: Teacher ID to search in specific knowledge base
            filter_expr: Optional Milvus filter expression
            use_cache: Whether to use Redis for caching results
            system_prompt: Optional custom system prompt for the LLM
            
        Returns:
            Dict with generated response and metadata
        """
        start_time = time.time()
        cache_key = None
        
        # Check cache if enabled
        if use_cache:
            # Create deterministic cache key from query params
            cache_key = f"rag:query:{query}:{top_k}:{top_n}:{temperature}:{teacher_id or 'default'}:{filter_expr or 'none'}"
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                cached_result["cached"] = True
                cached_result["processing_time"] = 0
                return cached_result
        
        try:
            # Step 1: Initial retrieval from Milvus
            retrieved_docs = await self.milvus_client.search(
                query=query,
                top_k=top_k,
                teacher_id=teacher_id,
                filter_expr=filter_expr
            )
            
            logger.info(f"Initial retrieval returned {len(retrieved_docs)} documents")
            
            # Step 2: Rerank documents if we have results
            if retrieved_docs:
                reranked_docs = await self.reranker.rerank(
                    query=query,
                    documents=retrieved_docs,
                    top_n=top_n
                )
                logger.info(f"Reranking complete, kept top {len(reranked_docs)} documents")
            else:
                reranked_docs = []
                logger.info("No documents retrieved, skipping reranking")
            
            # Step 3: Generate response with LLM
            response = await self.llm_service.generate_response(
                query=query,
                retrieved_documents=reranked_docs,
                system_prompt=system_prompt,
                temperature=temperature
            )
            
            # Add processing metadata
            response["processing_time"] = time.time() - start_time
            response["retrieval_count"] = len(retrieved_docs)
            response["reranked_count"] = len(reranked_docs)
            response["cached"] = False
            response["teacher_id"] = teacher_id
            
            # Cache result if enabled
            if use_cache and cache_key:
                await self._save_to_cache(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}", exc_info=True)
            
            # Return fallback response with error details
            return {
                "response": "I'm sorry, I encountered an error while processing your question. Please try again later.",
                "error": str(e),
                "rag_enhanced": False,
                "sources_used": [],
                "processing_time": time.time() - start_time,
                "cached": False,
                "teacher_id": teacher_id
            }
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        teacher_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add documents to the RAG knowledge base
        
        Args:
            documents: List of documents to add
            teacher_id: Optional teacher ID
            
        Returns:
            Dict with operation results
        """
        try:
            # Validate documents have required fields
            for doc in documents:
                if "content" not in doc:
                    return {
                        "success": False,
                        "error": "All documents must have 'content' field",
                        "documents_added": 0
                    }
            
            # Add teacher_id to metadata if provided
            if teacher_id:
                for doc in documents:
                    if 'metadata' not in doc:
                        doc['metadata'] = {}
                    if 'teacher_id' not in doc['metadata']:
                        doc['metadata']['teacher_id'] = teacher_id
            
            # Insert documents into Milvus
            success, doc_ids = await self.milvus_client.insert_documents(
                documents=documents,
                teacher_id=teacher_id
            )
            
            if success:
                return {
                    "success": True,
                    "documents_added": len(doc_ids),
                    "document_ids": doc_ids,
                    "teacher_id": teacher_id
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to insert documents",
                    "documents_added": 0,
                    "teacher_id": teacher_id
                }
                
        except Exception as e:
            logger.error(f"Error adding documents to RAG: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "documents_added": 0,
                "teacher_id": teacher_id
            }
        
    async def delete_documents(
        self,
        document_ids: List[str],
        teacher_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Delete documents from the RAG knowledge base
        
        Args:
            document_ids: List of document IDs to delete
            teacher_id: Teacher ID to determine which knowledge base to use
            
        Returns:
            Dict with operation results
        """
        try:
            success = await self.milvus_client.delete_documents(
                doc_ids=document_ids,
                teacher_id=teacher_id
            )
            
            if success:
                return {
                    "success": True,
                    "documents_deleted": len(document_ids),
                    "teacher_id": teacher_id
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to delete documents",
                    "documents_deleted": 0,
                    "teacher_id": teacher_id
                }
                
        except Exception as e:
            logger.error(f"Error deleting documents from RAG: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "documents_deleted": 0,
                "teacher_id": teacher_id
            }
    
    async def delete_teacher_documents(self, teacher_id: str) -> Dict[str, Any]:
        """
        Delete all documents for a specific teacher
        
        Args:
            teacher_id: Teacher ID whose documents should be deleted
            
        Returns:
            Dict with operation results
        """
        if not teacher_id:
            return {
                "success": False,
                "error": "Teacher ID is required",
                "documents_deleted": 0
            }
            
        try:
            success = await self.milvus_client.delete_teacher_documents(teacher_id)
            
            if success:
                return {
                    "success": True,
                    "message": f"All documents for teacher {teacher_id} deleted successfully",
                    "teacher_id": teacher_id
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to delete documents for teacher {teacher_id}",
                    "teacher_id": teacher_id
                }
                
        except Exception as e:
            logger.error(f"Error deleting teacher documents: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "teacher_id": teacher_id
            }
    
    async def list_teacher_knowledge_bases(self) -> Dict[str, Any]:
        """List all teacher knowledge bases with document counts"""
        try:
            teacher_stats = await self.milvus_client.get_teacher_stats()
            
            return {
                "success": True,
                "teacher_knowledge_bases": teacher_stats,
                "total_teachers": len(teacher_stats)
            }
            
        except Exception as e:
            logger.error(f"Error listing teacher knowledge bases: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "teacher_knowledge_bases": {},
                "total_teachers": 0
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all RAG components"""
        milvus_status = await self.milvus_client.health_check()
        
        return {
            "status": "healthy" if milvus_status.get("connected", False) else "unhealthy",
            "milvus": milvus_status,
            "components": {
                "retriever": milvus_status.get("connected", False),
                "reranker": self.reranker._model is not None,
                "llm": self.llm_service.openai_client is not None
            },
            "document_count": milvus_status.get("document_count", 0),
            "teacher_stats": milvus_status.get("teacher_stats", {})
        }
    
    async def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get result from Redis cache"""
        if not redis_client.connected:
            return None
            
        try:
            cached = await redis_client.json_get(key)
            if cached:
                logger.info(f"Cache hit for key: {key}")
                return cached
            return None
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
    
    async def _save_to_cache(self, key: str, data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Save result to Redis cache with TTL"""
        if not redis_client.connected:
            return False
            
        try:
            # Store data in Redis with expiration
            success = await redis_client.json_set(key, data)
            if success:
                # Set expiration (non-critical operation)
                asyncio.create_task(redis_client.set_expiration(key, ttl))
            return success
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
            return False


# Global singleton instance
rag_pipeline = RAGPipeline()