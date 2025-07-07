from typing import List, Dict, Any, Optional, Tuple, Union
import asyncio
import uuid
import numpy as np
from pymilvus import (
    connections, 
    Collection,
    utility,
    FieldSchema, 
    CollectionSchema,
    DataType
)
from core.config import Settings
from core.logger import logger
from services.embedding import get_embedding_service

settings = Settings()

class MilvusClientService:
    """Service for interacting with Milvus vector database"""
    
    def __init__(self):
        """Initialize the Milvus client"""
        self._connection_lock = asyncio.Lock()
        self._collection_lock = asyncio.Lock()
        self._connected = False
        self._embedding_service = get_embedding_service()
        
        # Single main collection name
        self._main_collection_name = "teacher"
        
        # Cache for collection objects to avoid repeated initialization
        self._collection_cache = {}
        
    async def connect(self) -> bool:
        """Connect to Milvus server with async support"""
        if self._connected:
            return True
            
        async with self._connection_lock:
            if self._connected:  # Double-check after acquiring lock
                return True
                
            try:
                # Run connection in a thread to avoid blocking
                await asyncio.to_thread(
                    connections.connect,
                    alias="default",
                    host=settings.milvus.host,
                    port=settings.milvus.port,
                    user=settings.milvus.user,
                    password=settings.milvus.password,
                    secure=settings.milvus.secure
                )
                
                self._connected = True
                logger.info(f"Connected to Milvus at {settings.milvus.host}:{settings.milvus.port}")
                return True
            except Exception as e:
                logger.error(f"Failed to connect to Milvus: {e}", exc_info=True)
                return False
    
    async def create_collection(self) -> bool:
        """Create the main teacher collection if it doesn't exist"""
        if not await self.connect():
            return False
        
        async with self._collection_lock:
            try:
                # Check if collection exists
                has_collection = await asyncio.to_thread(
                    utility.has_collection,
                    self._main_collection_name
                )
                
                if has_collection:
                    logger.info(f"Collection {self._main_collection_name} already exists")
                    return True
                
                # Define collection schema with teacher_id field for partitioning
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
                    FieldSchema(name="teacher_id", dtype=DataType.VARCHAR, max_length=100),  # For partitioning
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="metadata", dtype=DataType.JSON),
                    FieldSchema(
                        name="embedding", 
                        dtype=DataType.FLOAT_VECTOR, 
                        dim=self._embedding_service.get_dimension()
                    )
                ]
                
                schema = CollectionSchema(
                    fields=fields, 
                    description=f"Unified teacher document collection"
                )
                
                # Create collection
                collection = await asyncio.to_thread(
                    Collection,
                    name=self._main_collection_name,
                    schema=schema,
                    using="default",
                    shards_num=2  # Use multiple shards for better performance
                )
                
                # Create HNSW index on the embedding field
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 8, "efConstruction": 64}
                }
                
                await asyncio.to_thread(
                    collection.create_index,
                    field_name="embedding",
                    index_params=index_params
                )
                
                # Create index on teacher_id for faster filtering
                await asyncio.to_thread(
                    collection.create_index,
                    field_name="teacher_id",
                    index_params={"index_type": "INVERTED"},
                )
                
                # Load collection for searching
                await asyncio.to_thread(collection.load)
                
                # Cache the collection
                self._collection_cache[self._main_collection_name] = collection
                
                logger.info(f"Created collection {self._main_collection_name} with HNSW index")
                return True
                
            except Exception as e:
                logger.error(f"Failed to create collection {self._main_collection_name}: {e}", exc_info=True)
                return False
    
    async def _get_collection(self) -> Optional[Collection]:
        """Get the main collection object with caching"""
        if not await self.connect():
            return None
            
        # Check if we already have the collection in cache
        if self._main_collection_name in self._collection_cache:
            return self._collection_cache[self._main_collection_name]
            
        try:
            # Check if collection exists
            has_collection = await asyncio.to_thread(
                utility.has_collection,
                self._main_collection_name
            )
            
            if not has_collection:
                # Create collection if it doesn't exist
                if not await self.create_collection():
                    return None
                # Return from cache which should be populated by create_collection
                return self._collection_cache.get(self._main_collection_name)
            
            # Get collection
            collection = await asyncio.to_thread(
                Collection,
                name=self._main_collection_name
            )
            
            # Ensure collection is loaded
            try:
                await asyncio.to_thread(collection.load)
            except Exception as e:
                logger.warning(f"Failed to load collection, attempting to create missing indexes: {e}")
                
                # Create HNSW index on the embedding field if it's missing
                try:
                    index_params = {
                        "metric_type": "COSINE",
                        "index_type": "HNSW",
                        "params": {"M": 8, "efConstruction": 64}
                    }
                    
                    await asyncio.to_thread(
                        collection.create_index,
                        field_name="embedding",
                        index_params=index_params
                    )
                    
                    # Create index on teacher_id for faster filtering
                    await asyncio.to_thread(
                        collection.create_index,
                        field_name="teacher_id",
                        index_params={"index_type": "INVERTED"},
                    )
                    
                    # Try loading again
                    await asyncio.to_thread(collection.load)
                except Exception as inner_e:
                    logger.error(f"Failed to create indexes and load collection: {inner_e}", exc_info=True)
                    return None
            
            # Cache the collection
            self._collection_cache[self._main_collection_name] = collection
            
            return collection
        except Exception as e:
            logger.error(f"Failed to get collection {self._main_collection_name}: {e}", exc_info=True)
            return None
    
    async def insert_documents(
        self, 
        documents: List[Dict[str, Any]], 
        teacher_id: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        Insert documents into Milvus.
        
        Args:
            documents: List of document dicts with 'content' and optional 'metadata'
            teacher_id: Teacher ID to associate with the documents (required)
            
        Returns:
            Tuple of (success_bool, list_of_inserted_ids)
        """
        if not await self.connect():
            return False, []
            
        if not documents:
            return True, []
        
        # Ensure the collection exists
        if not await self.create_collection():
            return False, []
        
        try:
            # Get the collection
            collection = await self._get_collection()
            if not collection:
                return False, []
            
            # Extract content for embedding
            contents = [doc.get('content', '') for doc in documents]
            
            # Generate embeddings for all documents
            embeddings = await self._embedding_service.batch_embed_text(contents)
            
            if len(embeddings) != len(documents):
                logger.error(f"Embedding count mismatch: got {len(embeddings)}, expected {len(documents)}")
                return False, []
            
            # Prepare data for insertion
            doc_ids = []
            teacher_ids = []
            content_list = []
            metadata_list = []
            embedding_list = []
            
            for i, doc in enumerate(documents):
                # Ensure ID is within the 36 character limit required by Milvus
                original_id = doc.get('id', str(uuid.uuid4()))
                if len(original_id) > 36:
                    # Truncate or hash long IDs to fit within limit
                    doc_id = original_id[:36]
                    # Store original ID in metadata for reference
                    if 'metadata' not in doc:
                        doc['metadata'] = {}
                    doc['metadata']['original_id'] = original_id
                else:
                    doc_id = original_id
                
                doc_ids.append(doc_id)
                
                # Use the provided teacher_id for all documents
                teacher_ids.append(teacher_id or "unknown")
                
                # Ensure content is within Milvus VARCHAR length limit (65535 chars)
                content = doc.get('content', '')
                if len(content) > 65535:
                    # Truncate content and add note to metadata
                    if 'metadata' not in doc:
                        doc['metadata'] = {}
                    
                    # Store original content length in metadata
                    doc['metadata']['original_content_length'] = len(content)
                    doc['metadata']['content_truncated'] = True
                    
                    # Truncate to slightly under the limit to be safe
                    content_list.append(content[:65000] + "\n\n[Content truncated due to size limits]")
                    logger.warning(f"Content truncated for document {doc_id}: {len(content)} chars -> 65000 chars")
                else:
                    content_list.append(content)
                
                # Ensure metadata includes teacher_id
                metadata = doc.get('metadata', {}).copy()
                if teacher_id and 'teacher_id' not in metadata:
                    metadata['teacher_id'] = teacher_id
                
                metadata_list.append(metadata)
                embedding_list.append(embeddings[i].tolist())
            
            # Insert data
            await asyncio.to_thread(
                collection.insert,
                [
                    doc_ids,         # id field
                    teacher_ids,     # teacher_id field for partitioning
                    content_list,    # content field
                    metadata_list,   # metadata field
                    embedding_list   # embedding field
                ]
            )
            
            # Flush to ensure data is persisted
            await asyncio.to_thread(collection.flush)
            
            logger.info(f"Inserted {len(doc_ids)} documents for teacher {teacher_id}")
            return True, doc_ids
            
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}", exc_info=True)
            return False, []
    
    async def search(
        self, 
        query: str, 
        top_k: int = 5, 
        teacher_id: Optional[str] = None,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in Milvus
        
        Args:
            query: User's query text
            top_k: Number of results to return
            teacher_id: Optional teacher ID to filter results
            filter_expr: Optional additional filter expression
            
        Returns:
            List of matching documents with scores
        """
        if not await self.connect():
            return []
        
        # Ensure the collection exists
        collection = await self._get_collection()
        if not collection:
            if not await self.create_collection():
                return []
            collection = await self._get_collection()
            if not collection:
                return []
        
        try:
            # Generate embedding for query
            query_embedding = await self._embedding_service.embed_text(query)
            
            # Prepare search params
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}  # Higher ef means more accurate but slower search
            }
            
            # Combine teacher_id filter with any additional filters
            expr = None
            if teacher_id:
                expr = f"teacher_id == '{teacher_id}'"
                if filter_expr:
                    expr = f"({expr}) && ({filter_expr})"
            elif filter_expr:
                expr = filter_expr
            
            # Execute search
            results = await asyncio.to_thread(
                collection.search,
                data=[query_embedding.tolist()],  # Convert to list for Milvus
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["content", "metadata", "teacher_id"]
            )
            
            # Format results
            formatted_results = []
            for hits in results:  # results is a list of list, outer list is for each query
                for hit in hits:
                    try:
                        # Extract data directly from hit.entity field
                        content = ""
                        metadata = {}
                        teacher_id_value = "unknown"
                        
                        # Direct attribute access
                        if hasattr(hit, 'entity'):
                            # For pymilvus >= 2.0, entity is an object with fields as attributes
                            if hasattr(hit.entity, 'content'):
                                content = hit.entity.content
                            if hasattr(hit.entity, 'metadata'):
                                metadata = hit.entity.metadata
                            if hasattr(hit.entity, 'teacher_id'):
                                teacher_id_value = hit.entity.teacher_id
                            
                            # Some versions store data in fields
                            if hasattr(hit.entity, 'fields'):
                                fields = hit.entity.fields
                                if isinstance(fields, dict):
                                    content = fields.get('content', content)
                                    metadata = fields.get('metadata', metadata)
                                    teacher_id_value = fields.get('teacher_id', teacher_id_value)
                                
                        # Add the formatted result
                        formatted_results.append({
                            "id": hit.id,
                            "content": content,
                            "metadata": metadata,
                            "teacher_id": teacher_id_value,
                            "score": hit.score,  # Similarity score
                        })
                    except Exception as e:
                        logger.error(f"Error processing search result: {e}", exc_info=True)
                        # Still add a partial result to avoid losing data
                        formatted_results.append({
                            "id": getattr(hit, 'id', 'unknown'),
                            "content": "Error retrieving content",
                            "metadata": {},
                            "score": getattr(hit, 'score', 0),
                            "error": str(e)
                        })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}", exc_info=True)
            return []
    
    async def delete_documents(
        self, 
        doc_ids: List[str], 
        teacher_id: Optional[str] = None
    ) -> bool:
        """
        Delete documents from Milvus by their IDs
        
        Args:
            doc_ids: List of document IDs to delete
            teacher_id: Optional teacher ID to restrict deletion to specific teacher's documents
            
        Returns:
            Boolean indicating success
        """
        if not await self.connect():
            return False
        
        # Get collection
        collection = await self._get_collection()
        if not collection:
            logger.warning(f"Collection {self._main_collection_name} does not exist, nothing to delete")
            return False
        
        try:
            # Build expression to delete only the specified documents
            expr = f"id in {doc_ids}"
            
            # Add teacher_id constraint if provided
            if teacher_id:
                expr = f"({expr}) && (teacher_id == '{teacher_id}')"
            
            # Delete documents
            await asyncio.to_thread(
                collection.delete,
                expr
            )
            
            logger.info(f"Deleted documents with IDs: {doc_ids}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}", exc_info=True)
            return False
    
    async def delete_teacher_documents(self, teacher_id: str) -> bool:
        """
        Delete all documents for a specific teacher
        
        Args:
            teacher_id: Teacher ID whose documents should be deleted
            
        Returns:
            Boolean indicating success
        """
        if not await self.connect() or not teacher_id:
            return False
        
        # Get collection
        collection = await self._get_collection()
        if not collection:
            logger.warning(f"Collection {self._main_collection_name} does not exist, nothing to delete")
            return False
        
        try:
            # Build expression to delete all documents for this teacher
            expr = f"teacher_id == '{teacher_id}'"
            
            # Delete documents
            await asyncio.to_thread(
                collection.delete,
                expr
            )
            
            logger.info(f"Deleted all documents for teacher: {teacher_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete teacher documents: {e}", exc_info=True)
            return False
    
    async def get_teacher_document_count(self, teacher_id: str) -> int:
        """
        Get the number of documents for a specific teacher
        
        Args:
            teacher_id: Teacher ID to count documents for
            
        Returns:
            Number of documents for the teacher
        """
        if not await self.connect() or not teacher_id:
            return 0
        
        # Get collection
        collection = await self._get_collection()
        if not collection:
            return 0
        
        try:
            # FIXED: Use query approach instead of problematic count syntax
            expr = f"teacher_id == '{teacher_id}'"
            
            # Query for documents matching the teacher_id
            results = await asyncio.to_thread(
                collection.query,
                expr=expr,
                output_fields=["id"],  # Only need IDs for counting
                limit=16384  # Set reasonable limit for large collections
            )
            
            # Return the count of results
            return len(results)
            
        except Exception as e:
            logger.error(f"Failed to get document count for teacher {teacher_id}: {e}", exc_info=True)
            return 0
        
    async def list_teachers(self) -> List[str]:
        """
        List all teacher IDs that have documents in the collection
        
        Returns:
            List of teacher IDs
        """
        if not await self.connect():
            return []
        
        # Get collection
        collection = await self._get_collection()
        if not collection:
            return []
        
        try:
            # Query distinct teacher_ids
            results = await asyncio.to_thread(
                collection.query,
                expr="teacher_id != ''",
                output_fields=["teacher_id"],
                limit=10000
            )
            
            # Extract unique teacher IDs
            teacher_ids = set()
            for result in results:
                teacher_id = result.get("teacher_id")
                if teacher_id:
                    teacher_ids.add(teacher_id)
            
            return list(teacher_ids)
            
        except Exception as e:
            logger.error(f"Failed to list teachers: {e}", exc_info=True)
            return []
    
    async def get_teacher_stats(self) -> Dict[str, int]:
        """
        Get document counts for all teachers
        
        Returns:
            Dict mapping teacher IDs to their document counts
        """
        if not await self.connect():
            return {}
        
        # Get collection
        collection = await self._get_collection()
        if not collection:
            return {}
        
        try:
            # Get all documents with teacher_id field
            results = await asyncio.to_thread(
                collection.query,
                expr="teacher_id != ''",  # Get all documents with teacher_id
                output_fields=["teacher_id"],
                limit=16384  # Reasonable limit
            )
            
            # Count documents per teacher
            teacher_counts = {}
            for result in results:
                teacher_id = result.get("teacher_id")
                if teacher_id:
                    teacher_counts[teacher_id] = teacher_counts.get(teacher_id, 0) + 1
            
            return teacher_counts
            
        except Exception as e:
            logger.error(f"Failed to get teacher stats: {e}", exc_info=True)
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of Milvus connection and return status
        
        Returns:
            Dict with health status information
        """
        health = {
            "connected": False,
            "collection_exists": False,
            "document_count": 0
        }
        
        try:
            # Check connection
            health["connected"] = await self.connect()
            
            if health["connected"]:
                # Check if collection exists
                collection = await self._get_collection()
                health["collection_exists"] = collection is not None
                
                if health["collection_exists"]:
                    # FIXED: Remove asyncio.to_thread wrapper for property access
                    try:
                        count = collection.num_entities  # This is a property, not a method
                        health["document_count"] = count
                    except Exception as e:
                        logger.error(f"Error getting document count: {e}")
                        health["document_count"] = 0
                        health["error"] = str(e)
                    
                    # Get teacher stats
                    try:
                        health["teacher_stats"] = await self.get_teacher_stats()
                    except Exception as e:
                        logger.error(f"Error getting teacher stats: {e}")
                        health["teacher_stats"] = {}
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            health["error"] = str(e)
            return health

# Global singleton instance
milvus_client = MilvusClientService()