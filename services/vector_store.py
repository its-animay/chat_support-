from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType
)
from typing import List, Dict, Any, Optional, Tuple, Union
import time
import json
from datetime import datetime
import re
from core.logger import logger
from core.config import config
from services.embedder import GeminiEmbedder

class MilvusVectorStore:
    """Handles vector storage and retrieval using Milvus"""
    
    def __init__(self):
        self.host = config.milvus.host
        self.port = config.milvus.port
        self.user = config.milvus.user
        self.password = config.milvus.password
        self.secure = config.milvus.secure
        self.collection_prefix = config.milvus.collection_prefix
        self.embedding_dim = config.milvus.embedding_dim
        self.embedder = GeminiEmbedder()
        self._connect()
    
    def _connect(self):
        """Establish connection to Milvus"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                secure=self.secure
            )
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}", exc_info=True)
            raise
    
    def _sanitize_id(self, id_str: str) -> str:
        """Sanitize ID to make it Milvus-compatible (only letters, numbers, underscores)"""
        # Replace hyphens with underscores and remove any other invalid characters
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', id_str)
        
        # Ensure it doesn't start with a number (Milvus requirement)
        if sanitized and sanitized[0].isdigit():
            sanitized = 't_' + sanitized
            
        return sanitized
    
    def _get_collection_name(self, teacher_id: str) -> str:
        """Generate collection name from teacher ID"""
        sanitized_id = self._sanitize_id(teacher_id)
        return f"{self.collection_prefix}{sanitized_id}"
    
    def _get_original_id_from_collection(self, collection_name: str) -> str:
        """Extract original ID from collection name (for reference only)"""
        if collection_name.startswith(self.collection_prefix):
            return collection_name[len(self.collection_prefix):]
        return collection_name
    
    def collection_exists(self, teacher_id: str) -> bool:
        """Check if a collection exists for the teacher"""
        try:
            collection_name = self._get_collection_name(teacher_id)
            return utility.has_collection(collection_name)
        except Exception as e:
            logger.error(f"Failed to check if collection exists: {e}", exc_info=True)
            return False
    
    def create_collection(self, teacher_id: str) -> bool:
        """Create a new collection for a teacher"""
        try:
            collection_name = self._get_collection_name(teacher_id)
            
            # Define the collection schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            
            schema = CollectionSchema(fields=fields, description=f"Knowledge base for teacher {teacher_id}")
            
            # Create the collection
            collection = Collection(name=collection_name, schema=schema)
            
            # Create an IVF_FLAT index for the embeddings (efficient for medium-sized collections)
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 3072}
            }
            collection.create_index("embedding", index_params)
            
            logger.info(f"Created collection {collection_name} for teacher {teacher_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection for teacher {teacher_id}: {e}", exc_info=True)
            return False
    
    def delete_collection(self, teacher_id: str) -> bool:
        """Delete a teacher's collection"""
        try:
            collection_name = self._get_collection_name(teacher_id)
            
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                logger.info(f"Deleted collection {collection_name}")
                return True
            else:
                logger.warning(f"Collection {collection_name} does not exist")
                return False
        except Exception as e:
            logger.error(f"Failed to delete collection for teacher {teacher_id}: {e}", exc_info=True)
            return False
    
    async def insert_documents(
        self, 
        teacher_id: str, 
        documents: List[Dict[str, Any]]
    ) -> Tuple[bool, int]:
        """Insert documents into a teacher's collection"""
        try:
            collection_name = self._get_collection_name(teacher_id)
            
            # Create collection if it doesn't exist
            if not self.collection_exists(teacher_id):
                if not self.create_collection(teacher_id):
                    return False, 0
            
            # Get the collection
            collection = Collection(collection_name)
            collection.load()
            
            # Process documents into insertable format
            doc_ids = []
            doc_texts = []
            doc_embeddings = []
            doc_metadata = []
            
            for doc in documents:
                # Get text and metadata
                doc_id = doc.get('id', str(int(time.time() * 1000)))
                text = doc.get('text', '')
                metadata = doc.get('metadata', {})
                
                # Store original teacher ID in metadata for reference
                if 'teacher_id' not in metadata:
                    metadata['teacher_id'] = teacher_id
                
                # Generate embedding
                embedding = await self.embedder.get_embedding(text)
                
                # Add to insertion lists
                doc_ids.append(doc_id)
                doc_texts.append(text)
                doc_embeddings.append(embedding)
                doc_metadata.append(json.dumps(metadata))
            
            # Insert the documents
            insert_data = [
                doc_ids,
                doc_texts,
                doc_embeddings,
                doc_metadata
            ]
            
            collection.insert(insert_data)
            collection.flush()
            
            logger.info(f"Inserted {len(doc_ids)} documents into collection {collection_name}")
            return True, len(doc_ids)
        except Exception as e:
            logger.error(f"Failed to insert documents for teacher {teacher_id}: {e}", exc_info=True)
            return False, 0
    
    async def search(
        self, 
        teacher_id: str, 
        query: str, 
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """Search for documents in a teacher's collection"""
        try:
            if top_k is None:
                top_k = config.rag.default_top_k
                
            if similarity_threshold is None:
                similarity_threshold = config.rag.similarity_threshold
                
            collection_name = self._get_collection_name(teacher_id)
            
            # Check if collection exists
            if not self.collection_exists(teacher_id):
                logger.warning(f"Collection for teacher {teacher_id} does not exist")
                return []
            
            # Get the collection
            collection = Collection(collection_name)
            collection.load()
            
            # Generate query embedding
            query_embedding = await self.embedder.get_query_embedding(query)
            
            # Prepare search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # Prepare output fields
            output_fields = ["text", "metadata"]
            
            # Execute the search
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=output_fields
            )
            
            # Process results
            processed_results = []
            for hits in results:
                for hit in hits:
                    score = hit.score
                    
                    # Apply similarity threshold
                    if score < similarity_threshold:
                        continue
                    
                    # Get result data
                    result_id = hit.id
                    result_text = hit.entity.get('text')
                    result_metadata_str = hit.entity.get('metadata')
                    result_metadata = json.loads(result_metadata_str) if result_metadata_str else {}
                    
                    # Apply filters if provided
                    if filters and not self._apply_filters(result_metadata, filters):
                        continue
                    
                    processed_results.append({
                        'id': result_id,
                        'text': result_text,
                        'metadata': result_metadata,
                        'score': score
                    })
            
            logger.info(f"Found {len(processed_results)} results for query in collection {collection_name}")
            return processed_results
        except Exception as e:
            logger.error(f"Failed to search for teacher {teacher_id}: {e}", exc_info=True)
            return []
    
    def _apply_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply metadata filters to search results"""
        for key, value in filters.items():
            # Handle nested fields with dot notation
            if '.' in key:
                parts = key.split('.')
                current = metadata
                for part in parts[:-1]:
                    if part not in current:
                        return False
                    current = current[part]
                last_part = parts[-1]
                if last_part not in current or current[last_part] != value:
                    return False
            
            # Handle special operators
            elif key.endswith('__gt'):
                field = key[:-4]
                if field not in metadata or metadata[field] <= value:
                    return False
            elif key.endswith('__lt'):
                field = key[:-4]
                if field not in metadata or metadata[field] >= value:
                    return False
            elif key.endswith('__in'):
                field = key[:-4]
                if field not in metadata or metadata[field] not in value:
                    return False
            elif key.endswith('__contains'):
                field = key[:-10]
                if field not in metadata or value not in metadata[field]:
                    return False
            
            # Handle array fields
            elif isinstance(value, list):
                if key not in metadata:
                    return False
                # Check if there's any overlap between the filter values and metadata values
                metadata_value = metadata[key]
                if isinstance(metadata_value, list):
                    if not any(v in metadata_value for v in value):
                        return False
                else:
                    if metadata_value not in value:
                        return False
            
            # Handle exact match
            else:
                if key not in metadata or metadata[key] != value:
                    return False
        
        return True
    
    def delete_document(self, teacher_id: str, doc_id: str) -> bool:
        """Delete a document from a teacher's collection"""
        try:
            collection_name = self._get_collection_name(teacher_id)
            
            if not self.collection_exists(teacher_id):
                logger.warning(f"Collection for teacher {teacher_id} does not exist")
                return False
            
            collection = Collection(collection_name)
            collection.load()
            
            expr = f'id == "{doc_id}"'
            collection.delete(expr)
            
            logger.info(f"Deleted document {doc_id} from collection {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}", exc_info=True)
            return False
    
    def get_document_count(self, teacher_id: str) -> int:
        """Get the number of documents in a teacher's collection"""
        try:
            if not self.collection_exists(teacher_id):
                return 0
                
            collection_name = self._get_collection_name(teacher_id)
            collection = Collection(collection_name)
            return collection.num_entities
        except Exception as e:
            logger.error(f"Failed to get document count: {e}", exc_info=True)
            return 0