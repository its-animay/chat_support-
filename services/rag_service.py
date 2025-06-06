from typing import List, Dict, Any, Optional, Tuple
from models.teacher import EnhancedTeacher
from models.chat import Message, MessageRole
from core.logger import logger
from services.agent import RagAgent
from services.document_processor import DocumentProcessor
from services.vector_store import MilvusVectorStore
from services.embedder import GeminiEmbedder
import asyncio

class RagService:
    """Service for managing RAG operations in the teaching platform"""
    
    def __init__(self):
        self.agent = RagAgent()
        self.vector_store = MilvusVectorStore()
        self.embedder = GeminiEmbedder()
        self.doc_processor = DocumentProcessor()
    
    async def add_documents_for_teacher(
        self, 
        teacher_id: str, 
        documents: List[Dict[str, Any]]
    ) -> Tuple[bool, int]:
        """Process and add documents to a teacher's knowledge base"""
        try:
            # Process each document
            processed_chunks = []
            for doc in documents:
                chunks = self.doc_processor.process_document(doc)
                processed_chunks.extend(chunks)
            
            # Insert into vector store
            success, count = await self.vector_store.insert_documents(teacher_id, processed_chunks)
            
            return success, count
        except Exception as e:
            logger.error(f"Failed to add documents for teacher {teacher_id}: {e}", exc_info=True)
            return False, 0
    
    def delete_document(self, teacher_id: str, doc_id: str) -> bool:
        """Delete a document from a teacher's knowledge base"""
        return self.vector_store.delete_document(teacher_id, doc_id)
    
    def get_document_count(self, teacher_id: str) -> int:
        """Get the number of documents in a teacher's knowledge base"""
        return self.vector_store.get_document_count(teacher_id)
    
    def collection_exists(self, teacher_id: str) -> bool:
        """Check if a knowledge base exists for the teacher"""
        return self.vector_store.collection_exists(teacher_id)
    
    def create_collection(self, teacher_id: str) -> bool:
        """Create a new knowledge base for a teacher"""
        return self.vector_store.create_collection(teacher_id)
    
    def delete_collection(self, teacher_id: str) -> bool:
        """Delete a teacher's knowledge base"""
        return self.vector_store.delete_collection(teacher_id)
    
    async def enhance_teacher_response(
        self, 
        teacher: EnhancedTeacher,
        query: str,
        chat_history: List[Message]
    ) -> Dict[str, Any]:
        """
        Generate an enhanced response using RAG
        Returns both the response and metadata about the retrieval process
        """
        try:
            # Convert teacher to info dict for the agent
            teacher_info = {
                'id': teacher.id,
                'name': teacher.name,
                'domain': teacher.specialization.primary_domain,
                'sub_domains': teacher.specialization.specializations,
                'teaching_style': teacher.personality.teaching_style.value,
                'personality_traits': [trait.value for trait in teacher.personality.primary_traits],
                'formality_level': teacher.personality.formality_level,
                'difficulty_level': {
                    'min': teacher.specialization.min_difficulty.value,
                    'max': teacher.specialization.max_difficulty.value
                }
            }
            
            # Format chat history for the agent
            formatted_history = []
            for msg in chat_history:
                if msg.role != MessageRole.SYSTEM:  # Skip system messages
                    formatted_history.append({
                        'role': msg.role.value,
                        'content': msg.content
                    })
            
            # Perform multi-hop retrieval
            retrieval_result = await self.agent.multi_hop_retrieval(
                teacher_id=teacher.id,
                initial_query=query,
                teacher_info=teacher_info
            )
            
            # Generate response
            response = await self.agent.generate_response(
                teacher_id=teacher.id,
                query=query,
                contexts=retrieval_result.get('contexts', []),
                teacher_info=teacher_info,
                chat_history=formatted_history
            )
            
            # Return the response with metadata
            return {
                'response': response,
                'retrieval_metadata': {
                    'contexts_used': len(retrieval_result.get('contexts', [])),
                    'queries_used': retrieval_result.get('queries_used', []),
                    'sources': [
                        {
                            'title': ctx.get('metadata', {}).get('title', 'Untitled'),
                            'source': ctx.get('metadata', {}).get('source', 'Unknown'),
                            'score': ctx.get('score', 0)
                        }
                        for ctx in retrieval_result.get('contexts', [])
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Failed to enhance teacher response: {e}", exc_info=True)
            return {
                'response': f"I apologize, but I'm having trouble retrieving the information you need. Could you please try again or rephrase your question?",
                'retrieval_metadata': {
                    'contexts_used': 0,
                    'queries_used': [query],
                    'sources': [],
                    'error': str(e)
                }
            }

