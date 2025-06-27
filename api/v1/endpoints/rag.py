from fastapi import APIRouter, HTTPException, Depends, Query, Body, Path, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from services.rag_pipeline import rag_pipeline
from services.chat_service import ChatService
from core.logger import logger
import uuid

router = APIRouter(prefix="/rag", tags=["rag"])

# Pydantic models for request/response
class RagQueryRequest(BaseModel):
    query: str = Field(..., description="User query to process")
    top_k: int = Field(10, ge=1, le=50, description="Number of initial documents to retrieve")
    top_n: int = Field(3, ge=1, le=10, description="Number of documents to keep after reranking")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature for LLM generation")
    filter_expr: Optional[str] = Field(None, description="Optional filter expression for retrieval")
    use_cache: bool = Field(True, description="Whether to use cache for results")
    collection_name: Optional[str] = Field(None, description="Custom collection name")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt for LLM")

class RagDocumentBase(BaseModel):
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")

class RagDocumentCreate(RagDocumentBase):
    id: Optional[str] = Field(None, description="Optional document ID (will be generated if not provided)")

class RagDocumentResponse(RagDocumentBase):
    id: str = Field(..., description="Document ID")

class RagQueryResponse(BaseModel):
    response: str = Field(..., description="Generated response")
    sources_used: List[Dict[str, Any]] = Field(default_factory=list, description="Sources used in generation")
    rag_enhanced: bool = Field(..., description="Whether response was enhanced with RAG")
    processing_time: float = Field(..., description="Processing time in seconds")
    cached: bool = Field(..., description="Whether result was from cache")
    retrieval_count: Optional[int] = Field(None, description="Number of documents retrieved")
    reranked_count: Optional[int] = Field(None, description="Number of documents after reranking")
    error: Optional[str] = Field(None, description="Error message if any")

class RagAddDocumentsRequest(BaseModel):
    documents: List[RagDocumentCreate] = Field(..., description="Documents to add")
    collection_name: Optional[str] = Field(None, description="Custom collection name")

class RagAddDocumentsResponse(BaseModel):
    success: bool = Field(..., description="Whether operation was successful")
    documents_added: int = Field(..., description="Number of documents added")
    document_ids: List[str] = Field(default_factory=list, description="IDs of added documents")
    error: Optional[str] = Field(None, description="Error message if any")

class RagDeleteDocumentsRequest(BaseModel):
    document_ids: List[str] = Field(..., description="Document IDs to delete")
    collection_name: Optional[str] = Field(None, description="Custom collection name")

class RagDeleteDocumentsResponse(BaseModel):
    success: bool = Field(..., description="Whether operation was successful")
    documents_deleted: int = Field(..., description="Number of documents deleted")
    error: Optional[str] = Field(None, description="Error message if any")

class RagHealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status")
    milvus: Dict[str, Any] = Field(..., description="Milvus health status")
    components: Dict[str, bool] = Field(..., description="Component health statuses")
    document_count: int = Field(..., description="Number of documents in collection")


@router.post("/query", response_model=RagQueryResponse)
async def query_rag(
    request: RagQueryRequest = Body(..., description="Query request parameters"),
    background_tasks: BackgroundTasks = None
):
    """Process a query through the RAG pipeline and return enhanced response"""
    try:
        response = await rag_pipeline.process_query(
            query=request.query,
            top_k=request.top_k,
            top_n=request.top_n,
            temperature=request.temperature,
            collection_name=request.collection_name,
            filter_expr=request.filter_expr,
            use_cache=request.use_cache,
            system_prompt=request.system_prompt
        )
        
        return response
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/documents", response_model=RagAddDocumentsResponse)
async def add_documents(
    request: RagAddDocumentsRequest = Body(..., description="Documents to add"),
    background_tasks: BackgroundTasks = None
):
    """Add documents to the RAG knowledge base"""
    try:
        # Prepare documents, assigning IDs if not provided
        documents = []
        for doc in request.documents:
            doc_dict = doc.dict()
            if not doc_dict.get("id"):
                doc_dict["id"] = str(uuid.uuid4())
            documents.append(doc_dict)
        
        result = await rag_pipeline.add_documents(
            documents=documents,
            collection_name=request.collection_name
        )
        
        return result
    except Exception as e:
        logger.error(f"Error adding documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error adding documents: {str(e)}")


@router.delete("/documents", response_model=RagDeleteDocumentsResponse)
async def delete_documents(
    request: RagDeleteDocumentsRequest = Body(..., description="Documents to delete"),
    background_tasks: BackgroundTasks = None
):
    """Delete documents from the RAG knowledge base"""
    try:
        result = await rag_pipeline.delete_documents(
            document_ids=request.document_ids,
            collection_name=request.collection_name
        )
        
        return result
    except Exception as e:
        logger.error(f"Error deleting documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting documents: {str(e)}")


@router.get("/health", response_model=RagHealthResponse)
async def health_check():
    """Check health of the RAG system"""
    try:
        health = await rag_pipeline.health_check()
        return health
    except Exception as e:
        logger.error(f"Error checking RAG health: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error checking health: {str(e)}")


# Integration with chat system
@router.post("/chat/{chat_id}/enhance", response_model=Dict[str, Any])
async def enhance_chat_with_rag(
    chat_id: str = Path(..., description="Chat session ID"),
    message_id: str = Query(..., description="Message ID to enhance"),
    user_id: str = Query(..., description="User ID"),
    top_k: int = Query(10, ge=1, le=50, description="Number of initial documents to retrieve"),
    top_n: int = Query(3, ge=1, le=10, description="Number of documents to keep after reranking")
):
    """Enhance a chat message with RAG capabilities"""
    try:
        # Get the message from chat history
        messages = await ChatService.get_chat_history(chat_id, user_id)
        
        target_message = None
        for message in messages:
            if message.id == message_id:
                target_message = message
                break
        
        if not target_message:
            raise HTTPException(status_code=404, detail=f"Message {message_id} not found")
        
        # Process the message content through RAG
        response = await rag_pipeline.process_query(
            query=target_message.content,
            top_k=top_k,
            top_n=top_n
        )
        
        # Create new message with RAG response
        chat_message = {
            "content": response["response"],
            "metadata": {
                "rag_enhanced": True,
                "sources_used": response.get("sources_used", []),
                "original_message_id": message_id
            }
        }
        
        # Add the enhanced response to the chat
        result = await ChatService.send_message(chat_id, user_id, chat_message)
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to add enhanced message to chat")
        
        return {
            "success": True,
            "original_message_id": message_id,
            "enhanced_message_id": result.message_id,
            "rag_enhanced": True,
            "sources_used": response.get("sources_used", [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enhancing chat with RAG: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error enhancing chat: {str(e)}")