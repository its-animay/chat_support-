from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body, UploadFile, File
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
from services.rag_service import RagService

router = APIRouter(prefix="/knowledge-base", tags=["knowledge-base"])

# Models for API requests and responses
class DocumentCreate(BaseModel):
    title: str = Field(..., description="Document title")
    text: str = Field(..., description="Document content")
    source: Optional[str] = Field(None, description="Source of the document")
    domain: Optional[str] = Field(None, description="Primary domain")
    sub_domains: List[str] = Field(default_factory=list, description="List of sub-domains")
    difficulty_level: Optional[str] = Field(None, description="Difficulty level")
    tags: List[str] = Field(default_factory=list, description="List of tags")
    
class DocumentResponse(BaseModel):
    id: str
    title: str
    source: Optional[str] = None
    domain: Optional[str] = None
    sub_domains: List[str] = []
    difficulty_level: Optional[str] = None
    tags: List[str] = []
    timestamp: datetime
    
class DocumentsAddResponse(BaseModel):
    success: bool
    count: int
    message: str
    
class CollectionResponse(BaseModel):
    teacher_id: str
    document_count: int
    exists: bool

# Initialize the RAG service
rag_service = RagService()

# Knowledge base endpoints
@router.post("/document/{teacher_id}", response_model=DocumentsAddResponse)
async def add_document(
    teacher_id: str = Path(..., description="The teacher ID"),
    document: DocumentCreate = Body(..., description="The document to add")
):
    """Add a document to the teacher's knowledge base"""
    # Format the document
    doc = {
        "id": f"doc_{int(datetime.now().timestamp())}",
        "title": document.title,
        "text": document.text,
        "source": document.source,
        "domain": document.domain,
        "sub_domains": document.sub_domains,
        "difficulty_level": document.difficulty_level,
        "tags": document.tags,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add the document
    success, count = await rag_service.add_documents_for_teacher(teacher_id, [doc])
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to add document to knowledge base"
        )
    
    return {
        "success": success,
        "count": count,
        "message": f"Successfully added {count} document chunks to the knowledge base"
    }

@router.post("/documents/{teacher_id}", response_model=DocumentsAddResponse)
async def add_documents(
    teacher_id: str = Path(..., description="The teacher ID"),
    documents: List[DocumentCreate] = Body(..., description="The documents to add")
):
    """Add multiple documents to the teacher's knowledge base"""
    # Format the documents
    docs = []
    for i, doc in enumerate(documents):
        docs.append({
            "id": f"doc_{int(datetime.now().timestamp())}_{i}",
            "title": doc.title,
            "text": doc.text,
            "source": doc.source,
            "domain": doc.domain,
            "sub_domains": doc.sub_domains,
            "difficulty_level": doc.difficulty_level,
            "tags": doc.tags,
            "timestamp": datetime.now().isoformat()
        })
    
    # Add the documents
    success, count = await rag_service.add_documents_for_teacher(teacher_id, docs)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to add documents to knowledge base"
        )
    
    return {
        "success": success,
        "count": count,
        "message": f"Successfully added {count} document chunks to the knowledge base"
    }

@router.delete("/document/{teacher_id}/{document_id}", response_model=Dict[str, Any])
def delete_document(
    teacher_id: str = Path(..., description="The teacher ID"),
    document_id: str = Path(..., description="The document ID")
):
    """Delete a document from the teacher's knowledge base"""
    success = rag_service.delete_document(teacher_id, document_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found or could not be deleted"
        )
    
    return {
        "success": success,
        "message": f"Document {document_id} deleted successfully"
    }

@router.get("/collection/{teacher_id}", response_model=CollectionResponse)
def get_collection_info(
    teacher_id: str = Path(..., description="The teacher ID")
):
    """Get information about a teacher's knowledge base collection"""
    exists = rag_service.collection_exists(teacher_id)
    count = 0
    
    if exists:
        count = rag_service.get_document_count(teacher_id)
    
    return {
        "teacher_id": teacher_id,
        "document_count": count,
        "exists": exists
    }

@router.post("/collection/{teacher_id}", response_model=Dict[str, Any])
def create_collection(
    teacher_id: str = Path(..., description="The teacher ID")
):
    """Create a knowledge base collection for a teacher"""
    if rag_service.collection_exists(teacher_id):
        return {
            "success": True,
            "message": f"Collection for teacher {teacher_id} already exists"
        }
    
    success = rag_service.create_collection(teacher_id)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create collection for teacher {teacher_id}"
        )
    
    return {
        "success": success,
        "message": f"Collection for teacher {teacher_id} created successfully"
    }

@router.delete("/collection/{teacher_id}", response_model=Dict[str, Any])
def delete_collection(
    teacher_id: str = Path(..., description="The teacher ID")
):
    """Delete a teacher's knowledge base collection"""
    success = rag_service.delete_collection(teacher_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Collection for teacher {teacher_id} not found or could not be deleted"
        )
    
    return {
        "success": success,
        "message": f"Collection for teacher {teacher_id} deleted successfully"
    }
