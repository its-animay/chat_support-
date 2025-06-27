from fastapi import APIRouter, HTTPException, Depends, Query, Body, Path, UploadFile, File, Form, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from services.rag_pipeline import rag_pipeline
from services.file_processor import file_processor
from services.document_processor import DocumentProcessor
from services.teacher_service import EnhancedTeacherService
from services.milvus_client import milvus_client
from core.logger import logger
import uuid
import json
import asyncio
import re

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

# Pydantic models for request/response
class TeacherKnowledgeBase(BaseModel):
    teacher_id: str = Field(..., description="Teacher ID")
    document_count: int = Field(..., description="Number of documents in knowledge base")

class TeacherKnowledgeBasesResponse(BaseModel):
    knowledge_bases: List[TeacherKnowledgeBase] = Field(..., description="List of teacher knowledge bases")
    total_teachers: int = Field(..., description="Total number of teachers with knowledge bases")

class FileUploadResponse(BaseModel):
    success: bool = Field(..., description="Whether upload was successful")
    teacher_id: str = Field(..., description="Teacher ID the file was uploaded for")
    filename: str = Field(..., description="Name of the uploaded file")
    documents_added: int = Field(..., description="Number of document chunks added")
    document_ids: List[str] = Field(default_factory=list, description="IDs of added documents")
    error: Optional[str] = Field(None, description="Error message if any")

class DocumentMetadata(BaseModel):
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    source: Optional[str] = Field(None, description="Document source")
    domain: Optional[str] = Field(None, description="Document domain or subject area")
    created_by: Optional[str] = Field(None, description="User who uploaded the document")
    custom_metadata: Optional[Dict[str, Any]] = Field(None, description="Custom metadata fields")

async def get_current_user(x_user_id: str = Depends(lambda: "default_user")):
    return x_user_id

@router.get("/teachers", response_model=TeacherKnowledgeBasesResponse)
async def list_teacher_knowledge_bases():
    """List all teacher knowledge bases with document counts"""
    try:
        result = await rag_pipeline.list_teacher_knowledge_bases()
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to list teacher knowledge bases"))
        
        # Convert to response format
        knowledge_bases = []
        for teacher_id, doc_count in result["teacher_knowledge_bases"].items():
            knowledge_bases.append({
                "teacher_id": teacher_id,
                "document_count": doc_count
            })
        
        return {
            "knowledge_bases": knowledge_bases,
            "total_teachers": result["total_teachers"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing teacher knowledge bases: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing teacher knowledge bases: {str(e)}")

@router.post("/upload", response_model=FileUploadResponse)
async def upload_file_to_teacher_knowledge_base(
    teacher_id: str = Form(..., description="Teacher ID to add document to"),
    file: UploadFile = File(..., description="File to upload"),
    title: Optional[str] = Form(None, description="Document title"),
    domain: Optional[str] = Form(None, description="Document domain or subject area"),
    author: Optional[str] = Form(None, description="Document author"),
    source: Optional[str] = Form(None, description="Document source"),
    custom_metadata: Optional[str] = Form(None, description="Custom metadata as JSON string"),
    background_tasks: BackgroundTasks = None,
    user_id: str = Depends(get_current_user)
):
    """Upload a file to a teacher's knowledge base"""
    try:
        # Verify the teacher exists
        teacher = await EnhancedTeacherService.get_teacher(teacher_id)
        if not teacher:
            raise HTTPException(status_code=404, detail=f"Teacher with ID {teacher_id} not found")
        
        # Read file content
        file_content = await file.read()
        
        # Parse custom metadata if provided
        parsed_custom_metadata = {}
        if custom_metadata:
            try:
                parsed_custom_metadata = json.loads(custom_metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in custom_metadata")
        
        # Prepare metadata
        metadata = {
            "title": title or file.filename,
            "author": author,
            "source": source,
            "domain": domain or teacher.specialization.primary_domain,
            "teacher_id": teacher_id,
            "teacher_name": teacher.name,
            "uploaded_by": user_id,
            "uploaded_at": str(uuid.uuid4()),
            **parsed_custom_metadata
        }
        
        # Process the file
        document_chunks = await file_processor.process_file(
            file_content=file_content,
            filename=file.filename,
            metadata=metadata
        )
        
        if not document_chunks:
            raise HTTPException(status_code=400, detail="Failed to process file. No document chunks were generated.")
        
        # Add document chunks directly to the teacher's knowledge base using teacher_id
        result = await rag_pipeline.add_documents(
            documents=document_chunks,
            teacher_id=teacher_id  # Use teacher_id directly - no collection_name needed
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to add documents to knowledge base"))
        
        return {
            "success": True,
            "teacher_id": teacher_id,
            "filename": file.filename,
            "documents_added": result["documents_added"],
            "document_ids": result["document_ids"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file to teacher knowledge base: {e}", exc_info=True)
        return {
            "success": False,
            "teacher_id": teacher_id,
            "filename": file.filename if 'file' in locals() else "unknown",
            "documents_added": 0,
            "document_ids": [],
            "error": str(e)
        }
    
@router.post("/bulk-upload", response_model=List[FileUploadResponse])
async def bulk_upload_files_to_teacher_knowledge_base(
    teacher_id: str = Form(..., description="Teacher ID to add documents to"),
    files: List[UploadFile] = File(..., description="Files to upload"),
    domain: Optional[str] = Form(None, description="Document domain or subject area"),
    source: Optional[str] = Form(None, description="Document source"),
    background_tasks: BackgroundTasks = None,
    user_id: str = Depends(get_current_user)
):
    """Upload multiple files to a teacher's knowledge base"""
    try:
        # Verify the teacher exists
        teacher = await EnhancedTeacherService.get_teacher(teacher_id)
        if not teacher:
            raise HTTPException(status_code=404, detail=f"Teacher with ID {teacher_id} not found")
        
        results = []
        
        for file in files:
            try:
                # Read file content
                file_content = await file.read()
                
                # Prepare metadata
                metadata = {
                    "title": file.filename,
                    "domain": domain or teacher.specialization.primary_domain,
                    "source": source,
                    "teacher_id": teacher_id,
                    "teacher_name": teacher.name,
                    "uploaded_by": user_id,
                    "uploaded_at": str(uuid.uuid4())  # Use UUID as a timestamp proxy
                }
                
                # Process the file
                document_chunks = await file_processor.process_file(
                    file_content=file_content,
                    filename=file.filename,
                    metadata=metadata
                )
                
                if not document_chunks:
                    results.append({
                        "success": False,
                        "teacher_id": teacher_id,
                        "filename": file.filename,
                        "documents_added": 0,
                        "document_ids": [],
                        "error": "Failed to process file. No document chunks were generated."
                    })
                    continue
                
                # Add document chunks to the teacher's knowledge base
                result = await rag_pipeline.add_documents(
                    documents=document_chunks,
                    teacher_id=teacher_id
                )
                
                if not result["success"]:
                    results.append({
                        "success": False,
                        "teacher_id": teacher_id,
                        "filename": file.filename,
                        "documents_added": 0,
                        "document_ids": [],
                        "error": result.get("error", "Failed to add documents to knowledge base")
                    })
                    continue
                
                results.append({
                    "success": True,
                    "teacher_id": teacher_id,
                    "filename": file.filename,
                    "documents_added": result["documents_added"],
                    "document_ids": result["document_ids"]
                })
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}", exc_info=True)
                results.append({
                    "success": False,
                    "teacher_id": teacher_id,
                    "filename": file.filename,
                    "documents_added": 0,
                    "document_ids": [],
                    "error": str(e)
                })
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in bulk upload: {str(e)}")

@router.post("/add-text", response_model=FileUploadResponse)
async def add_text_to_teacher_knowledge_base(
    teacher_id: str = Form(..., description="Teacher ID to add document to"),
    content: str = Form(..., description="Text content to add"),
    title: str = Form(..., description="Document title"),
    domain: Optional[str] = Form(None, description="Document domain or subject area"),
    author: Optional[str] = Form(None, description="Document author"),
    source: Optional[str] = Form(None, description="Document source"),
    custom_metadata: Optional[str] = Form(None, description="Custom metadata as JSON string"),
    user_id: str = Depends(get_current_user)
):
    """Add text content directly to a teacher's knowledge base"""
    try:
        # Verify the teacher exists
        teacher = await EnhancedTeacherService.get_teacher(teacher_id)
        if not teacher:
            raise HTTPException(status_code=404, detail=f"Teacher with ID {teacher_id} not found")
        
        # Parse custom metadata if provided
        parsed_custom_metadata = {}
        if custom_metadata:
            try:
                parsed_custom_metadata = json.loads(custom_metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in custom_metadata")
        
        # Prepare metadata
        metadata = {
            "title": title,
            "author": author,
            "source": source,
            "domain": domain or teacher.specialization.primary_domain,
            "teacher_id": teacher_id,
            "teacher_name": teacher.name,
            "uploaded_by": user_id,
            "uploaded_at": str(uuid.uuid4()),  # Use UUID as a timestamp proxy
            "format": "text",
            **parsed_custom_metadata
        }
        
        # Create document
        document = {
            "id": str(uuid.uuid4()),
            "content": content,
            "metadata": metadata
        }
        
        # Chunk the document
        document_chunks = DocumentProcessor.chunk_document(
            document=document
        )
        
        if not document_chunks:
            raise HTTPException(status_code=400, detail="Failed to process text. No document chunks were generated.")
        
        # Add document chunks to the teacher's knowledge base
        result = await rag_pipeline.add_documents(
            documents=document_chunks,
            teacher_id=teacher_id
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to add documents to knowledge base"))
        
        return {
            "success": True,
            "teacher_id": teacher_id,
            "filename": title,
            "documents_added": result["documents_added"],
            "document_ids": result["document_ids"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding text to teacher knowledge base: {e}", exc_info=True)
        return {
            "success": False,
            "teacher_id": teacher_id,
            "filename": title if 'title' in locals() else "unknown",
            "documents_added": 0,
            "document_ids": [],
            "error": str(e)
        }

@router.delete("/{teacher_id}/documents", response_model=Dict[str, Any])
async def delete_documents_from_knowledge_base(
    teacher_id: str = Path(..., description="Teacher ID"),
    document_ids: List[str] = Body(..., description="List of document IDs to delete"),
    user_id: str = Depends(get_current_user)
):
    """Delete documents from a teacher's knowledge base"""
    try:
        # Verify the teacher exists
        teacher = await EnhancedTeacherService.get_teacher(teacher_id)
        if not teacher:
            raise HTTPException(status_code=404, detail=f"Teacher with ID {teacher_id} not found")
        
        # Delete documents
        result = await rag_pipeline.delete_documents(
            document_ids=document_ids,
            teacher_id=teacher_id
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to delete documents"))
        
        return {
            "success": True,
            "teacher_id": teacher_id,
            "documents_deleted": result["documents_deleted"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting documents from knowledge base: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting documents: {str(e)}")

@router.post("/{teacher_id}/query", response_model=Dict[str, Any])
async def query_teacher_knowledge_base(
    teacher_id: str = Path(..., description="Teacher ID"),
    query: str = Body(..., description="Query to search for"),
    top_k: int = Body(10, description="Number of results to return"),
    user_id: str = Depends(get_current_user)
):
    """Query a teacher's knowledge base directly"""
    try:
        # Verify the teacher exists
        teacher = await EnhancedTeacherService.get_teacher(teacher_id)
        if not teacher:
            raise HTTPException(status_code=404, detail=f"Teacher with ID {teacher_id} not found")
        
        # Search the knowledge base
        results = await rag_pipeline.process_query(
            query=query,
            top_k=top_k,
            top_n=min(5, top_k),
            teacher_id=teacher_id
        )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying teacher knowledge base: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error querying knowledge base: {str(e)}")

@router.get("/{teacher_id}/stats", response_model=Dict[str, Any])
async def get_teacher_knowledge_base_stats(
    teacher_id: str = Path(..., description="Teacher ID"),
    user_id: str = Depends(get_current_user)
):
    """Get statistics about a teacher's knowledge base"""
    try:
        # Verify the teacher exists
        teacher = await EnhancedTeacherService.get_teacher(teacher_id)
        if not teacher:
            raise HTTPException(status_code=404, detail=f"Teacher with ID {teacher_id} not found")
        
        # Get document count directly from Milvus client
        doc_count = await milvus_client.get_teacher_document_count(teacher_id)
        
        return {
            "teacher_id": teacher_id,
            "teacher_name": teacher.name,
            "document_count": doc_count,
            "domain": teacher.specialization.primary_domain,
            "has_knowledge_base": doc_count > 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting teacher knowledge base stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting knowledge base stats: {str(e)}")

@router.post("/{teacher_id}/clear", response_model=Dict[str, Any])
async def clear_teacher_knowledge_base(
    teacher_id: str = Path(..., description="Teacher ID"),
    confirm: bool = Body(..., description="Confirmation flag"),
    user_id: str = Depends(get_current_user)
):
    """Clear all documents from a teacher's knowledge base"""
    try:
        if not confirm:
            raise HTTPException(status_code=400, detail="Confirmation is required to clear the knowledge base")
        
        # Verify the teacher exists
        teacher = await EnhancedTeacherService.get_teacher(teacher_id)
        if not teacher:
            raise HTTPException(status_code=404, detail=f"Teacher with ID {teacher_id} not found")
        
        # Delete all documents for this teacher
        result = await rag_pipeline.delete_teacher_documents(teacher_id)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to clear knowledge base"))
        
        return {
            "success": True,
            "teacher_id": teacher_id,
            "message": "Knowledge base cleared successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing teacher knowledge base: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error clearing knowledge base: {str(e)}")