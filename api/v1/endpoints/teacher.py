from fastapi import APIRouter, HTTPException, Depends, Query, Body, Path
from typing import List, Optional, Dict, Any
from models.teacher import (
    EnhancedTeacher, 
    EnhancedTeacherCreate, 
    EnhancedTeacherUpdate,
    TeachingStyle,
    PersonalityTrait,
    DifficultyLevel
)
from services.teacher_service import EnhancedTeacherService
from core.logger import logger
from pydantic import BaseModel, Field
from datetime import datetime

router = APIRouter(prefix="/enhanced-teacher", tags=["enhanced-teachers"])

# Additional Pydantic models for API responses
class TeacherSearchResponse(BaseModel):
    teachers: List[EnhancedTeacher]
    pagination: Dict[str, Any]

class TeacherRatingRequest(BaseModel):
    rating: float = Field(..., ge=0, le=5, description="Rating value between 0 and 5")

class TeacherPromptRequest(BaseModel):
    context: Dict[str, Any] = Field(default_factory=dict, description="Optional context for prompt generation")

class TeacherPromptResponse(BaseModel):
    teacher_id: str
    name: str
    system_prompt: str

class TeacherStylesResponse(BaseModel):
    teaching_styles: List[Dict[str, str]]
    personality_traits: List[Dict[str, str]]
    difficulty_levels: List[Dict[str, str]]

# Standard CRUD endpoints
@router.post("/", response_model=EnhancedTeacher)
async def create_teacher(teacher_data: EnhancedTeacherCreate):
    """Create a new enhanced AI teacher with rich personality and optional custom ID"""
    teacher = await EnhancedTeacherService.create_teacher(teacher_data)
    if not teacher:
        # Check if it's a duplicate ID issue
        if teacher_data.id:
            existing_teacher = await EnhancedTeacherService.get_teacher(teacher_data.id)
            if existing_teacher:
                raise HTTPException(
                    status_code=409, 
                    detail=f"A teacher with ID '{teacher_data.id}' already exists. Please use a different ID."
                )
        
        raise HTTPException(
            status_code=500, 
            detail="Failed to create enhanced teacher. Please check the logs or contact support."
        )
    return teacher

@router.get("/{teacher_id}", response_model=EnhancedTeacher)
async def get_teacher(teacher_id: str = Path(..., description="The ID of the enhanced teacher to retrieve")):
    """Get a specific enhanced teacher by ID"""
    teacher = await EnhancedTeacherService.get_teacher(teacher_id)
    if not teacher:
        raise HTTPException(
            status_code=404, 
            detail=f"Enhanced teacher with ID {teacher_id} not found"
        )
    return teacher

@router.put("/{teacher_id}", response_model=EnhancedTeacher)
async def update_teacher(
    teacher_id: str = Path(..., description="The ID of the enhanced teacher to update"),
    update_data: EnhancedTeacherUpdate = Body(..., description="The data to update")
):
    """Update an enhanced teacher's configuration"""
    teacher = await EnhancedTeacherService.update_teacher(teacher_id, update_data)
    if not teacher:
        raise HTTPException(
            status_code=404, 
            detail=f"Enhanced teacher with ID {teacher_id} not found or update failed"
        )
    return teacher

@router.delete("/{teacher_id}")
async def delete_teacher(teacher_id: str = Path(..., description="The ID of the enhanced teacher to delete")):
    """Delete an enhanced teacher and all associated data"""
    success = await EnhancedTeacherService.delete_teacher(teacher_id)
    if not success:
        raise HTTPException(
            status_code=404, 
            detail=f"Enhanced teacher with ID {teacher_id} not found or deletion failed"
        )
    return {"message": f"Enhanced teacher {teacher_id} deleted successfully"}

@router.get("/", response_model=List[EnhancedTeacher])
async def list_teachers():
    """List all available enhanced teachers"""
    return await EnhancedTeacherService.list_teachers()

# Enhanced endpoints
@router.get("/search", response_model=TeacherSearchResponse)
async def search_teachers(
    domain: Optional[str] = Query(None, description="Filter by primary domain or specialization"),
    teaching_style: Optional[TeachingStyle] = Query(None, description="Filter by teaching style"),
    difficulty_level: Optional[DifficultyLevel] = Query(None, description="Filter by difficulty level capability"),
    traits: Optional[List[PersonalityTrait]] = Query(None, description="Filter by personality traits"),
    query: Optional[str] = Query(None, description="Search in name, domain, and specializations"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page")
):
    """Search enhanced teachers with rich filtering options"""
    return await EnhancedTeacherService.search_teachers(
        domain=domain,
        teaching_style=teaching_style,
        difficulty_level=difficulty_level,
        traits=traits,
        query=query,
        page=page,
        limit=limit
    )

@router.get("/domain/{domain}", response_model=EnhancedTeacher)
async def get_teacher_by_domain(domain: str = Path(..., description="The domain to search for")):
    """Get an enhanced teacher by its domain expertise"""
    teacher = await EnhancedTeacherService.get_teacher_by_domain(domain)
    if not teacher:
        raise HTTPException(
            status_code=404, 
            detail=f"No enhanced teacher found for domain '{domain}'"
        )
    return teacher

@router.post("/{teacher_id}/rating", response_model=Dict[str, Any])
async def add_teacher_rating(
    teacher_id: str = Path(..., description="The ID of the enhanced teacher"),
    rating_data: TeacherRatingRequest = Body(..., description="Rating data")
):
    """Add a rating for an enhanced teacher"""
    success = await EnhancedTeacherService.add_teacher_rating(teacher_id, rating_data.rating)
    if not success:
        raise HTTPException(
            status_code=404, 
            detail=f"Enhanced teacher with ID {teacher_id} not found or rating failed"
        )
    return {"message": f"Rating added successfully for teacher {teacher_id}"}

@router.post("/{teacher_id}/increment-session", response_model=Dict[str, Any])
async def increment_session(
    teacher_id: str = Path(..., description="The ID of the enhanced teacher")
):
    """Increment the session count for an enhanced teacher"""
    success = await EnhancedTeacherService.increment_session_count(teacher_id)
    if not success:
        raise HTTPException(
            status_code=404, 
            detail=f"Enhanced teacher with ID {teacher_id} not found or increment failed"
        )
    return {"message": f"Session count incremented for teacher {teacher_id}"}

@router.post("/{teacher_id}/generate-prompt", response_model=TeacherPromptResponse)
async def generate_system_prompt(
    teacher_id: str = Path(..., description="The ID of the enhanced teacher"),
    prompt_data: TeacherPromptRequest = Body(TeacherPromptRequest(), description="Context for prompt generation")
):
    """Generate a context-aware system prompt for an enhanced teacher"""
    teacher = await EnhancedTeacherService.get_teacher(teacher_id)
    if not teacher:
        raise HTTPException(
            status_code=404, 
            detail=f"Enhanced teacher with ID {teacher_id} not found"
        )
    
    system_prompt = await EnhancedTeacherService.generate_system_prompt(teacher_id, prompt_data.context)
    if not system_prompt:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate system prompt for teacher {teacher_id}"
        )
    
    return {
        "teacher_id": teacher_id,
        "name": teacher.name,
        "system_prompt": system_prompt
    }

@router.post("/create-defaults", response_model=Dict[str, Any])
async def create_default_teachers():
    """Create default enhanced teacher profiles if none exist"""
    return await EnhancedTeacherService.create_default_teachers()

@router.get("/styles/all", response_model=TeacherStylesResponse)
async def get_all_styles():
    """Get all available teaching styles, personality traits, and difficulty levels"""
    return {
        "teaching_styles": EnhancedTeacherService.get_all_teaching_styles(),
        "personality_traits": EnhancedTeacherService.get_all_personality_traits(),
        "difficulty_levels": EnhancedTeacherService.get_all_difficulty_levels()
    }