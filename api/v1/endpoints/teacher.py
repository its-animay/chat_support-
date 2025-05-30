from fastapi import APIRouter, HTTPException, Depends
from typing import List
from models.teacher import Teacher, TeacherCreate, TeacherUpdate
from services.teacher_service import TeacherService
from core.logger import logger

router = APIRouter(prefix="/teacher", tags=["teachers"])

@router.post("/", response_model=Teacher)
async def create_teacher(teacher_data: TeacherCreate):
    """Create a new AI teacher agent"""
    teacher = TeacherService.create_teacher(teacher_data)
    if not teacher:
        raise HTTPException(status_code=500, detail="Failed to create teacher")
    return teacher

@router.get("/{teacher_id}", response_model=Teacher)
async def get_teacher(teacher_id: str):
    """Get a specific teacher by ID"""
    teacher = TeacherService.get_teacher(teacher_id)
    if not teacher:
        raise HTTPException(status_code=404, detail="Teacher not found")
    return teacher

@router.put("/{teacher_id}", response_model=Teacher)
async def update_teacher(teacher_id: str, update_data: TeacherUpdate):
    """Update a teacher's configuration"""
    teacher = TeacherService.update_teacher(teacher_id, update_data)
    if not teacher:
        raise HTTPException(status_code=404, detail="Teacher not found")
    return teacher

@router.delete("/{teacher_id}")
async def delete_teacher(teacher_id: str):
    """Delete a teacher and all associated data"""
    success = TeacherService.delete_teacher(teacher_id)
    if not success:
        raise HTTPException(status_code=404, detail="Teacher not found")
    return {"message": "Teacher deleted successfully"}

@router.get("/", response_model=List[Teacher])
async def list_teachers():
    """List all available teachers"""
    return TeacherService.list_teachers()