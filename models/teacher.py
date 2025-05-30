from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class TeacherCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    domain: str = Field(..., min_length=1, max_length=100)
    personality: str = Field(..., min_length=1, max_length=500)
    system_prompt: str = Field(..., min_length=1)
    tools: Optional[List[str]] = Field(default_factory=list)

class TeacherUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    domain: Optional[str] = Field(None, min_length=1, max_length=100)
    personality: Optional[str] = Field(None, min_length=1, max_length=500)
    system_prompt: Optional[str] = Field(None, min_length=1)
    tools: Optional[List[str]] = None

class Teacher(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    domain: str
    personality: str
    system_prompt: str
    tools: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }