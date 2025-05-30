from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ChatSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    teacher_id: str
    title: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ChatStart(BaseModel):
    teacher_id: str
    title: Optional[str] = None

class ChatMessage(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ChatResponse(BaseModel):
    message_id: str
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
