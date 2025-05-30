from fastapi import APIRouter, HTTPException, Depends, Header, Query, Path, Body
from typing import List, Optional
from models.chat import ChatSession, Message, ChatStart, ChatMessage, ChatResponse
from services.chat_service import ChatService
from core.logger import logger
from pydantic import BaseModel, Field

router = APIRouter(prefix="/chat", tags=["chat"])

# Additional models for new endpoints
class MessageRating(BaseModel):
    rating: float = Field(..., ge=1, le=5, description="Rating from 1-5")

# Simple user ID extraction (in production, use proper auth)
async def get_current_user(x_user_id: str = Header(...)):
    return x_user_id

@router.post("/start", response_model=ChatSession)
async def start_chat(
    chat_data: ChatStart,
    user_id: str = Depends(get_current_user)
):
    """Start a new chat session with an AI teacher"""
    chat = ChatService.start_chat(user_id, chat_data)
    if not chat:
        raise HTTPException(
            status_code=400, 
            detail="Failed to start chat. The teacher may not exist."
        )
    return chat

@router.post("/{chat_id}/send", response_model=ChatResponse)
async def send_message(
    chat_id: str = Path(..., description="The ID of the chat session"),
    message: ChatMessage = Body(..., description="The message to send"),
    user_id: str = Depends(get_current_user)
):
    """Send a message in a chat session and get AI teacher response"""
    response = ChatService.send_message(chat_id, user_id, message)
    if not response:
        raise HTTPException(
            status_code=400, 
            detail="Failed to send message or generate response"
        )
    return response

@router.get("/{chat_id}/history", response_model=List[Message])
async def get_chat_history(
    chat_id: str = Path(..., description="The ID of the chat session"),
    user_id: str = Depends(get_current_user)
):
    """Get the full message history for a chat"""
    messages = ChatService.get_chat_history(chat_id, user_id)
    return messages

@router.get("/", response_model=List[ChatSession])
async def get_user_chats(
    teacher_id: Optional[str] = Query(None, description="Filter by teacher ID"),
    user_id: str = Depends(get_current_user)
):
    """Get all chat sessions for the current user"""
    chats = ChatService.get_user_chats(user_id, teacher_id)
    return chats

@router.post("/{chat_id}/message/{message_id}/rate", response_model=dict)
async def rate_message(
    chat_id: str = Path(..., description="The ID of the chat session"),
    message_id: str = Path(..., description="The ID of the message to rate"),
    rating_data: MessageRating = Body(..., description="Rating data"),
    user_id: str = Depends(get_current_user)
):
    """Rate a teacher's response and provide feedback"""
    success = ChatService.rate_chat_response(chat_id, message_id, user_id, rating_data.rating)
    if not success:
        raise HTTPException(
            status_code=400, 
            detail="Failed to rate message"
        )
    return {"message": "Rating submitted successfully", "rating": rating_data.rating}

@router.post("/{chat_id}/end", response_model=dict)
async def end_chat(
    chat_id: str = Path(..., description="The ID of the chat session"),
    user_id: str = Depends(get_current_user)
):
    """End a chat session"""
    success = ChatService.end_chat(chat_id, user_id)
    if not success:
        raise HTTPException(
            status_code=400, 
            detail="Failed to end chat session"
        )
    return {"message": "Chat session ended successfully"}