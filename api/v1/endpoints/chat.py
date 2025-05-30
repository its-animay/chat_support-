from fastapi import APIRouter, HTTPException, Depends, Header
from typing import List, Optional
from models.chat import ChatSession, Message, ChatStart, ChatMessage, ChatResponse
from services.chat_service import ChatService
from core.logger import logger

router = APIRouter(prefix="/chat", tags=["chat"])

# Simple user ID extraction (in production, use proper auth)
async def get_current_user(x_user_id: str = Header(...)):
    return x_user_id

@router.post("/start", response_model=ChatSession)
async def start_chat(
    chat_data: ChatStart,
    user_id: str = Depends(get_current_user)
):
    """Start a new chat session with a teacher"""
    chat = ChatService.start_chat(user_id, chat_data)
    if not chat:
        raise HTTPException(status_code=400, detail="Failed to start chat")
    return chat

@router.post("/{chat_id}/send", response_model=ChatResponse)
async def send_message(
    chat_id: str,
    message: ChatMessage,
    user_id: str = Depends(get_current_user)
):
    """Send a message in a chat session"""
    response = ChatService.send_message(chat_id, user_id, message)
    if not response:
        raise HTTPException(status_code=400, detail="Failed to send message")
    return response

@router.get("/{chat_id}/history", response_model=List[Message])
async def get_chat_history(
    chat_id: str,
    user_id: str = Depends(get_current_user)
):
    """Get the full message history for a chat"""
    messages = ChatService.get_chat_history(chat_id, user_id)
    return messages

@router.get("/", response_model=List[ChatSession])
async def get_user_chats(
    teacher_id: Optional[str] = None,
    user_id: str = Depends(get_current_user)
):
    """Get all chat sessions for the current user"""
    chats = ChatService.get_user_chats(user_id, teacher_id)
    return chats
