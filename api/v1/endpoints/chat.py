from fastapi import APIRouter, HTTPException, Depends, Header, Query, Path, Body, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from typing import List, Optional, Dict, Any
from models.chat import ChatSession, Message, ChatStart, ChatMessage, ChatResponse, MessageRole
from services.chat_service import ChatService
from services.chat_rag_integration import ChatRAGIntegration
from core.auth import get_current_user, get_user_from_token
from core.logger import logger
from pydantic import BaseModel, Field
import json 

router = APIRouter(prefix="/chat", tags=["chat"])

# Additional models for new endpoints
class MessageRating(BaseModel):
    rating: float = Field(..., ge=1, le=5, description="Rating from 1-5")

class MessageSourcesResponse(BaseModel):
    message_id: str
    rag_enhanced: bool
    sources: List[Dict[str, Any]] = []

@router.websocket("/{chat_id}")
async def websocket_chat(chat_id: str, websocket: WebSocket, token: str = Query(None)):
    if not token:
        await websocket.close(code=1008, reason="Missing token")
        return

    try:
        # Token validation before accepting connection
        user_info = await get_user_from_token(token)
        user_id = user_info.get("id")

        if not user_id:
            await websocket.close(code=1008, reason="Invalid user info")
            return

        await websocket.accept()  # Accept only after token is validated

        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = ChatMessage(**payload)

            response: ChatResponse = await ChatService.send_message(chat_id, user_id, message)

            if response:
                await websocket.send_json({
                    "type": "response",
                    "content": response.content,
                    "timestamp": str(response.timestamp),
                    "metadata": response.metadata,
                    "message_id": response.message_id
                })
            else:
                await websocket.send_json({
                    "type": "error",
                    "content": "Failed to generate AI response"
                })

    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {chat_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close(code=1011)


@router.post("/start", response_model=ChatSession)
async def start_chat(
    chat_data: ChatStart,
    request: Request,
    background_tasks: BackgroundTasks = None
):
    """Start a new chat session with an AI teacher"""
    # Get user ID from token
    user_id = await get_current_user(request)
    
    chat = await ChatService.start_chat(user_id, chat_data)
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
    request: Request = None
):
    """Send a message in a chat session and get AI teacher response with optional RAG enhancement"""
    # Get user ID from token
    user_id = await get_current_user(request)
    
    # Process with ChatService
    response = await ChatService.send_message(chat_id, user_id, message)
    if not response:
        raise HTTPException(
            status_code=400, 
            detail="Failed to send message or generate response"
        )
    return response

@router.get("/{chat_id}/history", response_model=List[Message])
async def get_chat_history(
    chat_id: str = Path(..., description="The ID of the chat session"),
    request: Request = None
):
    """Get the full message history for a chat"""
    # Get user ID from token
    user_id = await get_current_user(request)
    
    messages = await ChatService.get_chat_history(chat_id, user_id)
    return messages

@router.get("/", response_model=List[ChatSession])
async def get_user_chats(
    teacher_id: Optional[str] = Query(None, description="Filter by teacher ID"),
    request: Request = None
):
    """Get all chat sessions for the current user"""
    # Get user ID from token
    user_id = await get_current_user(request)
    
    chats = await ChatService.get_user_chats(user_id, teacher_id)
    return chats

@router.post("/{chat_id}/message/{message_id}/rate", response_model=dict)
async def rate_message(
    chat_id: str = Path(..., description="The ID of the chat session"),
    message_id: str = Path(..., description="The ID of the message to rate"),
    rating_data: MessageRating = Body(..., description="Rating data"),
    request: Request = None
):
    """Rate a teacher's response and provide feedback"""
    # Get user ID from token
    user_id = await get_current_user(request)
    
    success = await ChatService.rate_chat_response(chat_id, message_id, user_id, rating_data.rating)
    if not success:
        raise HTTPException(
            status_code=400, 
            detail="Failed to rate message"
        )
    return {"message": "Rating submitted successfully", "rating": rating_data.rating}

@router.post("/{chat_id}/end", response_model=dict)
async def end_chat(
    chat_id: str = Path(..., description="The ID of the chat session"),
    request: Request = None
):
    """End a chat session"""
    # Get user ID from token
    user_id = await get_current_user(request)
    
    success = await ChatService.end_chat(chat_id, user_id)
    if not success:
        raise HTTPException(
            status_code=400, 
            detail="Failed to end chat session"
        )
    return {"message": "Chat session ended successfully"}


@router.get("/{chat_id}/message/{message_id}/sources", response_model=MessageSourcesResponse)
async def get_message_sources(
    chat_id: str = Path(..., description="The ID of the chat session"),
    message_id: str = Path(..., description="The ID of the message"),
    request: Request = None
):
    """Get the sources used for a RAG-enhanced message"""
    # Get user ID from token
    user_id = await get_current_user(request)
    
    sources = await ChatService.get_message_sources(chat_id, message_id, user_id)
    if not sources:
        raise HTTPException(
            status_code=404, 
            detail="Message not found or not RAG-enhanced"
        )
    return sources

@router.get("/stats", response_model=Dict[str, Any])
async def get_chat_statistics(
    request: Request = None
):
    """Get statistics about chat usage for monitoring"""
    # Get user ID from token
    user_id = await get_current_user(request)
    
    # In a production system, you might want to check if the user has admin rights
    stats = await ChatService.get_chat_statistics()
    return stats

@router.post("/cleanup", response_model=Dict[str, Any])
async def cleanup_old_chats(
    days_threshold: int = Query(30, description="Age in days of chats to clean up"),
    request: Request = None,
    background_tasks: BackgroundTasks = None
):
    """Clean up old, inactive chats (admin operation)"""
    # Get user ID from token
    user_id = await get_current_user(request)
    
    # In a production system, you might want to check if the user has admin rights
    if background_tasks:
        background_tasks.add_task(ChatService.clean_expired_chats, background_tasks, days_threshold)
        return {"message": f"Cleanup of chats older than {days_threshold} days initiated"}
    else:
        deleted = await ChatService.clean_expired_chats(None, days_threshold)
        return {"message": f"Cleanup completed", "chats_deleted": deleted}
    
class RagEnableRequest(BaseModel):
    """Request to enable RAG for a chat message"""
    enable_rag: bool = Field(True, description="Whether to enable RAG")

@router.post("/{chat_id}/message/{message_id}/enable-rag", response_model=dict)
async def enable_rag_for_message(
    chat_id: str = Path(..., description="The ID of the chat session"),
    message_id: str = Path(..., description="The ID of the message to regenerate with RAG"),
    rag_data: RagEnableRequest = Body(..., description="RAG enable request"),
    request: Request = None
):
    """Enable Retrieval-Augmented Generation for a specific message"""
    # Get user ID from token
    user_id = await get_current_user(request)
    
    # Get chat history
    messages = await ChatService.get_chat_history(chat_id, user_id)
    
    # Find the target message
    target_message = None
    for message in messages:
        if message.id == message_id:
            target_message = message
            break
    
    if not target_message:
        raise HTTPException(
            status_code=404,
            detail="Message not found"
        )
    
    # Ensure it's a user message
    if target_message.role != MessageRole.USER:
        raise HTTPException(
            status_code=400,
            detail="RAG can only be enabled for user messages"
        )
    
    # Create new message with RAG enabled
    chat_message = ChatMessage(
        content=target_message.content,
        metadata={
            "use_rag": rag_data.enable_rag,
            "original_message_id": message_id
        }
    )
    
    # Send the message to get a RAG-enhanced response
    response = await ChatService.send_message(chat_id, user_id, chat_message)
    
    if not response:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate RAG-enhanced response"
        )
    
    return {
        "message": "RAG-enhanced response generated successfully",
        "original_message_id": message_id,
        "new_message_id": response.message_id,
        "rag_enabled": rag_data.enable_rag
    }