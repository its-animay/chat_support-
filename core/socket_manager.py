from typing import Dict, List, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio
from datetime import datetime
from core.logger import logger

class ConnectionManager:
    """Manager for WebSocket connections with support for rooms (chat sessions)"""
    
    def __init__(self):
        # Map of chat_id -> list of WebSocket connections
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Map of user_id -> WebSocket for direct messages
        self.user_connections: Dict[str, WebSocket] = {}
        # Track users active in specific chats
        self.chat_users: Dict[str, List[str]] = {}
        # Track typing status: chat_id -> {user_id: timestamp}
        self.typing_status: Dict[str, Dict[str, float]] = {}
    
    async def connect(self, websocket: WebSocket, chat_id: str, user_id: str):
        """Connect a client to a specific chat room"""
        await websocket.accept()
        
        # Add connection to chat room
        if chat_id not in self.active_connections:
            self.active_connections[chat_id] = []
            self.chat_users[chat_id] = []
            self.typing_status[chat_id] = {}
            
        self.active_connections[chat_id].append(websocket)
        
        # Track user connection
        self.user_connections[user_id] = websocket
        
        # Add user to chat room if not already there
        if user_id not in self.chat_users[chat_id]:
            self.chat_users[chat_id].append(user_id)
            
        # Notify other users in this chat that a new user has joined
        await self.broadcast_to_chat(
            chat_id=chat_id,
            message={
                "event": "user_joined",
                "chat_id": chat_id,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "active_users": self.chat_users[chat_id]
            },
            exclude_user=None  # Include the joining user in the broadcast
        )
        
        logger.info(f"User {user_id} connected to chat {chat_id}")
    
    async def disconnect(self, websocket: WebSocket, chat_id: str, user_id: str):
        """Disconnect a client from a specific chat room"""
        # Remove from chat room
        if chat_id in self.active_connections:
            if websocket in self.active_connections[chat_id]:
                self.active_connections[chat_id].remove(websocket)
            
            # If this was the last connection for this chat, clean up
            if not self.active_connections[chat_id]:
                del self.active_connections[chat_id]
                del self.chat_users[chat_id]
                if chat_id in self.typing_status:
                    del self.typing_status[chat_id]
            else:
                # Remove user from chat users list
                if user_id in self.chat_users[chat_id]:
                    self.chat_users[chat_id].remove(user_id)
                
                # Remove typing status
                if chat_id in self.typing_status and user_id in self.typing_status[chat_id]:
                    del self.typing_status[chat_id][user_id]
                
                # Notify other users that this user has left
                await self.broadcast_to_chat(
                    chat_id=chat_id,
                    message={
                        "event": "user_left",
                        "chat_id": chat_id,
                        "user_id": user_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "active_users": self.chat_users[chat_id]
                    },
                    exclude_user=user_id
                )
        
        # Remove from user connections
        if user_id in self.user_connections:
            del self.user_connections[user_id]
            
        logger.info(f"User {user_id} disconnected from chat {chat_id}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}", exc_info=True)
    
    async def broadcast_to_chat(self, chat_id: str, message: Dict[str, Any], exclude_user: Optional[str] = None):
        """Broadcast a message to all connected clients in a chat room, with optional exclusion"""
        if chat_id not in self.active_connections:
            return
            
        disconnected = []
        for connection in self.active_connections[chat_id]:
            try:
                # Skip excluded user if specified
                if exclude_user:
                    user_connection = self.user_connections.get(exclude_user)
                    if connection == user_connection:
                        continue
                        
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to chat {chat_id}: {e}", exc_info=True)
                disconnected.append(connection)
        
        # Clean up any disconnected websockets
        for conn in disconnected:
            for room_id, connections in self.active_connections.items():
                if conn in connections:
                    connections.remove(conn)
    
    async def send_to_user(self, user_id: str, message: Dict[str, Any]):
        """Send a message to a specific user by user_id"""
        if user_id not in self.user_connections:
            return False
            
        try:
            await self.user_connections[user_id].send_json(message)
            return True
        except Exception as e:
            logger.error(f"Error sending to user {user_id}: {e}", exc_info=True)
            # Clean up the broken connection
            del self.user_connections[user_id]
            return False
    
    async def set_typing_status(self, chat_id: str, user_id: str, is_typing: bool):
        """Set a user's typing status in a chat"""
        if chat_id not in self.typing_status:
            self.typing_status[chat_id] = {}
            
        if is_typing:
            # Set typing timestamp
            self.typing_status[chat_id][user_id] = asyncio.get_event_loop().time()
        else:
            # Clear typing status
            if user_id in self.typing_status[chat_id]:
                del self.typing_status[chat_id][user_id]
        
        # Broadcast typing status to all users in the chat
        typing_users = list(self.typing_status[chat_id].keys()) if is_typing else []
        
        await self.broadcast_to_chat(
            chat_id=chat_id,
            message={
                "event": "typing_update",
                "chat_id": chat_id,
                "typing_users": typing_users,
                "timestamp": datetime.utcnow().isoformat()
            },
            exclude_user=None  # Include the typing user in the broadcast
        )
        
        return True
    
    async def mark_message_read(self, chat_id: str, message_id: str, user_id: str):
        """Mark a message as read by a user and notify other users"""
        await self.broadcast_to_chat(
            chat_id=chat_id,
            message={
                "event": "message_read",
                "chat_id": chat_id,
                "message_id": message_id,
                "read_by": user_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            exclude_user=None
        )
        
        return True
    
    def get_active_users(self, chat_id: str) -> List[str]:
        """Get list of active users in a chat"""
        return self.chat_users.get(chat_id, [])

# Create global instance
socket_manager = ConnectionManager()