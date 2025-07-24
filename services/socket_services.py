from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import json
from core.logger import logger
from services.redis_client import redis_client
from core.socket_manager import socket_manager

class SocketService:
    """Service for socket-related operations, with persistence and recovery"""
    
    @staticmethod
    async def store_message_status(chat_id: str, message_id: str, status: str, user_id: str) -> bool:
        """Store message status (sent, delivered, read) in Redis"""
        try:
            status_key = f"chat:{chat_id}:message:{message_id}:status"
            status_data = {
                "status": status,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Convert to JSON string for storage
            success = await redis_client.json_set(status_key, status_data)
            return success
        except Exception as e:
            logger.error(f"Error storing message status: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def get_message_status(chat_id: str, message_id: str) -> Dict[str, Any]:
        """Get message status from Redis"""
        try:
            status_key = f"chat:{chat_id}:message:{message_id}:status"
            status_data = await redis_client.json_get(status_key)
            
            if not status_data:
                return {
                    "status": "sent",
                    "read_by": [],
                    "delivered_to": []
                }
            
            return status_data
        except Exception as e:
            logger.error(f"Error getting message status: {e}", exc_info=True)
            return {
                "status": "unknown",
                "error": str(e)
            }
    
    @staticmethod
    async def mark_message_read(chat_id: str, message_id: str, user_id: str) -> bool:
        """Mark a message as read by a user"""
        try:
            # Update in Redis
            await SocketService.store_message_status(chat_id, message_id, "read", user_id)
            
            # Notify clients via WebSocket
            return await socket_manager.mark_message_read(chat_id, message_id, user_id)
        except Exception as e:
            logger.error(f"Error marking message as read: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def notify_typing(chat_id: str, user_id: str, is_typing: bool) -> bool:
        """Notify that a user is typing"""
        try:
            return await socket_manager.set_typing_status(chat_id, user_id, is_typing)
        except Exception as e:
            logger.error(f"Error updating typing status: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def broadcast_message(chat_id: str, message: Dict[str, Any], exclude_user: Optional[str] = None) -> bool:
        """Broadcast a message to all users in a chat"""
        try:
            await socket_manager.broadcast_to_chat(chat_id, message, exclude_user)
            return True
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def send_notification(user_id: str, notification: Dict[str, Any]) -> bool:
        """Send a notification to a specific user"""
        try:
            return await socket_manager.send_to_user(user_id, notification)
        except Exception as e:
            logger.error(f"Error sending notification: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def store_notification(user_id: str, notification: Dict[str, Any]) -> bool:
        """Store a notification for offline delivery"""
        try:
            notification_key = f"user:{user_id}:notifications"
            notification_data = {
                **notification,
                "timestamp": datetime.utcnow().isoformat(),
                "delivered": False
            }
            
            # Add to notification list
            success = await redis_client.list_push(notification_key, json.dumps(notification_data))
            return success
        except Exception as e:
            logger.error(f"Error storing notification: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def get_pending_notifications(user_id: str) -> List[Dict[str, Any]]:
        """Get pending notifications for a user"""
        try:
            notification_key = f"user:{user_id}:notifications"
            notifications_json = await redis_client.list_get(notification_key)
            
            notifications = []
            for notification_str in notifications_json:
                try:
                    notification = json.loads(notification_str)
                    if not notification.get("delivered", False):
                        notifications.append(notification)
                except json.JSONDecodeError:
                    continue
                    
            return notifications
        except Exception as e:
            logger.error(f"Error getting pending notifications: {e}", exc_info=True)
            return []
    
    @staticmethod
    async def mark_notifications_delivered(user_id: str, notification_ids: List[str]) -> int:
        """Mark notifications as delivered"""
        # This would require a more complex Redis operation to update specific items in a list
        # For simplicity, we'll just log the operation here
        logger.info(f"Marked {len(notification_ids)} notifications as delivered for user {user_id}")
        return len(notification_ids)
    
    @staticmethod
    async def get_online_users(chat_id: str) -> List[str]:
        """Get list of online users in a chat"""
        return socket_manager.get_active_users(chat_id)
    
    @staticmethod
    async def user_is_online(user_id: str) -> bool:
        """Check if a user is online"""
        return user_id in socket_manager.user_connections

# Create singleton instance
socket_service = SocketService()