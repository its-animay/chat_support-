from typing import List, Optional, Dict, Any
from models.chat import ChatSession, Message, MessageRole, ChatStart, ChatMessage, ChatResponse
from services.redis_client import redis_client
from services.teacher_service import TeacherService
from core.logger import logger
from datetime import datetime
import json

class ChatService:
    @staticmethod
    def start_chat(user_id: str, chat_data: ChatStart) -> Optional[ChatSession]:
        try:
            # Verify teacher exists
            teacher = TeacherService.get_teacher(chat_data.teacher_id)
            if not teacher:
                logger.error(f"Teacher not found: {chat_data.teacher_id}")
                return None
            
            # Create chat session
            chat = ChatSession(
                user_id=user_id,
                teacher_id=chat_data.teacher_id,
                title=chat_data.title or f"Chat with {teacher.name}"
            )
            
            # Store chat metadata
            chat_key = f"chat:{chat.id}"
            chat_dict = chat.dict()
            chat_dict['created_at'] = chat.created_at.isoformat()
            chat_dict['updated_at'] = chat.updated_at.isoformat()
            
            if not redis_client.json_set(chat_key, chat_dict):
                return None
            
            # Add to user's chat list
            user_chats_key = f"user:{user_id}:chats:{chat_data.teacher_id}"
            redis_client.list_push(user_chats_key, chat.id)
            
            # Add system message
            system_message = Message(
                role=MessageRole.SYSTEM,
                content=teacher.system_prompt
            )
            
            ChatService._add_message(chat.id, system_message)
            
            logger.info(f"Chat started: {chat.id}")
            return chat
            
        except Exception as e:
            logger.error(f"Failed to start chat: {e}")
            return None
    
    @staticmethod
    def send_message(chat_id: str, user_id: str, message_data: ChatMessage) -> Optional[ChatResponse]:
        try:
            # Verify chat exists and belongs to user
            chat_key = f"chat:{chat_id}"
            chat_data = redis_client.json_get(chat_key)
            
            if not chat_data or chat_data.get('user_id') != user_id:
                logger.error(f"Chat not found or unauthorized: {chat_id}")
                return None
            
            # Get teacher for this chat
            teacher = TeacherService.get_teacher(chat_data['teacher_id'])
            if not teacher:
                logger.error(f"Teacher not found: {chat_data['teacher_id']}")
                return None
            
            # Add user message
            user_message = Message(
                role=MessageRole.USER,
                content=message_data.content,
                metadata=message_data.metadata
            )
            
            ChatService._add_message(chat_id, user_message)
            
            # Generate AI response using LangGraph
            ai_response = ChatService._generate_ai_response(chat_id, teacher)
            
            if ai_response:
                # Add AI message
                ai_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=ai_response,
                    metadata={"teacher_id": teacher.id}
                )
                
                ChatService._add_message(chat_id, ai_message)
                
                # Update chat timestamp
                chat_data['updated_at'] = datetime.utcnow().isoformat()
                redis_client.json_set(chat_key, chat_data)
                
                return ChatResponse(
                    message_id=ai_message.id,
                    content=ai_response,
                    timestamp=ai_message.timestamp,
                    metadata=ai_message.metadata
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return None
    
    @staticmethod
    def get_chat_history(chat_id: str, user_id: str) -> List[Message]:
        try:
            # Verify chat exists and belongs to user
            chat_key = f"chat:{chat_id}"
            chat_data = redis_client.json_get(chat_key)
            
            if not chat_data or chat_data.get('user_id') != user_id:
                return []
            
            # Get messages from stream
            stream_key = f"chat:{chat_id}:messages"
            stream_messages = redis_client.stream_read(stream_key)
            
            messages = []
            for stream_msg in stream_messages:
                msg_data = stream_msg['data']
                msg_data['timestamp'] = datetime.fromisoformat(msg_data['timestamp'])
                if 'metadata' in msg_data:
                    msg_data['metadata'] = json.loads(msg_data['metadata'])
                messages.append(Message(**msg_data))
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get chat history: {e}")
            return []
    
    @staticmethod
    def get_user_chats(user_id: str, teacher_id: Optional[str] = None) -> List[ChatSession]:
        try:
            chats = []
            
            if teacher_id:
                # Get chats for specific teacher
                user_chats_key = f"user:{user_id}:chats:{teacher_id}"
                chat_ids = redis_client.list_get(user_chats_key)
            else:
                # Get all chats for user
                pattern = f"user:{user_id}:chats:*"
                chat_list_keys = redis_client.scan_keys(pattern)
                chat_ids = []
                for key in chat_list_keys:
                    chat_ids.extend(redis_client.list_get(key))
            
            for chat_id in chat_ids:
                chat_key = f"chat:{chat_id}"
                chat_data = redis_client.json_get(chat_key)
                if chat_data:
                    chat_data['created_at'] = datetime.fromisoformat(chat_data['created_at'])
                    chat_data['updated_at'] = datetime.fromisoformat(chat_data['updated_at'])
                    chats.append(ChatSession(**chat_data))
            
            return sorted(chats, key=lambda x: x.updated_at, reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to get user chats: {e}")
            return []
    
    @staticmethod
    def _add_message(chat_id: str, message: Message):
        stream_key = f"chat:{chat_id}:messages"
        message_data = {
            "id": message.id,
            "role": message.role.value,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
            "metadata": json.dumps(message.metadata)
        }
        redis_client.stream_add(stream_key, message_data)
    
    @staticmethod
    def _generate_ai_response(chat_id: str, teacher) -> Optional[str]:
        try:
            # Get recent chat history for context
            stream_key = f"chat:{chat_id}:messages"
            recent_messages = redis_client.stream_read(stream_key, count=10)
            
            # Build context
            context_messages = []
            for msg in recent_messages[-5:]:  # Last 5 messages
                role = msg['data']['role']
                content = msg['data']['content']
                if role != 'system':  # Skip system messages in context
                    context_messages.append(f"{role}: {content}")
            
            # Simple response generation (in production, integrate with LangGraph + OpenAI)
            context = "\n".join(context_messages)
            
            # For demo purposes, return a simple response
            # In production, this would use LangGraph with the teacher's configuration
            response = f"As {teacher.name}, a {teacher.domain} expert with a {teacher.personality} personality, I understand your message. Let me help you with that topic."
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate AI response: {e}")
            return "I apologize, but I'm having trouble processing your message right now. Please try again."
