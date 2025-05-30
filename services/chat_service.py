from typing import List, Optional, Dict, Any
from models.chat import ChatSession, Message, MessageRole, ChatStart, ChatMessage, ChatResponse
from models.teacher import EnhancedTeacher
from services.redis_client import redis_client
from services.teacher_service import EnhancedTeacherService
from core.logger import logger
from utils.helpers import serialize_datetime, deserialize_datetime
from datetime import datetime
import json
from langgraph.factory import LangGraphAgentFactory

class ChatService:
    @staticmethod
    def start_chat(user_id: str, chat_data: ChatStart) -> Optional[ChatSession]:
        try:
            # Verify teacher exists - use EnhancedTeacherService
            teacher = EnhancedTeacherService.get_teacher(chat_data.teacher_id)
            if not teacher:
                logger.error(f"Teacher not found: {chat_data.teacher_id}")
                return None
            
            # Create chat session
            chat = ChatSession(
                user_id=user_id,
                teacher_id=chat_data.teacher_id,
                title=chat_data.title or f"Chat with {teacher.name}"
            )
            
            # Store chat metadata with proper datetime serialization
            chat_key = f"chat:{chat.id}"
            chat_dict = chat.dict()
            chat_dict['created_at'] = serialize_datetime(chat.created_at)
            chat_dict['updated_at'] = serialize_datetime(chat.updated_at)
            
            if not redis_client.json_set(chat_key, chat_dict):
                return None
            
            # Add to user's chat list
            user_chats_key = f"user:{user_id}:chats:{chat_data.teacher_id}"
            redis_client.list_push(user_chats_key, chat.id)
            
            # Generate the personalized system prompt for this teacher
            context = {
                "user_id": user_id,
                "chat_id": chat.id,
                "session_start": True
            }
            system_prompt = teacher.generate_system_prompt(context)
            
            # Add system message
            system_message = Message(
                role=MessageRole.SYSTEM,
                content=system_prompt
            )
            
            ChatService._add_message(chat.id, system_message)
            
            # Increment the teacher's session count
            EnhancedTeacherService.increment_session_count(chat_data.teacher_id)
            
            logger.info(f"Chat started: {chat.id} with teacher: {teacher.name}")
            return chat
            
        except Exception as e:
            logger.error(f"Failed to start chat: {e}", exc_info=True)
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
            
            # Get enhanced teacher for this chat
            teacher = EnhancedTeacherService.get_teacher(chat_data['teacher_id'])
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
            
            # Get chat history for context
            chat_history = ChatService.get_chat_history(chat_id, user_id)
            
            # Format chat history for LLM
            formatted_messages = []
            for msg in chat_history:
                if msg.role != MessageRole.SYSTEM:  # Skip system messages
                    formatted_messages.append({
                        "role": msg.role.value,
                        "content": msg.content
                    })
            
            # Generate AI response using teacher personality
            context = {
                "chat_id": chat_id,
                "teaching_style": teacher.personality.teaching_style.value,
                "formality_level": teacher.personality.formality_level,
                "response_length": teacher.personality.response_length,
                "use_examples": teacher.personality.use_examples,
                "use_analogies": teacher.personality.use_analogies,
                "primary_traits": [t.value for t in teacher.personality.primary_traits],
                "domain": teacher.specialization.primary_domain,
                "specializations": teacher.specialization.specializations,
                "difficulty_level": {
                    "min": teacher.specialization.min_difficulty.value,
                    "max": teacher.specialization.max_difficulty.value
                }
            }
            
            ai_response = LangGraphAgentFactory.generate_response(
                teacher=teacher,
                messages=formatted_messages,
                context=context
            )
            
            if ai_response:
                # Add AI message
                ai_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=ai_response,
                    metadata={
                        "teacher_id": teacher.id,
                        "teacher_name": teacher.name,
                        "domain": teacher.specialization.primary_domain,
                        "teaching_style": teacher.personality.teaching_style.value
                    }
                )
                
                ChatService._add_message(chat_id, ai_message)
                
                # Update chat timestamp with proper serialization
                chat_data['updated_at'] = serialize_datetime(datetime.utcnow())
                redis_client.json_set(chat_key, chat_data)
                
                return ChatResponse(
                    message_id=ai_message.id,
                    content=ai_response,
                    timestamp=ai_message.timestamp,
                    metadata=ai_message.metadata
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}", exc_info=True)
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
                msg_data['timestamp'] = deserialize_datetime(msg_data['timestamp'])
                if 'metadata' in msg_data and isinstance(msg_data['metadata'], str):
                    try:
                        msg_data['metadata'] = json.loads(msg_data['metadata'])
                    except json.JSONDecodeError:
                        msg_data['metadata'] = {}
                messages.append(Message(**msg_data))
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get chat history: {e}", exc_info=True)
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
                    chat_data['created_at'] = deserialize_datetime(chat_data['created_at'])
                    chat_data['updated_at'] = deserialize_datetime(chat_data['updated_at'])
                    chats.append(ChatSession(**chat_data))
            
            return sorted(chats, key=lambda x: x.updated_at, reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to get user chats: {e}", exc_info=True)
            return []
    
    @staticmethod
    def rate_chat_response(chat_id: str, message_id: str, user_id: str, rating: float) -> bool:
        """Rate a teacher's response and update their statistics"""
        try:
            # Verify chat exists and belongs to user
            chat_key = f"chat:{chat_id}"
            chat_data = redis_client.json_get(chat_key)
            
            if not chat_data or chat_data.get('user_id') != user_id:
                return False
            
            # Add rating for the teacher
            teacher_id = chat_data.get('teacher_id')
            if teacher_id:
                # Store rating for this specific message
                rating_key = f"chat:{chat_id}:message:{message_id}:rating"
                redis_client.json_set(rating_key, {"rating": rating, "timestamp": serialize_datetime(datetime.utcnow())})
                
                # Update teacher's overall rating
                EnhancedTeacherService.add_teacher_rating(teacher_id, rating)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to rate chat response: {e}", exc_info=True)
            return False
    
    @staticmethod
    def end_chat(chat_id: str, user_id: str) -> bool:
        """Mark a chat as ended"""
        try:
            # Verify chat exists and belongs to user
            chat_key = f"chat:{chat_id}"
            chat_data = redis_client.json_get(chat_key)
            
            if not chat_data or chat_data.get('user_id') != user_id:
                return False
            
            # Update chat metadata
            chat_data['ended_at'] = serialize_datetime(datetime.utcnow())
            chat_data['is_active'] = False
            
            return redis_client.json_set(chat_key, chat_data)
            
        except Exception as e:
            logger.error(f"Failed to end chat: {e}", exc_info=True)
            return False
    
    @staticmethod
    def _add_message(chat_id: str, message: Message):
        stream_key = f"chat:{chat_id}:messages"
        message_data = {
            "id": message.id,
            "role": message.role.value,
            "content": message.content,
            "timestamp": serialize_datetime(message.timestamp),
            "metadata": json.dumps(message.metadata)
        }
        redis_client.stream_add(stream_key, message_data)