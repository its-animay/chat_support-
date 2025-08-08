from typing import List, Optional, Dict, Any
from models.chat import ChatSession, Message, MessageRole, ChatStart, ChatMessage, ChatResponse
from models.teacher import EnhancedTeacher
from services.redis_client import redis_client
from services.teacher_service import EnhancedTeacherService
from services.chat_rag_integration import ChatRAGIntegration
from core.logger import logger
from utils.helpers import serialize_datetime, deserialize_datetime
from datetime import datetime
import json
import asyncio
import uuid
from langgraph.factory import LangGraphAgentFactory
from fastapi import BackgroundTasks

class ChatService:
    """Enhanced chat service with async support, fault tolerance, and high concurrency capabilities"""

    # Cache for active chat sessions to reduce Redis load
    _active_sessions_cache = {}
    # Maximum size of active sessions cache
    _max_cache_size = 10000
    # Semaphore for limiting concurrent Redis operations
    _redis_semaphore = asyncio.Semaphore(100)  # Limit to 100 concurrent Redis operations


    @staticmethod
    async def start_chat(user_id: str, chat_data: ChatStart) -> Optional[ChatSession]:
        """Start a new chat session with async/transaction support"""
        try:
            # Use semaphore to limit concurrent Redis operations
            async with ChatService._redis_semaphore:
                # Verify teacher exists - use EnhancedTeacherService
                teacher = await EnhancedTeacherService.get_teacher(chat_data.teacher_id)
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
                
                # User's chat list key
                user_chats_key = f"user:{user_id}:chats:{chat_data.teacher_id}"
                
                # Generate the personalized system prompt for this teacher
                context = {
                    "user_id": user_id,
                    "chat_id": chat.id,
                    "session_start": True,
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
                    },
                    "use_rag": False  
                }
                system_prompt = teacher.generate_system_prompt(context)
                
                # Create system message
                system_message = Message(
                    role=MessageRole.SYSTEM,
                    content=system_prompt
                )
                
                stream_key = f"chat:{chat.id}:messages"
                message_data = {
                    "id": system_message.id,
                    "role": system_message.role.value,
                    "content": system_message.content,
                    "timestamp": serialize_datetime(system_message.timestamp),
                    "metadata": json.dumps(system_message.metadata)
                }
                
                # Use transaction to ensure all operations succeed or fail together
                async with await redis_client.transaction() as tx:
                    await tx.json_set(chat_key, chat_dict)
                    await tx.list_push(user_chats_key, chat.id)
                    await tx.stream_add(stream_key, message_data)
                
                if not tx.success:
                    logger.error(f"Failed to create chat session in Redis: {chat.id}")
                    return None
                
                # Store in local cache for faster access
                ChatService._add_to_cache(chat_key, chat_dict)
                
                # Increment the teacher's session count (non-critical operation, do in background)
                asyncio.create_task(EnhancedTeacherService.increment_session_count(chat_data.teacher_id))
                
                logger.info(f"Chat started: {chat.id} with teacher: {teacher.name}")
                return chat
                
        except Exception as e:
            logger.error(f"Failed to start chat: {e}", exc_info=True)
            return None
    
    @staticmethod
    async def send_message(chat_id: str, user_id: str, message_data: ChatMessage) -> Optional[ChatResponse]:
        """Send a message in a chat session and get AI teacher response"""
        try:
            # Use semaphore to limit concurrent Redis operations
            async with ChatService._redis_semaphore:
                # Verify chat exists and belongs to user
                chat_key = f"chat:{chat_id}"
                
                # Try cache first
                chat_data = ChatService._get_from_cache(chat_key)
                
                # If not in cache, get from Redis
                if not chat_data:
                    chat_data = await redis_client.json_get(chat_key)
                
                if not chat_data or chat_data.get('user_id') != user_id:
                    logger.error(f"Chat not found or unauthorized: {chat_id}")
                    return None
                
                # Get enhanced teacher for this chat (can run in parallel with message storage)
                teacher_future = asyncio.create_task(
                    EnhancedTeacherService.get_teacher(chat_data['teacher_id'])
                )
                
                # Add user message
                user_message = Message(
                    role=MessageRole.USER,
                    content=message_data.content,
                    metadata=message_data.metadata
                )
                
                # Add message to Redis
                message_added = await ChatService._add_message(chat_id, user_message)
                if not message_added:
                    logger.error(f"Failed to add user message to chat: {chat_id}")
                    return None
                
                # Get chat history for context (in parallel)
                history_future = asyncio.create_task(
                    ChatService.get_chat_history(chat_id, user_id)
                )
                
                # Wait for teacher and history futures to complete
                teacher = await teacher_future
                if not teacher:
                    logger.error(f"Teacher not found: {chat_data['teacher_id']}")
                    return None
                
                chat_history = await history_future
            
            # Determine if RAG should be used
            use_rag = await ChatRAGIntegration.should_use_rag(message_data.dict(), teacher)
            
            # Context for response generation
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
                },
                "use_rag": use_rag
            }
            
            ai_response = None
            sources_used = []
            rag_enhanced = False
            
            # Generate response based on RAG setting
            if use_rag:
                # Use RAG pipeline for enhanced response
                rag_result = await ChatRAGIntegration.enhance_response_with_rag(
                    user_message=message_data.content,
                    chat_history=chat_history,
                    teacher_id=teacher.id,
                    context=context
                )
                
                ai_response = rag_result.get("content")
                sources_used = rag_result.get("sources_used", [])
                rag_enhanced = rag_result.get("rag_enhanced", False)
                processing_time = rag_result.get("processing_time", 0)
                
                logger.info(f"RAG response generated in {processing_time:.2f}s with {len(sources_used)} sources")
            else:
                # Use standard LLM response
                formatted_messages = []
                for msg in chat_history:
                    if msg.role != MessageRole.SYSTEM:  # Skip system messages
                        formatted_messages.append({
                            "role": msg.role.value,
                            "content": msg.content
                        })
                
                ai_response = await LangGraphAgentFactory.generate_response(
                    teacher=teacher,
                    messages=formatted_messages,
                    context=context
                )
                
                logger.info(f"Standard response generated for chat {chat_id}")
            
            if ai_response:
                # Create AI message with appropriate metadata
                ai_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=ai_response,
                    metadata={
                        "teacher_id": teacher.id,
                        "teacher_name": teacher.name,
                        "domain": teacher.specialization.primary_domain,
                        "teaching_style": teacher.personality.teaching_style.value,
                        "rag_enhanced": rag_enhanced,
                        "sources_used": sources_used
                    }
                )
                
                # Store the AI message in Redis (with semaphore)
                async with ChatService._redis_semaphore:
                    # Add AI message
                    await ChatService._add_message(chat_id, ai_message)
                    
                    # Update chat timestamp with proper serialization
                    chat_data['updated_at'] = serialize_datetime(datetime.utcnow())
                    await redis_client.json_set(chat_key, chat_data)
                    
                    # Update cache
                    ChatService._add_to_cache(chat_key, chat_data)
                
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
    async def get_chat_history(chat_id: str, user_id: str) -> List[Message]:
        """Get chat history with async support"""
        try:
            async with ChatService._redis_semaphore:
                # Verify chat exists and belongs to user
                chat_key = f"chat:{chat_id}"
                
                # Try cache first
                chat_data = ChatService._get_from_cache(chat_key)
                
                # If not in cache, get from Redis
                if not chat_data:
                    chat_data = await redis_client.json_get(chat_key)
                
                if not chat_data or chat_data.get('user_id') != user_id:
                    return []
                
                # Get messages from stream
                stream_key = f"chat:{chat_id}:messages"
                stream_messages = await redis_client.stream_read(stream_key)
                
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
    async def get_user_chats(user_id: str, teacher_id: Optional[str] = None) -> List[ChatSession]:
        """Get user chats with async support and batching"""
        try:
            chat_ids = []
            
            async with ChatService._redis_semaphore:
                if teacher_id:
                    # Get chats for specific teacher
                    user_chats_key = f"user:{user_id}:chats:{teacher_id}"
                    chat_ids = await redis_client.list_get(user_chats_key)
                else:
                    # Get all chats for user
                    pattern = f"user:{user_id}:chats:*"
                    chat_list_keys = await redis_client.scan_keys(pattern)
                    
                    # Get chat IDs in parallel for better performance
                    chat_id_tasks = [redis_client.list_get(key) for key in chat_list_keys]
                    chat_id_lists = await asyncio.gather(*chat_id_tasks)
                    
                    # Flatten the lists
                    for id_list in chat_id_lists:
                        chat_ids.extend(id_list)
            
            # Batch chat data retrieval for better performance
            # Process in batches of 20 to avoid overwhelming Redis
            chats = []
            batch_size = 20
            
            for i in range(0, len(chat_ids), batch_size):
                batch = chat_ids[i:i+batch_size]
                
                # Create tasks for parallel execution
                async with ChatService._redis_semaphore:
                    chat_tasks = [redis_client.json_get(f"chat:{chat_id}") for chat_id in batch]
                    chat_data_list = await asyncio.gather(*chat_tasks)
                
                # Process results
                for chat_data in chat_data_list:
                    if chat_data:
                        chat_data['created_at'] = deserialize_datetime(chat_data['created_at'])
                        chat_data['updated_at'] = deserialize_datetime(chat_data['updated_at'])
                        chats.append(ChatSession(**chat_data))
            
            return sorted(chats, key=lambda x: x.updated_at, reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to get user chats: {e}", exc_info=True)
            return []
    
    @staticmethod
    async def rate_chat_response(chat_id: str, message_id: str, user_id: str, rating: float) -> bool:
        """Rate a teacher's response with async support"""
        try:
            async with ChatService._redis_semaphore:
                # Verify chat exists and belongs to user
                chat_key = f"chat:{chat_id}"
                
                # Try cache first
                chat_data = ChatService._get_from_cache(chat_key)
                
                # If not in cache, get from Redis
                if not chat_data:
                    chat_data = await redis_client.json_get(chat_key)
                
                if not chat_data or chat_data.get('user_id') != user_id:
                    return False
                
                # Add rating for the teacher
                teacher_id = chat_data.get('teacher_id')
                if teacher_id:
                    # Store rating for this specific message
                    rating_key = f"chat:{chat_id}:message:{message_id}:rating"
                    rating_data = {
                        "rating": rating, 
                        "timestamp": serialize_datetime(datetime.utcnow()),
                        "user_id": user_id
                    }
                    
                    # Use transaction for the rating operations
                    async with await redis_client.transaction() as tx:
                        await tx.json_set(rating_key, rating_data)
                    
                    if not tx.success:
                        logger.error(f"Failed to store rating for message {message_id}")
                        return False
                    
                    # Update teacher's overall rating (non-critical, can be done in background)
                    asyncio.create_task(
                        EnhancedTeacherService.add_teacher_rating(teacher_id, rating)
                    )
                    
                    return True
                
                return False
            
        except Exception as e:
            logger.error(f"Failed to rate chat response: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def end_chat(chat_id: str, user_id: str) -> bool:
        """Mark a chat as ended with async support"""
        try:
            async with ChatService._redis_semaphore:
                # Verify chat exists and belongs to user
                chat_key = f"chat:{chat_id}"
                
                # Try cache first
                chat_data = ChatService._get_from_cache(chat_key)
                
                # If not in cache, get from Redis
                if not chat_data:
                    chat_data = await redis_client.json_get(chat_key)
                
                if not chat_data or chat_data.get('user_id') != user_id:
                    return False
                
                # Update chat metadata
                chat_data['ended_at'] = serialize_datetime(datetime.utcnow())
                chat_data['is_active'] = False
                
                # Store the updated chat data
                success = await redis_client.json_set(chat_key, chat_data)
                
                # Update cache if successful
                if success:
                    ChatService._add_to_cache(chat_key, chat_data)
                
                return success
            
        except Exception as e:
            logger.error(f"Failed to end chat: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def _add_message(chat_id: str, message: Message) -> bool:
        """Add a message to the chat with async support"""
        try:
            stream_key = f"chat:{chat_id}:messages"
            message_data = {
                "id": message.id,
                "role": message.role.value,
                "content": message.content,
                "timestamp": serialize_datetime(message.timestamp),
                "metadata": json.dumps(message.metadata)
            }
            
            msg_id = await redis_client.stream_add(stream_key, message_data)
            return msg_id is not None
        
        except Exception as e:
            logger.error(f"Failed to add message to chat {chat_id}: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def get_message_sources(chat_id: str, message_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get sources used for a specific RAG-enhanced message with async support"""
        try:
            async with ChatService._redis_semaphore:
                # Verify chat exists and belongs to user
                chat_key = f"chat:{chat_id}"
                
                # Try cache first
                chat_data = ChatService._get_from_cache(chat_key)
                
                # If not in cache, get from Redis
                if not chat_data:
                    chat_data = await redis_client.json_get(chat_key)
                
                if not chat_data or chat_data.get('user_id') != user_id:
                    logger.error(f"Chat not found or unauthorized: {chat_id}")
                    return None
                
                # Get chat history
                stream_key = f"chat:{chat_id}:messages"
                stream_messages = await redis_client.stream_read(stream_key)
                
                # Find the specific message
                for stream_msg in stream_messages:
                    msg_data = stream_msg['data']
                    if msg_data.get('id') == message_id:
                        # Parse metadata
                        metadata = msg_data.get('metadata', '{}')
                        if isinstance(metadata, str):
                            try:
                                metadata = json.loads(metadata)
                            except json.JSONDecodeError:
                                metadata = {}
                        
                        # Extract sources
                        sources = metadata.get('sources_used', [])
                        rag_enhanced = metadata.get('rag_enhanced', False)
                        
                        return {
                            "message_id": message_id,
                            "rag_enhanced": rag_enhanced,
                            "sources": sources
                        }
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get message sources: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _add_to_cache(key: str, data: Dict[str, Any]) -> None:
        """Add data to the in-memory cache with LRU eviction"""
        try:
            # Check if cache is full
            if len(ChatService._active_sessions_cache) >= ChatService._max_cache_size:
                # Remove the oldest item (simple LRU implementation)
                oldest_key = next(iter(ChatService._active_sessions_cache))
                ChatService._active_sessions_cache.pop(oldest_key)
            
            # Add to cache
            ChatService._active_sessions_cache[key] = {
                'data': data,
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Failed to add to cache: {e}", exc_info=True)
    
    @staticmethod
    def _get_from_cache(key: str) -> Optional[Dict[str, Any]]:
        """Get data from the in-memory cache"""
        try:
            cache_item = ChatService._active_sessions_cache.get(key)
            if cache_item:
                # Update timestamp to mark as recently used
                cache_item['timestamp'] = datetime.utcnow()
                return cache_item['data']
            return None
        except Exception as e:
            logger.error(f"Failed to get from cache: {e}", exc_info=True)
            return None
    
    @staticmethod
    async def clean_expired_chats(background_tasks: BackgroundTasks, days_threshold: int = 30) -> int:
        """Clean up old, inactive chats to save storage (run as scheduled task)"""
        try:
            pattern = "chat:*"
            cutoff_date = datetime.utcnow() - datetime.timedelta(days=days_threshold)
            cutoff_str = serialize_datetime(cutoff_date)
            
            chats_deleted = 0
            cursor = "0"
            batch_size = 100
            
            while True:
                # Scan for chat keys in batches
                async with ChatService._redis_semaphore:
                    keys = await redis_client.scan_keys(pattern, cursor, batch_size)
                    
                    if not keys:
                        break
                    
                    for key in keys:
                        chat_data = await redis_client.json_get(key)
                        if not chat_data:
                            continue
                        
                        updated_at = chat_data.get('updated_at', '')
                        is_active = chat_data.get('is_active', True)
                        
                        if not is_active and updated_at < cutoff_str:
                            chat_id = key.split(':')[1]
                            
                            async with await redis_client.transaction() as tx:
                                # Delete chat data
                                await tx.delete(key)
                                
                                # Delete messages
                                message_key = f"chat:{chat_id}:messages"
                                await tx.delete(message_key)
                                
                                # We don't delete from user's chat list as that would require
                                # complex operations - those will just reference non-existent chats
                            
                            if tx.success:
                                chats_deleted += 1
                
                if cursor == "0":
                    break
            
            logger.info(f"Cleaned up {chats_deleted} expired chat sessions")
            return chats_deleted
            
        except Exception as e:
            logger.error(f"Failed to clean expired chats: {e}", exc_info=True)
            return 0
    
    @staticmethod
    async def get_chat_statistics() -> Dict[str, Any]:
        """Get statistics about chat usage for monitoring"""
        try:
            stats = {
                "total_chats": 0,
                "active_chats": 0,
                "messages_last_24h": 0,
                "cache_size": len(ChatService._active_sessions_cache),
                "popular_teachers": {},
                "rag_usage": {
                    "enabled": 0,
                    "disabled": 0,
                    "percentage": 0
                }
            }
            
            chat_pattern = "chat:*"
            chat_keys = await redis_client.scan_keys(chat_pattern)
            stats["total_chats"] = len(chat_keys)
            
            sample_size = min(100, len(chat_keys))
            if sample_size > 0:
                sample_keys = chat_keys[:sample_size]
                
                chat_data_tasks = [redis_client.json_get(key) for key in sample_keys]
                chat_data_list = await asyncio.gather(*chat_data_tasks)
                
                active_count = sum(1 for data in chat_data_list if data and data.get('is_active', False))
                
                stats["active_chats"] = int((active_count / sample_size) * stats["total_chats"])
                
                for data in chat_data_list:
                    if data and 'teacher_id' in data:
                        teacher_id = data['teacher_id']
                        if teacher_id in stats["popular_teachers"]:
                            stats["popular_teachers"][teacher_id] += 1
                        else:
                            stats["popular_teachers"][teacher_id] = 1
                
                stats["popular_teachers"] = dict(
                    sorted(
                        stats["popular_teachers"].items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:5]
                )
            
            cutoff_time = datetime.utcnow() - datetime.timedelta(hours=24)
            cutoff_str = serialize_datetime(cutoff_time)
            
            message_pattern = "chat:*:messages"
            message_keys = await redis_client.scan_keys(message_pattern)
            
            sample_size = min(50, len(message_keys))
            if sample_size > 0:
                sample_keys = message_keys[:sample_size]
                
                recent_message_counts = []
                rag_enabled = 0
                rag_disabled = 0
                
                for key in sample_keys:
                    messages = await redis_client.stream_read(key)
                    recent_count = 0
                    
                    for msg in messages:
                        if msg['data'].get('timestamp', '') > cutoff_str:
                            recent_count += 1
                            
                            # Check RAG usage
                            metadata = msg['data'].get('metadata', '{}')
                            if isinstance(metadata, str):
                                try:
                                    metadata = json.loads(metadata)
                                except json.JSONDecodeError:
                                    metadata = {}
                            
                            if metadata.get('rag_enhanced', False):
                                rag_enabled += 1
                            else:
                                rag_disabled += 1
                    
                    recent_message_counts.append(recent_count)
                
                avg_recent_messages = sum(recent_message_counts) / len(recent_message_counts)
                
                stats["messages_last_24h"] = int(avg_recent_messages * len(message_keys))
                
                # Calculate RAG usage percentage
                total_rag_messages = rag_enabled + rag_disabled
                if total_rag_messages > 0:
                    stats["rag_usage"] = {
                        "enabled": rag_enabled,
                        "disabled": rag_disabled,
                        "percentage": round((rag_enabled / total_rag_messages) * 100, 2)
                    }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get chat statistics: {e}", exc_info=True)
            return {
                "error": str(e),
                "total_chats": 0,
                "active_chats": 0,
                "messages_last_24h": 0,
                "cache_size": len(ChatService._active_sessions_cache)
            }