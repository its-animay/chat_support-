from typing import List, Optional, Dict, Any
from models.teacher import Teacher, TeacherCreate, TeacherUpdate
from services.redis_client import redis_client
from core.logger import logger
import json
from datetime import datetime

class TeacherService:
    @staticmethod
    def create_teacher(teacher_data: TeacherCreate) -> Optional[Teacher]:
        try:
            teacher = Teacher(**teacher_data.dict())
            teacher_key = f"teacher:{teacher.id}"
            
            teacher_dict = teacher.dict()
            teacher_dict['created_at'] = teacher.created_at.isoformat()
            teacher_dict['updated_at'] = teacher.updated_at.isoformat()
            
            if redis_client.json_set(teacher_key, teacher_dict):
                logger.info(f"Teacher created: {teacher.id}")
                return teacher
            return None
        except Exception as e:
            logger.error(f"Failed to create teacher: {e}")
            return None
    
    @staticmethod
    def get_teacher(teacher_id: str) -> Optional[Teacher]:
        try:
            teacher_key = f"teacher:{teacher_id}"
            teacher_data = redis_client.json_get(teacher_key)
            
            if teacher_data:
                # Convert ISO strings back to datetime
                teacher_data['created_at'] = datetime.fromisoformat(teacher_data['created_at'])
                teacher_data['updated_at'] = datetime.fromisoformat(teacher_data['updated_at'])
                return Teacher(**teacher_data)
            return None
        except Exception as e:
            logger.error(f"Failed to get teacher {teacher_id}: {e}")
            return None
    
    @staticmethod
    def update_teacher(teacher_id: str, update_data: TeacherUpdate) -> Optional[Teacher]:
        try:
            teacher = TeacherService.get_teacher(teacher_id)
            if not teacher:
                return None
            
            update_dict = update_data.dict(exclude_unset=True)
            for key, value in update_dict.items():
                setattr(teacher, key, value)
            
            teacher.updated_at = datetime.utcnow()
            
            teacher_key = f"teacher:{teacher_id}"
            teacher_dict = teacher.dict()
            teacher_dict['created_at'] = teacher.created_at.isoformat()
            teacher_dict['updated_at'] = teacher.updated_at.isoformat()
            
            if redis_client.json_set(teacher_key, teacher_dict):
                logger.info(f"Teacher updated: {teacher_id}")
                return teacher
            return None
        except Exception as e:
            logger.error(f"Failed to update teacher {teacher_id}: {e}")
            return None
    
    @staticmethod
    def delete_teacher(teacher_id: str) -> bool:
        try:
            teacher_key = f"teacher:{teacher_id}"
            
            # Delete teacher profile
            if not redis_client.json_delete(teacher_key):
                return False
            
            # Clean up associated chats
            chat_pattern = f"chat:*:teacher:{teacher_id}"
            chat_keys = redis_client.scan_keys(chat_pattern)
            
            for chat_key in chat_keys:
                redis_client.delete(chat_key)
            
            # Clean up user chat lists
            user_chat_pattern = f"user:*:chats:{teacher_id}"
            user_chat_keys = redis_client.scan_keys(user_chat_pattern)
            
            for user_chat_key in user_chat_keys:
                redis_client.delete(user_chat_key)
            
            logger.info(f"Teacher deleted: {teacher_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete teacher {teacher_id}: {e}")
            return False
    
    @staticmethod
    def list_teachers() -> List[Teacher]:
        try:
            teacher_keys = redis_client.scan_keys("teacher:*")
            teachers = []
            
            for key in teacher_keys:
                teacher_data = redis_client.json_get(key)
                if teacher_data:
                    teacher_data['created_at'] = datetime.fromisoformat(teacher_data['created_at'])
                    teacher_data['updated_at'] = datetime.fromisoformat(teacher_data['updated_at'])
                    teachers.append(Teacher(**teacher_data))
            
            return sorted(teachers, key=lambda x: x.created_at, reverse=True)
        except Exception as e:
            logger.error(f"Failed to list teachers: {e}")
            return []
