from typing import List, Optional, Dict, Any, Union
from models.teacher import (
    EnhancedTeacher, 
    EnhancedTeacherCreate, 
    EnhancedTeacherUpdate,
    TeachingStyle,
    PersonalityTrait,
    DifficultyLevel,
    create_math_professor,
    create_coding_mentor
)
from services.redis_client import redis_client
from core.logger import logger
from utils.helpers import serialize_datetime, deserialize_datetime, safe_json_dumps, safe_json_loads, generate_id
import json
from datetime import datetime
import uuid

class EnhancedTeacherService:
    
    @staticmethod
    async def get_teacher(teacher_id: str) -> Optional[EnhancedTeacher]:
        """Get a teacher by ID with async support"""
        try:
            # Check cache first (if implemented)
            cache_key = f"teacher:{teacher_id}"
            
            # Get from Redis
            cached_data = await redis_client.json_get(cache_key)  # Make sure to await this
            
            if cached_data:
                # Deserialize datetime fields
                cached_data['created_at'] = deserialize_datetime(cached_data['created_at'])
                cached_data['updated_at'] = deserialize_datetime(cached_data['updated_at'])
                
                # Convert to EnhancedTeacher model
                return EnhancedTeacher(**cached_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get enhanced teacher {teacher_id}: {e}", exc_info=True)
            return None
    
    @staticmethod
    async def create_teacher(teacher_data: EnhancedTeacherCreate) -> Optional[EnhancedTeacher]:
        """Create a new teacher with async support and custom ID support"""
        try:
            # Create new teacher instance with custom ID if provided
            teacher_dict = teacher_data.dict(exclude={'created_by'})
            
            # If custom ID is provided, use it, otherwise generate UUID
            teacher_id = teacher_dict.pop('id') if teacher_dict.get('id') else str(uuid.uuid4())
            
            teacher = EnhancedTeacher(
                id=teacher_id,
                created_by=teacher_data.created_by,
                **teacher_dict
            )
            
            # Serialize and save to Redis
            teacher_key = f"teacher:{teacher.id}"
            teacher_dict = teacher.dict()
            teacher_dict['created_at'] = serialize_datetime(teacher.created_at)
            teacher_dict['updated_at'] = serialize_datetime(teacher.updated_at)
            
            # Check if teacher with this ID already exists
            existing_teacher = await redis_client.json_get(teacher_key)
            if existing_teacher:
                logger.error(f"Teacher with ID {teacher.id} already exists")
                return None
            
            success = await redis_client.json_set(teacher_key, teacher_dict)
            
            if not success:
                logger.error(f"Failed to save teacher to Redis: {teacher.id}")
                return None
            
            # Index by domain for domain-based retrieval
            domain_key = f"domain:{teacher.specialization.primary_domain}"
            await redis_client.list_push(domain_key, teacher.id)
            
            return teacher
            
        except Exception as e:
            logger.error(f"Failed to create teacher: {e}", exc_info=True)
            return None
    
    @staticmethod
    async def update_teacher(teacher_id: str, update_data: EnhancedTeacherUpdate) -> Optional[EnhancedTeacher]:
        """Update a teacher with async support"""
        try:
            # Get current teacher
            teacher = await EnhancedTeacherService.get_teacher(teacher_id)
            
            if not teacher:
                logger.error(f"Teacher not found for update: {teacher_id}")
                return None
            
            # Update fields from update_data
            update_dict = update_data.dict(exclude_unset=True)
            
            for field, value in update_dict.items():
                setattr(teacher, field, value)
            
            # Update timestamp
            teacher.updated_at = datetime.utcnow()
            
            # Save to Redis
            teacher_key = f"teacher:{teacher.id}"
            teacher_dict = teacher.dict()
            teacher_dict['created_at'] = serialize_datetime(teacher.created_at)
            teacher_dict['updated_at'] = serialize_datetime(teacher.updated_at)
            
            success = await redis_client.json_set(teacher_key, teacher_dict)
            
            if not success:
                logger.error(f"Failed to update teacher in Redis: {teacher.id}")
                return None
            
            return teacher
            
        except Exception as e:
            logger.error(f"Failed to update teacher: {e}", exc_info=True)
            return None
    
    @staticmethod
    async def delete_teacher(teacher_id: str) -> bool:
        """Delete a teacher with async support"""
        try:
            # Get teacher to check if it exists
            teacher = await EnhancedTeacherService.get_teacher(teacher_id)
            
            if not teacher:
                logger.error(f"Teacher not found for deletion: {teacher_id}")
                return False
            
            # Delete from Redis
            teacher_key = f"teacher:{teacher_id}"
            success = await redis_client.delete(teacher_key)
            
            if not success:
                logger.error(f"Failed to delete teacher from Redis: {teacher_id}")
                return False
            
            # Remove from domain index
            domain_key = f"domain:{teacher.specialization.primary_domain}"
            # Note: This is a simplified approach - in production you'd need to 
            # actually find and remove just this ID from the list
            await redis_client.delete(domain_key)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete teacher: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def list_teachers() -> List[EnhancedTeacher]:
        """List all teachers with async support"""
        try:
            # Scan for teacher keys
            pattern = "teacher:*"
            teacher_keys = await redis_client.scan_keys(pattern)
            
            teachers = []
            for key in teacher_keys:
                teacher_data = await redis_client.json_get(key)
                
                if teacher_data:
                    # Deserialize datetime fields
                    teacher_data['created_at'] = deserialize_datetime(teacher_data['created_at'])
                    teacher_data['updated_at'] = deserialize_datetime(teacher_data['updated_at'])
                    
                    # Convert to EnhancedTeacher model
                    teachers.append(EnhancedTeacher(**teacher_data))
            
            return teachers
            
        except Exception as e:
            logger.error(f"Failed to list teachers: {e}", exc_info=True)
            return []
    
    @staticmethod
    async def get_teacher_by_domain(domain: str) -> Optional[EnhancedTeacher]:
        """Get a teacher by domain with async support"""
        try:
            # Get from domain index
            domain_key = f"domain:{domain}"
            teacher_ids = await redis_client.list_get(domain_key)
            
            if not teacher_ids:
                return None
            
            # Get the first teacher in the list
            return await EnhancedTeacherService.get_teacher(teacher_ids[0])
            
        except Exception as e:
            logger.error(f"Failed to get teacher by domain {domain}: {e}", exc_info=True)
            return None
    
    @staticmethod
    async def increment_session_count(teacher_id: str) -> bool:
        """Increment a teacher's session count with async support"""
        try:
            # Get teacher
            teacher = await EnhancedTeacherService.get_teacher(teacher_id)
            
            if not teacher:
                logger.error(f"Teacher not found for session increment: {teacher_id}")
                return False
            
            # Increment session count
            teacher.total_sessions += 1
            
            # Save to Redis
            teacher_key = f"teacher:{teacher.id}"
            teacher_dict = teacher.dict()
            teacher_dict['created_at'] = serialize_datetime(teacher.created_at)
            teacher_dict['updated_at'] = serialize_datetime(teacher.updated_at)
            
            return await redis_client.json_set(teacher_key, teacher_dict)
            
        except Exception as e:
            logger.error(f"Failed to increment session count: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def add_teacher_rating(teacher_id: str, rating: float) -> bool:
        """Add a rating for a teacher with async support"""
        try:
            # Get teacher
            teacher = await EnhancedTeacherService.get_teacher(teacher_id)
            
            if not teacher:
                logger.error(f"Teacher not found for rating: {teacher_id}")
                return False
            
            # Store rating in a separate key for analytics
            rating_key = f"teacher:{teacher_id}:ratings"
            rating_data = {
                "rating": rating,
                "timestamp": serialize_datetime(datetime.utcnow())
            }
            
            # Add to ratings list
            await redis_client.list_push(rating_key, json.dumps(rating_data))
            
            # Update average rating
            ratings_list = await redis_client.list_get(rating_key)
            ratings = []
            
            for r_str in ratings_list:
                try:
                    r_data = json.loads(r_str)
                    ratings.append(r_data.get("rating", 0))
                except (json.JSONDecodeError, TypeError):
                    continue
            
            if ratings:
                teacher.average_rating = sum(ratings) / len(ratings)
            
            # Save to Redis
            teacher_key = f"teacher:{teacher.id}"
            teacher_dict = teacher.dict()
            teacher_dict['created_at'] = serialize_datetime(teacher.created_at)
            teacher_dict['updated_at'] = serialize_datetime(teacher.updated_at)
            
            return await redis_client.json_set(teacher_key, teacher_dict)
            
        except Exception as e:
            logger.error(f"Failed to add teacher rating: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def generate_system_prompt(teacher_id: str, context: Dict[str, Any] = None) -> Optional[str]:
        """Generate a system prompt for a teacher with async support"""
        try:
            # Get teacher
            teacher = await EnhancedTeacherService.get_teacher(teacher_id)
            
            if not teacher:
                logger.error(f"Teacher not found for prompt generation: {teacher_id}")
                return None
            
            # Generate prompt
            return teacher.generate_system_prompt(context or {})
            
        except Exception as e:
            logger.error(f"Failed to generate system prompt: {e}", exc_info=True)
            return None
    
    @staticmethod
    async def search_teachers(
        domain: Optional[str] = None,
        teaching_style: Optional[str] = None,
        difficulty_level: Optional[str] = None,
        traits: Optional[List[str]] = None,
        query: Optional[str] = None,
        page: int = 1,
        limit: int = 20
    ) -> Dict[str, Any]:
        """Search for teachers with async support"""
        try:
            # Get all teachers
            all_teachers = await EnhancedTeacherService.list_teachers()
            
            # Apply filters
            filtered_teachers = []
            
            for teacher in all_teachers:
                # Skip inactive teachers
                if not teacher.is_active:
                    continue
                
                # Apply domain filter
                if domain and domain.lower() not in teacher.specialization.primary_domain.lower():
                    if not any(domain.lower() in spec.lower() for spec in teacher.specialization.specializations):
                        continue
                
                # Apply teaching style filter
                if teaching_style and teacher.personality.teaching_style.value != teaching_style:
                    continue
                
                # Apply difficulty level filter
                if difficulty_level:
                    min_level = teacher.specialization.min_difficulty.value
                    max_level = teacher.specialization.max_difficulty.value
                    if not (min_level <= difficulty_level <= max_level):
                        continue
                
                # Apply traits filter
                if traits:
                    teacher_traits = [t.value for t in teacher.personality.primary_traits]
                    if not any(t in teacher_traits for t in traits):
                        continue
                
                # Apply text search
                if query:
                    query_lower = query.lower()
                    if (query_lower not in teacher.name.lower() and
                        query_lower not in teacher.specialization.primary_domain.lower() and
                        not any(query_lower in spec.lower() for spec in teacher.specialization.specializations)):
                        continue
                
                filtered_teachers.append(teacher)
            
            # Calculate pagination
            total = len(filtered_teachers)
            pages = (total + limit - 1) // limit if limit > 0 else 1
            
            start = (page - 1) * limit
            end = start + limit
            
            paginated_teachers = filtered_teachers[start:end]
            
            return {
                "teachers": paginated_teachers,
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": total,
                    "pages": pages
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to search teachers: {e}", exc_info=True)
            return {
                "teachers": [],
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": 0,
                    "pages": 0
                }
            }
    
    @staticmethod
    async def create_default_teachers() -> Dict[str, Any]:
        """Create default teacher profiles with async support and custom IDs"""
        try:
            # Check if teachers already exist
            existing_teachers = await EnhancedTeacherService.list_teachers()
            
            if existing_teachers:
                return {
                    "message": f"{len(existing_teachers)} teachers already exist, no defaults created",
                    "created": 0
                }
            
            # Create math professor with custom ID
            math_prof = create_math_professor("math_professor_001")
            await EnhancedTeacherService.create_teacher(
                EnhancedTeacherCreate(
                    id=math_prof.id,
                    **math_prof.dict(exclude={'id', 'created_at', 'updated_at', 'total_sessions', 'average_rating'})
                )
            )
            
            # Create coding mentor with custom ID
            coding_mentor = create_coding_mentor("coding_mentor_001")
            await EnhancedTeacherService.create_teacher(
                EnhancedTeacherCreate(
                    id=coding_mentor.id,
                    **coding_mentor.dict(exclude={'id', 'created_at', 'updated_at', 'total_sessions', 'average_rating'})
                )
            )
            
            return {
                "message": "Default teachers created successfully",
                "created": 2
            }
            
        except Exception as e:
            logger.error(f"Failed to create default teachers: {e}", exc_info=True)
            return {
                "message": f"Error creating default teachers: {str(e)}",
                "created": 0
            }
    
    @staticmethod
    def get_all_teaching_styles() -> List[Dict[str, str]]:
        """Get all available teaching styles"""
        return [{"value": style.value, "label": style.name.title()} for style in TeachingStyle]
    
    @staticmethod
    def get_all_personality_traits() -> List[Dict[str, str]]:
        """Get all available personality traits"""
        return [{"value": trait.value, "label": trait.name.title()} for trait in PersonalityTrait]
    
    @staticmethod
    def get_all_difficulty_levels() -> List[Dict[str, str]]:
        """Get all available difficulty levels"""
        return [{"value": level.value, "label": level.name.title()} for level in DifficultyLevel]