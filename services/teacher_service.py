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

class EnhancedTeacherService:
    CACHE_TTL = 3600  
    
    @staticmethod
    def create_teacher(teacher_data: EnhancedTeacherCreate) -> Optional[EnhancedTeacher]:
        """
        Create a new enhanced teacher with rich personality model
        """
        try:
            teacher_id = generate_id()
            
            teacher = EnhancedTeacher(
                id=teacher_id,
                name=teacher_data.name,
                title=teacher_data.title,
                avatar_url=teacher_data.avatar_url,
                personality=teacher_data.personality,
                specialization=teacher_data.specialization,
                adaptation=teacher_data.adaptation,
                system_prompt_template=teacher_data.system_prompt_template,
                created_by=teacher_data.created_by,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Prepare data for Redis with proper serialization
            teacher_key = f"enhanced_teacher:{teacher.id}"
            teacher_dict = teacher.dict()
            
            # Manually handle datetime serialization
            teacher_dict['created_at'] = serialize_datetime(teacher.created_at)
            teacher_dict['updated_at'] = serialize_datetime(teacher.updated_at)
            
            # Handle knowledge_cutoff if it exists
            if teacher.specialization.knowledge_cutoff:
                teacher_dict['specialization']['knowledge_cutoff'] = serialize_datetime(teacher.specialization.knowledge_cutoff)
            
            # Save to Redis using json_set method
            if redis_client.json_set(teacher_key, teacher_dict):
                # Update indexes for quick lookups
                EnhancedTeacherService._update_teacher_index(teacher)
                
                # Invalidate cache
                EnhancedTeacherService._invalidate_teacher_list_cache()
                
                logger.info(f"Enhanced teacher created: {teacher.id}")
                return teacher
            return None
        except Exception as e:
            logger.error(f"Failed to create enhanced teacher: {e}", exc_info=True)
            return None
    
    @staticmethod
    def get_teacher(teacher_id: str) -> Optional[EnhancedTeacher]:
        """
        Get an enhanced teacher by ID with caching
        """
        try:
            # Try to get from cache first
            cache_key = f"cache:enhanced_teacher:{teacher_id}"
            cached_data = None
            
            if redis_client.exists(cache_key):
                # Since your redis client doesn't have a direct 'get' method,
                # we'll use json_get here as well
                cached_data = redis_client.json_get(cache_key)
            
            if cached_data:
                # Convert ISO strings back to datetime
                cached_data['created_at'] = deserialize_datetime(cached_data['created_at'])
                cached_data['updated_at'] = deserialize_datetime(cached_data['updated_at'])
                
                # Handle knowledge_cutoff if it exists
                if cached_data.get('specialization', {}).get('knowledge_cutoff'):
                    cached_data['specialization']['knowledge_cutoff'] = deserialize_datetime(
                        cached_data['specialization']['knowledge_cutoff']
                    )
                
                logger.debug(f"Cache hit for enhanced teacher: {teacher_id}")
                return EnhancedTeacher(**cached_data)
            
            # If not in cache, get from main storage
            teacher_key = f"enhanced_teacher:{teacher_id}"
            teacher_data = redis_client.json_get(teacher_key)
            
            if teacher_data:
                # Convert ISO strings back to datetime
                teacher_data['created_at'] = deserialize_datetime(teacher_data['created_at'])
                teacher_data['updated_at'] = deserialize_datetime(teacher_data['updated_at'])
                
                # Handle knowledge_cutoff if it exists
                if teacher_data.get('specialization', {}).get('knowledge_cutoff'):
                    teacher_data['specialization']['knowledge_cutoff'] = deserialize_datetime(
                        teacher_data['specialization']['knowledge_cutoff']
                    )
                
                # Cache the result using json_set
                redis_client.json_set(cache_key, teacher_data)
                
                return EnhancedTeacher(**teacher_data)
            return None
        except Exception as e:
            logger.error(f"Failed to get enhanced teacher {teacher_id}: {e}", exc_info=True)
            return None
    
    @staticmethod
    def update_teacher(teacher_id: str, update_data: EnhancedTeacherUpdate) -> Optional[EnhancedTeacher]:
        """
        Update an enhanced teacher with validation
        """
        try:
            teacher = EnhancedTeacherService.get_teacher(teacher_id)
            if not teacher:
                return None
            
            # Update fields if provided
            update_dict = update_data.dict(exclude_unset=True)
            for key, value in update_dict.items():
                setattr(teacher, key, value)
            
            # Update timestamp
            teacher.updated_at = datetime.utcnow()
            
            # Save back to Redis with proper serialization
            teacher_key = f"enhanced_teacher:{teacher_id}"
            teacher_dict = teacher.dict()
            
            # Manually handle datetime serialization
            teacher_dict['created_at'] = serialize_datetime(teacher.created_at)
            teacher_dict['updated_at'] = serialize_datetime(teacher.updated_at)
            
            # Handle knowledge_cutoff if it exists
            if teacher.specialization.knowledge_cutoff:
                teacher_dict['specialization']['knowledge_cutoff'] = serialize_datetime(teacher.specialization.knowledge_cutoff)
            
            if redis_client.json_set(teacher_key, teacher_dict):
                # Update indexes
                EnhancedTeacherService._update_teacher_index(teacher)
                
                # Invalidate caches
                EnhancedTeacherService._invalidate_teacher_cache(teacher_id)
                EnhancedTeacherService._invalidate_teacher_list_cache()
                
                logger.info(f"Enhanced teacher updated: {teacher_id}")
                return teacher
            return None
        except Exception as e:
            logger.error(f"Failed to update enhanced teacher {teacher_id}: {e}", exc_info=True)
            return None
    
    @staticmethod
    def delete_teacher(teacher_id: str) -> bool:
        """
        Delete an enhanced teacher and all associated data
        """
        try:
            teacher = EnhancedTeacherService.get_teacher(teacher_id)
            if not teacher:
                return False
                
            teacher_key = f"enhanced_teacher:{teacher_id}"
            
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
            
            # Clean up from indexes
            EnhancedTeacherService._remove_from_teacher_index(teacher)
            
            # Invalidate caches
            EnhancedTeacherService._invalidate_teacher_cache(teacher_id)
            EnhancedTeacherService._invalidate_teacher_list_cache()
            
            logger.info(f"Enhanced teacher deleted: {teacher_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete enhanced teacher {teacher_id}: {e}", exc_info=True)
            return False
    
    @staticmethod
    def list_teachers() -> List[EnhancedTeacher]:
        """
        List all enhanced teachers with caching
        """
        try:
            # Try to get from cache first
            cache_key = "cache:enhanced_teacher:list"
            cached_data = None
            
            if redis_client.exists(cache_key):
                cached_data = redis_client.json_get(cache_key)
            
            if cached_data:
                teachers = []
                
                for teacher_data in cached_data:
                    # Convert ISO strings back to datetime
                    teacher_data['created_at'] = deserialize_datetime(teacher_data['created_at'])
                    teacher_data['updated_at'] = deserialize_datetime(teacher_data['updated_at'])
                    
                    # Handle knowledge_cutoff if it exists
                    if teacher_data.get('specialization', {}).get('knowledge_cutoff'):
                        teacher_data['specialization']['knowledge_cutoff'] = deserialize_datetime(
                            teacher_data['specialization']['knowledge_cutoff']
                        )
                    
                    teachers.append(EnhancedTeacher(**teacher_data))
                
                logger.debug("Cache hit for enhanced teacher list")
                return teachers
            
            # If not in cache, get from main storage
            teacher_keys = redis_client.scan_keys("enhanced_teacher:*")
            teachers = []
            
            for key in teacher_keys:
                teacher_data = redis_client.json_get(key)
                if teacher_data:
                    # Convert ISO strings back to datetime
                    teacher_data['created_at'] = deserialize_datetime(teacher_data['created_at'])
                    teacher_data['updated_at'] = deserialize_datetime(teacher_data['updated_at'])
                    
                    # Handle knowledge_cutoff if it exists
                    if teacher_data.get('specialization', {}).get('knowledge_cutoff'):
                        teacher_data['specialization']['knowledge_cutoff'] = deserialize_datetime(
                            teacher_data['specialization']['knowledge_cutoff']
                        )
                    
                    teachers.append(EnhancedTeacher(**teacher_data))
            
            # Sort by created_at by default
            teachers = sorted(teachers, key=lambda x: x.created_at, reverse=True)
            
            # Cache the result with proper serialization
            teachers_dict = []
            for t in teachers:
                t_dict = t.dict()
                t_dict['created_at'] = serialize_datetime(t.created_at)
                t_dict['updated_at'] = serialize_datetime(t.updated_at)
                
                if t.specialization.knowledge_cutoff:
                    t_dict['specialization']['knowledge_cutoff'] = serialize_datetime(t.specialization.knowledge_cutoff)
                
                teachers_dict.append(t_dict)
                
            # Cache using json_set
            redis_client.json_set(cache_key, teachers_dict)
            
            return teachers
        except Exception as e:
            logger.error(f"Failed to list enhanced teachers: {e}", exc_info=True)
            return []
    
    @staticmethod
    def search_teachers(
        domain: Optional[str] = None,
        teaching_style: Optional[TeachingStyle] = None,
        difficulty_level: Optional[DifficultyLevel] = None,
        traits: Optional[List[PersonalityTrait]] = None,
        query: Optional[str] = None,
        page: int = 1,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Search enhanced teachers with advanced filtering
        """
        try:
            all_teachers = EnhancedTeacherService.list_teachers()
            filtered_teachers = all_teachers
            
            # Filter by domain if specified
            if domain:
                filtered_teachers = [
                    t for t in filtered_teachers 
                    if t.specialization.primary_domain.lower() == domain.lower() or
                    domain.lower() in [s.lower() for s in t.specialization.specializations]
                ]
            
            # Filter by teaching style if specified
            if teaching_style:
                filtered_teachers = [
                    t for t in filtered_teachers 
                    if t.personality.teaching_style == teaching_style
                ]
            
            # Filter by difficulty level if specified
            if difficulty_level:
                filtered_teachers = [
                    t for t in filtered_teachers 
                    if (t.specialization.min_difficulty.value <= difficulty_level.value and
                        t.specialization.max_difficulty.value >= difficulty_level.value)
                ]
            
            # Filter by personality traits if specified
            if traits and len(traits) > 0:
                filtered_teachers = [
                    t for t in filtered_teachers 
                    if any(trait in t.personality.primary_traits for trait in traits)
                ]
            
            # Filter by query if specified
            if query:
                query = query.lower()
                filtered_teachers = [
                    t for t in filtered_teachers 
                    if (query in t.name.lower() or 
                        query in t.specialization.primary_domain.lower() or
                        any(query in s.lower() for s in t.specialization.specializations) or
                        (t.title and query in t.title.lower()))
                ]
            
            # Calculate pagination
            total = len(filtered_teachers)
            total_pages = (total + limit - 1) // limit  # Ceiling division
            offset = (page - 1) * limit
            
            # Apply pagination
            paginated_teachers = filtered_teachers[offset:offset + limit]
            
            return {
                "teachers": paginated_teachers,
                "pagination": {
                    "total": total,
                    "page": page,
                    "limit": limit,
                    "total_pages": total_pages
                }
            }
        except Exception as e:
            logger.error(f"Failed to search enhanced teachers: {e}", exc_info=True)
            return {"teachers": [], "pagination": {"total": 0, "page": page, "limit": limit, "total_pages": 0}}
    
    @staticmethod
    def get_teacher_by_domain(domain: str) -> Optional[EnhancedTeacher]:
        """
        Get an enhanced teacher by domain expertise
        """
        try:
            # Use the domain index for faster lookup
            domain_key = f"index:enhanced_teacher:domain:{domain.lower()}"
            
            # We'll need to use json_get since your client doesn't have a get method
            teacher_id = None
            if redis_client.exists(domain_key):
                teacher_data = redis_client.json_get(domain_key)
                if teacher_data:
                    # The teacher ID should be stored directly in the value
                    teacher_id = teacher_data
            
            if teacher_id:
                return EnhancedTeacherService.get_teacher(teacher_id)
            
            # Fallback to scanning all teachers
            all_teachers = EnhancedTeacherService.list_teachers()
            for teacher in all_teachers:
                if (teacher.specialization.primary_domain.lower() == domain.lower() or
                    domain.lower() in [s.lower() for s in teacher.specialization.specializations]):
                    # Update the index for future lookups - store teacher ID directly
                    redis_client.json_set(domain_key, teacher.id)
                    return teacher
            
            return None
        except Exception as e:
            logger.error(f"Failed to get enhanced teacher by domain {domain}: {e}", exc_info=True)
            return None
    
    @staticmethod
    def increment_session_count(teacher_id: str) -> bool:
        """
        Increment the total session count for a teacher
        """
        try:
            teacher = EnhancedTeacherService.get_teacher(teacher_id)
            if not teacher:
                return False
            
            # Increment session count
            teacher.total_sessions += 1
            
            # Save back to Redis with proper serialization
            teacher_key = f"enhanced_teacher:{teacher_id}"
            teacher_dict = teacher.dict()
            
            # Manually handle datetime serialization
            teacher_dict['created_at'] = serialize_datetime(teacher.created_at)
            teacher_dict['updated_at'] = serialize_datetime(teacher.updated_at)
            
            # Handle knowledge_cutoff if it exists
            if teacher.specialization.knowledge_cutoff:
                teacher_dict['specialization']['knowledge_cutoff'] = serialize_datetime(teacher.specialization.knowledge_cutoff)
            
            if redis_client.json_set(teacher_key, teacher_dict):
                # Invalidate caches
                EnhancedTeacherService._invalidate_teacher_cache(teacher_id)
                
                logger.info(f"Enhanced teacher session count incremented: {teacher_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to increment session count for enhanced teacher {teacher_id}: {e}", exc_info=True)
            return False
    
    @staticmethod
    def add_teacher_rating(teacher_id: str, rating: float) -> bool:
        """
        Add a rating for a teacher and update the average
        """
        try:
            teacher = EnhancedTeacherService.get_teacher(teacher_id)
            if not teacher:
                return False
            
            # Get current ratings
            ratings_key = f"ratings:enhanced_teacher:{teacher_id}"
            ratings = redis_client.list_get(ratings_key)
            ratings = [float(r) for r in ratings] if ratings else []
            
            # Add new rating
            redis_client.list_push(ratings_key, str(rating))
            
            # Calculate new average
            new_ratings = ratings + [rating]
            new_average = sum(new_ratings) / len(new_ratings)
            
            # Update teacher
            teacher.average_rating = new_average
            
            # Save back to Redis with proper serialization
            teacher_key = f"enhanced_teacher:{teacher_id}"
            teacher_dict = teacher.dict()
            
            # Manually handle datetime serialization
            teacher_dict['created_at'] = serialize_datetime(teacher.created_at)
            teacher_dict['updated_at'] = serialize_datetime(teacher.updated_at)
            
            # Handle knowledge_cutoff if it exists
            if teacher.specialization.knowledge_cutoff:
                teacher_dict['specialization']['knowledge_cutoff'] = serialize_datetime(teacher.specialization.knowledge_cutoff)
            
            if redis_client.json_set(teacher_key, teacher_dict):
                # Invalidate caches
                EnhancedTeacherService._invalidate_teacher_cache(teacher_id)
                
                logger.info(f"Enhanced teacher rating added: {teacher_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to add rating for enhanced teacher {teacher_id}: {e}", exc_info=True)
            return False
    
    @staticmethod
    def generate_system_prompt(teacher_id: str, context: Dict[str, Any] = None) -> Optional[str]:
        """
        Generate a context-aware system prompt for a teacher
        """
        try:
            teacher = EnhancedTeacherService.get_teacher(teacher_id)
            if not teacher:
                return None
            
            return teacher.generate_system_prompt(context)
        except Exception as e:
            logger.error(f"Failed to generate system prompt for enhanced teacher {teacher_id}: {e}", exc_info=True)
            return None
    
    @staticmethod
    def create_default_teachers() -> Dict[str, Any]:
        """
        Create default teacher profiles if none exist
        """
        try:
            existing_teachers = EnhancedTeacherService.list_teachers()
            if existing_teachers:
                return {"message": "Default teachers already exist", "count": len(existing_teachers)}
            
            # Create math professor
            math_professor = create_math_professor()
            math_teacher_data = EnhancedTeacherCreate(
                name=math_professor.name,
                title=math_professor.title,
                personality=math_professor.personality,
                specialization=math_professor.specialization,
                adaptation=math_professor.adaptation,
                system_prompt_template=math_professor.system_prompt_template
            )
            
            # Create coding mentor
            coding_mentor = create_coding_mentor()
            coding_teacher_data = EnhancedTeacherCreate(
                name=coding_mentor.name,
                title=coding_mentor.title,
                personality=coding_mentor.personality,
                specialization=coding_mentor.specialization,
                adaptation=coding_mentor.adaptation,
                system_prompt_template=coding_mentor.system_prompt_template
            )
            
            # Create the teachers
            math_teacher = EnhancedTeacherService.create_teacher(math_teacher_data)
            coding_teacher = EnhancedTeacherService.create_teacher(coding_teacher_data)
            
            created = []
            if math_teacher:
                created.append(math_teacher)
            if coding_teacher:
                created.append(coding_teacher)
            
            return {"message": "Default teachers created", "teachers": created, "count": len(created)}
        except Exception as e:
            logger.error(f"Failed to create default enhanced teachers: {e}", exc_info=True)
            return {"message": "Failed to create default teachers", "error": str(e)}
    
    @staticmethod
    def get_all_teaching_styles() -> List[Dict[str, str]]:
        """
        Get all available teaching styles
        """
        return [{"value": style.value, "label": style.name.title()} for style in TeachingStyle]
    
    @staticmethod
    def get_all_personality_traits() -> List[Dict[str, str]]:
        """
        Get all available personality traits
        """
        return [{"value": trait.value, "label": trait.name.title()} for trait in PersonalityTrait]
    
    @staticmethod
    def get_all_difficulty_levels() -> List[Dict[str, str]]:
        """
        Get all available difficulty levels
        """
        return [{"value": level.value, "label": level.name.title()} for level in DifficultyLevel]
    
    # Private helper methods
    
    @staticmethod
    def _update_teacher_index(teacher: EnhancedTeacher) -> None:
        """
        Update indexes for faster lookups
        """
        try:
            # Primary domain index - store just the ID value
            domain_key = f"index:enhanced_teacher:domain:{teacher.specialization.primary_domain.lower()}"
            redis_client.json_set(domain_key, teacher.id)
            
            # Specialization indexes
            for specialization in teacher.specialization.specializations:
                spec_key = f"index:enhanced_teacher:specialization:{specialization.lower()}"
                redis_client.json_set(spec_key, teacher.id)
            
            # Teaching style index - use list push instead of sadd
            style_key = f"index:enhanced_teacher:style:{teacher.personality.teaching_style.value}"
            if not redis_client.exists(style_key):
                redis_client.json_set(style_key, [teacher.id])
            else:
                # Get existing IDs
                existing_ids = redis_client.json_get(style_key) or []
                if teacher.id not in existing_ids:
                    existing_ids.append(teacher.id)
                    redis_client.json_set(style_key, existing_ids)
            
            # Personality trait indexes
            for trait in teacher.personality.primary_traits:
                trait_key = f"index:enhanced_teacher:trait:{trait.value}"
                if not redis_client.exists(trait_key):
                    redis_client.json_set(trait_key, [teacher.id])
                else:
                    # Get existing IDs
                    existing_ids = redis_client.json_get(trait_key) or []
                    if teacher.id not in existing_ids:
                        existing_ids.append(teacher.id)
                        redis_client.json_set(trait_key, existing_ids)
        except Exception as e:
            logger.error(f"Failed to update teacher index: {e}", exc_info=True)
    
    @staticmethod
    def _remove_from_teacher_index(teacher: EnhancedTeacher) -> None:
        """
        Remove teacher from indexes
        """
        try:
            # Primary domain index
            domain_key = f"index:enhanced_teacher:domain:{teacher.specialization.primary_domain.lower()}"
            redis_client.delete(domain_key)
            
            # Specialization indexes
            for specialization in teacher.specialization.specializations:
                spec_key = f"index:enhanced_teacher:specialization:{specialization.lower()}"
                redis_client.delete(spec_key)
            
            # Teaching style index
            style_key = f"index:enhanced_teacher:style:{teacher.personality.teaching_style.value}"
            if redis_client.exists(style_key):
                existing_ids = redis_client.json_get(style_key) or []
                if teacher.id in existing_ids:
                    existing_ids.remove(teacher.id)
                    if existing_ids:
                        redis_client.json_set(style_key, existing_ids)
                    else:
                        redis_client.delete(style_key)
            
            # Personality trait indexes
            for trait in teacher.personality.primary_traits:
                trait_key = f"index:enhanced_teacher:trait:{trait.value}"
                if redis_client.exists(trait_key):
                    existing_ids = redis_client.json_get(trait_key) or []
                    if teacher.id in existing_ids:
                        existing_ids.remove(teacher.id)
                        if existing_ids:
                            redis_client.json_set(trait_key, existing_ids)
                        else:
                            redis_client.delete(trait_key)
        except Exception as e:
            logger.error(f"Failed to remove teacher from index: {e}", exc_info=True)
    
    @staticmethod
    def _invalidate_teacher_cache(teacher_id: str) -> None:
        """
        Invalidate teacher cache
        """
        cache_key = f"cache:enhanced_teacher:{teacher_id}"
        redis_client.delete(cache_key)
    
    @staticmethod
    def _invalidate_teacher_list_cache() -> None:
        """
        Invalidate teacher list cache
        """
        cache_key = "cache:enhanced_teacher:list"
        redis_client.delete(cache_key)