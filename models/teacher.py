from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum
import uuid
from utils.helpers import serialize_datetime, deserialize_datetime

class TeachingStyle(str, Enum):
    SOCRATIC = "socratic"  # Asks guiding questions
    EXPLANATORY = "explanatory"  # Provides detailed explanations
    PRACTICAL = "practical"  # Focus on hands-on examples
    THEORETICAL = "theoretical"  # Emphasizes concepts and theory
    ADAPTIVE = "adaptive"  # Adjusts based on student needs

class PersonalityTrait(str, Enum):
    ENCOURAGING = "encouraging"
    PATIENT = "patient"
    CHALLENGING = "challenging"
    HUMOROUS = "humorous"
    FORMAL = "formal"
    CASUAL = "casual"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"

class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class TeacherPersonality(BaseModel):
    """Enhanced personality configuration for AI teachers"""
    
    # Core personality traits
    primary_traits: List[PersonalityTrait] = Field(
        ..., 
        min_items=1, 
        max_items=3,
        description="Primary personality traits that define the teacher"
    )
    
    # Teaching methodology
    teaching_style: TeachingStyle = Field(
        ...,
        description="Primary teaching approach"
    )
    
    # Communication preferences
    formality_level: Literal["very_formal", "formal", "neutral", "casual", "very_casual"] = Field(
        default="neutral",
        description="How formal the teacher's language should be"
    )
    
    # Interaction patterns
    question_frequency: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="How often the teacher asks questions"
    )
    
    encouragement_level: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="How much positive reinforcement to provide"
    )
    
    # Response characteristics
    response_length: Literal["concise", "moderate", "detailed"] = Field(
        default="moderate",
        description="Typical length of responses"
    )
    
    use_examples: bool = Field(
        default=True,
        description="Whether to frequently use examples"
    )
    
    use_analogies: bool = Field(
        default=True,
        description="Whether to use analogies and metaphors"
    )
    
    # Behavioral patterns
    patience_level: Literal["very_high", "high", "medium", "low"] = Field(
        default="high",
        description="How patient with repeated questions"
    )
    
    humor_usage: Literal["frequent", "occasional", "rare", "never"] = Field(
        default="occasional",
        description="How often to use humor"
    )
    
    # Custom quirks or catchphrases
    signature_phrases: List[str] = Field(
        default_factory=list,
        max_items=5,
        description="Unique phrases this teacher uses"
    )
    
    # Emotional intelligence
    empathy_level: Literal["very_high", "high", "medium", "low"] = Field(
        default="high",
        description="How empathetic the teacher is"
    )

    class Config:
        json_encoders = {
            PersonalityTrait: lambda v: v.value,
            TeachingStyle: lambda v: v.value
        }

class TeacherSpecialization(BaseModel):
    """Specific domain expertise and teaching capabilities"""
    
    # Primary domain
    primary_domain: str = Field(
        ...,
        description="Main area of expertise"
    )
    
    # Sub-domains
    specializations: List[str] = Field(
        default_factory=list,
        max_items=5,
        description="Specific topics within the domain"
    )
    
    # Difficulty range
    min_difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.BEGINNER,
        description="Minimum difficulty level this teacher handles"
    )
    
    max_difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.EXPERT,
        description="Maximum difficulty level this teacher handles"
    )
    
    # Teaching capabilities
    can_create_exercises: bool = Field(
        default=True,
        description="Can generate practice problems"
    )
    
    can_grade_work: bool = Field(
        default=True,
        description="Can evaluate student submissions"
    )
    
    can_create_curriculum: bool = Field(
        default=False,
        description="Can design learning paths"
    )
    enable_rag: bool = Field(
        default=False,
        description="Whether to use RAG for this teacher's responses"
    )
    
    # Knowledge base
    knowledge_cutoff: Optional[datetime] = Field(
        default=None,
        description="Latest date of knowledge"
    )
    
    external_resources: List[str] = Field(
        default_factory=list,
        description="Recommended external resources"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: serialize_datetime(v) if v else None,
            DifficultyLevel: lambda v: v.value
        }

class StudentAdaptation(BaseModel):
    """How the teacher adapts to different students"""
    
    # Learning style adaptation
    adapts_to_learning_style: bool = Field(
        default=True,
        description="Adjusts teaching based on student's learning style"
    )
    
    # Pace adjustment
    pace_adjustment: bool = Field(
        default=True,
        description="Adjusts teaching pace based on student progress"
    )
    
    # Difficulty scaling
    difficulty_scaling: bool = Field(
        default=True,
        description="Automatically adjusts difficulty"
    )
    
    # Personalization
    remembers_context: bool = Field(
        default=True,
        description="Remembers previous interactions"
    )
    
    tracks_progress: bool = Field(
        default=True,
        description="Monitors student progress over time"
    )

class EnhancedTeacherCreate(BaseModel):
    """Model for creating a new enhanced teacher"""
    
    # Custom ID field (optional)
    id: Optional[str] = Field(
        None,
        description="Custom teacher ID (if not provided, a UUID will be generated)"
    )
    
    # Basic information
    name: str = Field(..., min_length=1, max_length=100)
    title: Optional[str] = Field(
        None, 
        description="Professional title (e.g., 'Professor', 'Dr.')"
    )
    avatar_url: Optional[str] = Field(
        None,
        description="URL to teacher's avatar image"
    )
    
    # Personality configuration
    personality: TeacherPersonality
    
    # Domain expertise
    specialization: TeacherSpecialization
    
    # Student interaction
    adaptation: StudentAdaptation
    
    # Enhanced system prompt
    system_prompt_template: str = Field(
        ...,
        description="Template for generating context-aware prompts"
    )
    
    # Created by
    created_by: Optional[str] = None
    
    @validator('system_prompt_template')
    def validate_prompt_template(cls, v):
        """Ensure prompt template has required placeholders"""
        required_placeholders = ['{personality}', '{domain}', '{style}']
        for placeholder in required_placeholders:
            if placeholder not in v:
                raise ValueError(f"System prompt must contain {placeholder}")
        return v
    
    @validator('id')
    def validate_id(cls, v):
        """Validate custom ID if provided"""
        if v is not None and not v.strip():
            raise ValueError("ID cannot be empty string if provided")
        return v

class EnhancedTeacherUpdate(BaseModel):
    """Model for updating an existing enhanced teacher"""
    
    # Basic information
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    title: Optional[str] = Field(
        None, 
        description="Professional title (e.g., 'Professor', 'Dr.')"
    )
    avatar_url: Optional[str] = Field(
        None,
        description="URL to teacher's avatar image"
    )
    
    # Personality configuration
    personality: Optional[TeacherPersonality] = None
    
    # Domain expertise
    specialization: Optional[TeacherSpecialization] = None
    
    # Student interaction
    adaptation: Optional[StudentAdaptation] = None
    
    # Enhanced system prompt
    system_prompt_template: Optional[str] = Field(
        None,
        description="Template for generating context-aware prompts"
    )
    
    # Active status
    is_active: Optional[bool] = None
    
    @validator('system_prompt_template')
    def validate_prompt_template(cls, v):
        """Ensure prompt template has required placeholders"""
        if v is not None:
            required_placeholders = ['{personality}', '{domain}', '{style}']
            for placeholder in required_placeholders:
                if placeholder not in v:
                    raise ValueError(f"System prompt must contain {placeholder}")
        return v

class EnhancedTeacher(BaseModel):
    """Enhanced teacher model with rich personality system"""
    
    # Modified to use provided ID or generate one
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Basic information
    name: str = Field(..., min_length=1, max_length=100)
    title: Optional[str] = Field(
        None, 
        description="Professional title (e.g., 'Professor', 'Dr.')"
    )
    avatar_url: Optional[str] = Field(
        None,
        description="URL to teacher's avatar image"
    )
    
    # Personality configuration
    personality: TeacherPersonality
    
    # Domain expertise
    specialization: TeacherSpecialization
    
    # Student interaction
    adaptation: StudentAdaptation
    
    # Enhanced system prompt
    system_prompt_template: str = Field(
        ...,
        description="Template for generating context-aware prompts"
    )
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    is_active: bool = Field(default=True)
    
    # Statistics
    total_sessions: int = Field(default=0)
    average_rating: Optional[float] = Field(None, ge=0, le=5)
    
    class Config:
        json_encoders = {
            datetime: lambda v: serialize_datetime(v),
            PersonalityTrait: lambda v: v.value,
            TeachingStyle: lambda v: v.value,
            DifficultyLevel: lambda v: v.value
        }
    
    def generate_system_prompt(self, context: Dict[str, Any] = None) -> str:
        context = context or {}

        personality_desc = (
            f"You have these personality traits: {', '.join([t.value for t in self.personality.primary_traits])}. "
            f"Your teaching style is {self.personality.teaching_style.value}. "
            f"You communicate in a {self.personality.formality_level} manner. "
        )
        if self.personality.signature_phrases:
            personality_desc += f"You occasionally use these phrases: {', '.join(self.personality.signature_phrases)}. "

        domain_desc = f"You are an expert in {self.specialization.primary_domain}"
        if self.specialization.specializations:
            domain_desc += f", specializing in {', '.join(self.specialization.specializations)}"
        domain_desc += (
            f". You can teach from {self.specialization.min_difficulty.value} "
            f"to {self.specialization.max_difficulty.value} level. "
        )

        style_desc = (
            f"You {('frequently' if self.personality.use_examples else 'rarely')} use examples. "
            f"You {('often' if self.personality.use_analogies else 'rarely')} use analogies. "
            f"Your responses are typically {self.personality.response_length}. "
            f"You ask questions {self.personality.question_frequency}ly. "
        )

        # Reserved template fields you control
        mapping = {
            "personality": personality_desc,
            "domain": domain_desc,          # <-- this is the block text for the template
            "style": style_desc,            # <-- likewise
            "teacher_name": self.name,
            "title": self.title or "",
            "context": context,             # OK to pass the dict directly if template prints it
            "specialization": self.specialization.primary_domain,
            "specializations": ", ".join(self.specialization.specializations)
                if self.specialization.specializations else ""
        }

        # Drop any context keys that would collide with your reserved names
        sanitized_context = {k: v for k, v in context.items() if k not in mapping}

        prompt = self.system_prompt_template.format(**mapping, **sanitized_context)
        return prompt


    
    def get_personality_vector(self) -> Dict[str, float]:
        """Convert personality to numerical vector for ML models"""
        vector = {
            "formality": {"very_formal": 1.0, "formal": 0.75, "neutral": 0.5, "casual": 0.25, "very_casual": 0.0}[self.personality.formality_level],
            "question_frequency": {"high": 1.0, "medium": 0.5, "low": 0.0}[self.personality.question_frequency],
            "encouragement": {"high": 1.0, "medium": 0.5, "low": 0.0}[self.personality.encouragement_level],
            "patience": {"very_high": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}[self.personality.patience_level],
            "empathy": {"very_high": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}[self.personality.empathy_level],
            "humor": {"frequent": 1.0, "occasional": 0.5, "rare": 0.25, "never": 0.0}[self.personality.humor_usage],
        }
        
        # Add binary traits
        for trait in PersonalityTrait:
            vector[f"trait_{trait.value}"] = 1.0 if trait in self.personality.primary_traits else 0.0
        
        return vector

# Factory functions to create example teachers with custom IDs
def create_math_professor(custom_id: str = None):
    """Create a formal mathematics professor with optional custom ID"""
    return EnhancedTeacher(
        id=custom_id or str(uuid.uuid4()),
        name="Dr. Elizabeth Chen",
        title="Professor",
        personality=TeacherPersonality(
            primary_traits=[PersonalityTrait.ANALYTICAL, PersonalityTrait.PATIENT, PersonalityTrait.FORMAL],
            teaching_style=TeachingStyle.THEORETICAL,
            formality_level="formal",
            question_frequency="high",
            encouragement_level="medium",
            response_length="detailed",
            use_examples=True,
            use_analogies=True,
            patience_level="very_high",
            humor_usage="rare",
            signature_phrases=[
                "Let's explore this step by step",
                "An excellent question!",
                "Consider the following..."
            ],
            empathy_level="high"
        ),
        specialization=TeacherSpecialization(
            primary_domain="Mathematics",
            specializations=["Calculus", "Linear Algebra", "Statistics", "Number Theory"],
            min_difficulty=DifficultyLevel.BEGINNER,
            max_difficulty=DifficultyLevel.EXPERT,
            can_create_exercises=True,
            can_grade_work=True,
            can_create_curriculum=True
        ),
        adaptation=StudentAdaptation(
            adapts_to_learning_style=True,
            pace_adjustment=True,
            difficulty_scaling=True,
            remembers_context=True,
            tracks_progress=True
        ),
        system_prompt_template="""
        You are {teacher_name}, {title} of Mathematics.
        
        {personality}
        {domain}
        {style}
        
        Always maintain mathematical rigor while being accessible. Use proper mathematical notation when appropriate.
        When a student struggles, break down concepts into smaller, manageable pieces.
        
        Context: {context}
        """
    )

def create_coding_mentor(custom_id: str = None):
    """Create a casual coding mentor with optional custom ID"""
    return EnhancedTeacher(
        id=custom_id or str(uuid.uuid4()),
        name="Alex Rivera",
        title=None,
        personality=TeacherPersonality(
            primary_traits=[PersonalityTrait.ENCOURAGING, PersonalityTrait.CASUAL, PersonalityTrait.CREATIVE],
            teaching_style=TeachingStyle.PRACTICAL,
            formality_level="casual",
            question_frequency="medium",
            encouragement_level="high",
            response_length="moderate",
            use_examples=True,
            use_analogies=True,
            patience_level="high",
            humor_usage="frequent",
            signature_phrases=[
                "Let's code this up!",
                "Don't worry, we've all been there",
                "Here's a cool trick...",
                "You're getting the hang of it!"
            ],
            empathy_level="very_high"
        ),
        specialization=TeacherSpecialization(
            primary_domain="Programming",
            specializations=["Python", "JavaScript", "Web Development", "Data Structures"],
            min_difficulty=DifficultyLevel.BEGINNER,
            max_difficulty=DifficultyLevel.ADVANCED,
            can_create_exercises=True,
            can_grade_work=True,
            can_create_curriculum=False
        ),
        adaptation=StudentAdaptation(
            adapts_to_learning_style=True,
            pace_adjustment=True,
            difficulty_scaling=True,
            remembers_context=True,
            tracks_progress=True
        ),
        system_prompt_template="""
        Hey! I'm {teacher_name}, your coding buddy.
        
        {personality}
        {domain}
        {style}
        
        I believe in learning by doing. We'll write lots of code together, make mistakes, fix them, and have fun along the way.
        If something doesn't work, that's totally normal - debugging is half the job!
        
        Context: {context}
        """
    )