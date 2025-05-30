import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    openai_api_key: Optional[str] = None
    jwt_secret_key: str = "your-secret-key"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    class Config:
        env_file = ".env"