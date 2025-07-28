from typing import Optional, Dict, Any
import requests
from fastapi import HTTPException, Depends, Request
from core.config import Settings
from core.crpypt_utils import decrypt_payload
from core.logger import logger
from services.redis_client import redis_client
import json
from datetime import datetime, timedelta

settings = Settings()

class AuthService:
    """Authentication service for token validation and user identification"""
    _token_cache = {}
    
    @staticmethod
    async def validate_token(token: str) -> Dict[str, Any]:
        """
        Validate JWT token via introspect endpoint and return user data
        """
        # Check cache first to reduce API calls
        if token in AuthService._token_cache:
            cache_data = AuthService._token_cache[token]
            # Check if cache entry is still valid (not expired)
            if cache_data['expires_at'] > datetime.utcnow():
                return cache_data['user_data']
            AuthService._token_cache.pop(token)
        
        try:
            # Call introspect endpoint
            response = requests.post(
                "https://mordernera.com/api/v1/auth/introspect",
                json={"token": token},
                timeout=5  # Add timeout to prevent hanging requests
            )
            
            # Check for successful response
            if response.status_code != 200:
                logger.error(f"Token introspection failed: {response.status_code}, {response.text}")
                raise HTTPException(status_code=401, detail="Invalid or expired token")
            
            # Parse response
            data = response.json()
            
            # Extract user info from response
            if not data.get("data"):
                logger.error(f"Token introspection response missing data: {data}")
                raise HTTPException(status_code=401, detail="Invalid token format")
            
            user_info = decrypt_payload(data["data"])  # In production: decrypt_payload(data["data"])
            
            cache_expiry = datetime.utcnow() + timedelta(minutes=15)
            AuthService._token_cache[token] = {
                'user_data': user_info,
                'expires_at': cache_expiry
            }
            
            if len(AuthService._token_cache) > 1000:
                sorted_keys = sorted(
                    AuthService._token_cache.keys(),
                    key=lambda k: AuthService._token_cache[k]['expires_at']
                )
                for key in sorted_keys[:len(sorted_keys)//5]:
                    AuthService._token_cache.pop(key)
            
            return user_info
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to auth service: {e}")
            raise HTTPException(
                status_code=503, 
                detail="Authentication service unavailable. Please try again later."
            )
        except requests.exceptions.Timeout:
            logger.error("Authentication service timeout")
            raise HTTPException(
                status_code=504, 
                detail="Authentication service timeout. Please try again later."
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during token introspection: {e}")
            raise HTTPException(
                status_code=500, 
                detail="Authentication error. Please try again later."
            )
        except Exception as e:
            logger.error(f"Unexpected error in token validation: {e}")
            raise HTTPException(
                status_code=500, 
                detail="Internal server error during authentication"
            )

async def get_current_user(request: Request) -> str:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    token = auth_header.replace("Bearer ", "")
    user_info = await AuthService.validate_token(token)

    # FIXED LINE: user_id is now extracted from 'user_id'
    user_id = user_info.get("id")
    if not user_id:
        logger.error(f"Invalid user info: {user_info}")
        raise HTTPException(status_code=401, detail="Missing user_id in decrypted token")
    
    return user_id


# This function is for direct use without FastAPI dependency
async def get_user_from_token(token: str) -> Dict[str, Any]:
    """
    Get user information from token
    """
    return await AuthService.validate_token(token)