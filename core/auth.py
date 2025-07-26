from typing import Optional, Dict, Any
import requests
from fastapi import HTTPException, Depends, Request
from core.config import Settings
from core.logger import logger
from core.crpypt_utils import decrypt_payload

settings = Settings()

async def get_current_user(request: Request) -> Dict[str, Any]:
    """
    Extract and validate the JWT token from request headers via the auth microservice
    """
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized - Token missing")
    
    token = auth.replace("Bearer ", "")
    
    try:
        user_info = get_user_from_token(token)
        return user_info
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise HTTPException(status_code=401, detail=str(e))

def get_user_from_token(token: str) -> Dict[str, Any]:
    """
    Validate token and extract user information through auth microservice
    """
    try:
        # Call the auth microservice to introspect and validate the token
        logger.debug(f"Sending token to auth service at {settings.auth_service_url}/api/v1/auth/introspect")
        
        response = requests.post(
            f"{settings.auth_service_url}/api/v1/auth/introspect",
            json={"token": token},
            timeout=5,  # 5 second timeout
            headers={"Content-Type": "application/json"}
        )
        
        # Check for successful response
        if response.status_code != 200:
            logger.error(f"Auth service returned status code {response.status_code}: {response.text}")
            if response.status_code == 401:
                raise Exception("Invalid or expired token")
            else:
                raise Exception(f"Authentication service error: HTTP {response.status_code}")
        
        # Extract data from response
        try:
            data = response.json()
        except ValueError:
            logger.error(f"Invalid JSON response from auth service: {response.text}")
            raise Exception("Invalid response format from authentication service")
        
        if not data or "data" not in data:
            logger.error(f"Invalid response format from auth service: {data}")
            raise Exception("Invalid response format from authentication service")
        
        # Decrypt user info using provided utility
        encrypted_data = data["data"]
        user_info = decrypt_payload(encrypted_data)
        
        # Ensure we have a user ID
        if not user_info or "user_id" not in user_info:
            logger.error(f"User info missing user_id: {user_info}")
            raise Exception("Invalid user information - missing user ID")
        
        # Log successful validation
        logger.info(f"Token validated for user: {user_info.get('user_id')}")
        
        return user_info
        
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error to auth service: {e}")
        raise Exception(f"Authentication service unavailable: Could not connect to {settings.auth_service_url}")
        
    except requests.exceptions.Timeout:
        logger.error("Authentication service timeout")
        raise Exception("Authentication service timeout")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception when calling auth service: {e}")
        raise Exception(f"Error communicating with authentication service: {str(e)}")
        
    except Exception as e:
        logger.error(f"Authentication error: {e}", exc_info=True)
        raise Exception(f"Authentication error: {str(e)}")