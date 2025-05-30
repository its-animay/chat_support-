import redis
import json
from typing import Optional, Dict, Any, List
from core.config import Settings
from core.logger import logger

settings =Settings()

class RedisClient:
    def __init__(self):
        self.client = redis.from_url(settings.redis_url, decode_responses=True)
        
    async def ping(self) -> bool:
        try:
            return self.client.ping()
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False
    
    # JSON operations
    def json_set(self, key: str, data: Dict[str, Any]) -> bool:
        try:
            self.client.json().set(key, "$", data)
            return True
        except Exception as e:
            logger.error(f"Redis JSON set failed for key {key}: {e}")
            return False
    
    def json_get(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            result = self.client.json().get(key)
            return result
        except Exception as e:
            logger.error(f"Redis JSON get failed for key {key}: {e}")
            return None
    
    def json_delete(self, key: str) -> bool:
        try:
            self.client.json().delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis JSON delete failed for key {key}: {e}")
            return False
    
    # Stream operations
    def stream_add(self, stream_key: str, data: Dict[str, Any]) -> Optional[str]:
        try:
            message_id = self.client.xadd(stream_key, data)
            return message_id
        except Exception as e:
            logger.error(f"Redis stream add failed for {stream_key}: {e}")
            return None
    
    def stream_read(self, stream_key: str, count: int = 100) -> List[Dict[str, Any]]:
        try:
            messages = self.client.xread({stream_key: "0"}, count=count)
            if messages:
                return [
                    {
                        "id": msg_id,
                        "data": msg_data
                    }
                    for stream, msgs in messages
                    for msg_id, msg_data in msgs
                ]
            return []
        except Exception as e:
            logger.error(f"Redis stream read failed for {stream_key}: {e}")
            return []
    
    # List operations
    def list_push(self, key: str, value: str) -> bool:
        try:
            self.client.lpush(key, value)
            return True
        except Exception as e:
            logger.error(f"Redis list push failed for key {key}: {e}")
            return False
    
    def list_get(self, key: str) -> List[str]:
        try:
            return self.client.lrange(key, 0, -1)
        except Exception as e:
            logger.error(f"Redis list get failed for key {key}: {e}")
            return []
    
    # Key operations
    def exists(self, key: str) -> bool:
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Redis exists check failed for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete failed for key {key}: {e}")
            return False
    
    def scan_keys(self, pattern: str) -> List[str]:
        try:
            return list(self.client.scan_iter(match=pattern))
        except Exception as e:
            logger.error(f"Redis scan failed for pattern {pattern}: {e}")
            return []

redis_client = RedisClient()
