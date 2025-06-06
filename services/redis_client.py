import redis
import json
from typing import Optional, Dict, Any, List
from core.config import Settings
from core.logger import logger

settings = Settings()

class RedisClient:
    def __init__(self):
        self.client = redis.from_url(settings.redis_url, decode_responses=True)
        
    async def ping(self) -> bool:
        try:
            return self.client.ping()
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False
    
    # JSON operations using core Redis
    def json_set(self, key: str, data: Dict[str, Any]) -> bool:
        try:
            # Serialize to JSON string and store as regular string
            json_str = json.dumps(data, default=str)
            self.client.set(key, json_str)
            return True
        except Exception as e:
            logger.error(f"Redis JSON set failed for key {key}: {e}")
            return False
    
    def json_get(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            # Get string and deserialize from JSON
            json_str = self.client.get(key)
            if json_str:
                return json.loads(json_str)
            return None
        except Exception as e:
            logger.error(f"Redis JSON get failed for key {key}: {e}")
            return None
    
    def json_delete(self, key: str) -> bool:
        try:
            # Use regular delete for JSON keys
            result = self.client.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Redis JSON delete failed for key {key}: {e}")
            return False
    
    # Stream operations (these work with core Redis)
    def stream_add(self, stream_key: str, data: Dict[str, Any]) -> Optional[str]:
        try:
            # Convert dict values to strings for stream storage
            string_data = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                          for k, v in data.items()}
            message_id = self.client.xadd(stream_key, string_data)
            return message_id
        except Exception as e:
            logger.error(f"Redis stream add failed for {stream_key}: {e}")
            return None
    
    def stream_read(self, stream_key: str, count: int = 100) -> List[Dict[str, Any]]:
        try:
            messages = self.client.xread({stream_key: "0"}, count=count)
            if messages:
                result = []
                for stream, msgs in messages:
                    for msg_id, msg_data in msgs:
                        # Try to deserialize JSON values back
                        processed_data = {}
                        for k, v in msg_data.items():
                            try:
                                processed_data[k] = json.loads(v)
                            except (json.JSONDecodeError, TypeError):
                                processed_data[k] = v
                        
                        result.append({
                            "id": msg_id,
                            "data": processed_data
                        })
                return result
            return []
        except Exception as e:
            logger.error(f"Redis stream read failed for {stream_key}: {e}")
            return []
    
    # List operations (work with core Redis)
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
    
    # Key operations (work with core Redis)
    def exists(self, key: str) -> bool:
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Redis exists check failed for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        try:
            result = self.client.delete(key)
            return bool(result)
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