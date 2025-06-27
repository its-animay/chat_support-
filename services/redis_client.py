import redis
import json
import asyncio
from typing import Optional, Dict, Any, List, Union, Callable
from functools import wraps
from core.config import Settings
from core.logger import logger
import os
import pickle

settings = Settings()


_memory_cache = {}

def with_fallback(func):
    """Decorator to add memory fallback capability to Redis methods"""
    @wraps(func)
    async def wrapper(self, key: str, *args, **kwargs):
        try:
            result = await func(self, key, *args, **kwargs)
            
            if func.__name__ in ['json_set', 'list_push', 'stream_add']:
                if func.__name__ == 'json_set' and args:
                    _memory_cache[key] = args[0]
                elif func.__name__ == 'list_push' and args:
                    if key not in _memory_cache:
                        _memory_cache[key] = []
                    _memory_cache[key].append(args[0])
                elif func.__name__ == 'stream_add' and args:
                    if key not in _memory_cache:
                        _memory_cache[key] = []
                    _memory_cache[key].append({"id": f"mem_{len(_memory_cache[key])}", "data": args[0]})
            
            return result
        except Exception as e:
            logger.error(f"Redis operation {func.__name__} failed for key {key}: {e}")
            
            # For get operations, return from memory cache if available
            if func.__name__ == 'json_get':
                if key in _memory_cache:
                    logger.warning(f"Using memory fallback for key {key}")
                    return _memory_cache[key]
            elif func.__name__ == 'list_get':
                if key in _memory_cache:
                    logger.warning(f"Using memory fallback for key {key}")
                    return _memory_cache[key]
            elif func.__name__ == 'stream_read':
                if key in _memory_cache:
                    logger.warning(f"Using memory fallback for key {key}")
                    return _memory_cache[key]
            
            if func.__name__ in ['json_set', 'list_push', 'delete', 'json_delete']:
                return False
            elif func.__name__ == 'stream_add':
                return None
            elif func.__name__ in ['json_get']:
                return None
            elif func.__name__ in ['list_get', 'stream_read', 'scan_keys']:
                return []
            else:
                return None
    
    return wrapper

class RedisTransaction:
    """Redis transaction manager for atomic operations"""
    
    def __init__(self, client):
        self.client = client
        self.pipeline = None
        self.commands = []
        self.success = False
        self.enable_disk_fallback = getattr(settings, 'redis_fallback_enabled', False)
        self.fallback_dir = getattr(settings, 'redis_fallback_dir', "./fallback_storage")
    
    async def __aenter__(self):
        self.pipeline = self.client.pipeline(transaction=True)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and self.pipeline:
            try:
                # Execute all commands atomically
                results = await asyncio.to_thread(self.pipeline.execute)
                self.success = all(results) if results else False
                
                # If successful and using disk persistence, save changes
                if self.success and self.enable_disk_fallback:
                    await self._persist_transaction()
            except Exception as e:
                logger.error(f"Transaction execution failed: {e}")
                self.success = False
        
        self.pipeline = None
        return False
    
    async def json_set(self, key: str, data: Dict[str, Any]):
        """Add JSON set operation to transaction"""
        self.commands.append(('json_set', key, data))
        json_str = json.dumps(data, default=str)
        self.pipeline.set(key, json_str)
        return self
    
    async def list_push(self, key: str, value: str):
        """Add list push operation to transaction"""
        self.commands.append(('list_push', key, value))
        self.pipeline.lpush(key, value)
        return self
    
    async def stream_add(self, key: str, data: Dict[str, Any]):
        """Add stream add operation to transaction"""
        self.commands.append(('stream_add', key, data))
        string_data = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                      for k, v in data.items()}
        self.pipeline.xadd(key, string_data)
        return self
    
    async def delete(self, key: str):
        """Add delete operation to transaction"""
        self.commands.append(('delete', key))
        self.pipeline.delete(key)
        return self
    
    async def _persist_transaction(self):
        """Persist transaction commands to disk for recovery"""
        try:
            fallback_dir = os.environ.get("FALLBACK_STORAGE_DIR", "./fallback_storage")
            os.makedirs(fallback_dir, exist_ok=True)
            
            filepath = os.path.join(fallback_dir, f"transaction_{int(asyncio.get_event_loop().time() * 1000)}.pickle")
            
            with open(filepath, 'wb') as f:
                pickle.dump(self.commands, f)
        except Exception as e:
            logger.error(f"Failed to persist transaction to disk: {e}")


class RedisClient:
    """Enhanced Redis client with async support, transactions, and fallback mechanisms"""
    
    def __init__(self):
        # Initialize the synchronous client for background operations
        self.client = redis.from_url(settings.redis_url, decode_responses=True)
        self.connected = False
        
        # Check if fallback settings exist, otherwise use defaults
        self.enable_disk_fallback = getattr(settings, 'redis_fallback_enabled', False)
        self.fallback_dir = getattr(settings, 'redis_fallback_dir', "./fallback_storage")
        self.max_concurrent_ops = getattr(settings, 'redis_max_concurrent_ops', 100)
        
        # Try to connect immediately
        try:
            self.connected = self.client.ping()
            if self.connected:
                logger.info("Successfully connected to Redis")
            else:
                logger.warning("Redis ping returned False - connection issues may exist")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
    
    async def transaction(self) -> RedisTransaction:
        """Create a new transaction context"""
        return RedisTransaction(self.client)
    
    async def ping(self) -> bool:
        """Check if Redis is reachable"""
        try:
            result = await asyncio.to_thread(self.client.ping)
            self.connected = result
            return result
        except Exception as e:
            self.connected = False
            logger.error(f"Redis ping failed: {e}")
            return False
    
    async def _recover_from_disk(self):
        """Attempt to recover operations from disk storage"""
        if not self.enable_disk_fallback:
            return
            
        try:
            fallback_dir = os.environ.get("FALLBACK_STORAGE_DIR", "./fallback_storage")
            if not os.path.exists(fallback_dir):
                return
                
            files = sorted([f for f in os.listdir(fallback_dir) if f.startswith("transaction_")])
            
            for file in files:
                filepath = os.path.join(fallback_dir, file)
                try:
                    with open(filepath, 'rb') as f:
                        commands = pickle.load(f)
                    
                    # Attempt to replay the transaction
                    async with await self.transaction() as tx:
                        for cmd_type, key, data in commands:
                            if cmd_type == 'json_set':
                                await tx.json_set(key, data)
                            elif cmd_type == 'list_push':
                                await tx.list_push(key, data)
                            elif cmd_type == 'stream_add':
                                await tx.stream_add(key, data)
                            elif cmd_type == 'delete':
                                await tx.delete(key)
                    
                    # If successful, remove the file
                    if tx.success:
                        os.remove(filepath)
                except Exception as e:
                    logger.error(f"Failed to recover transaction from {filepath}: {e}")
        except Exception as e:
            logger.error(f"Error during recovery from disk: {e}")
    
    @with_fallback
    async def json_set(self, key: str, data: Dict[str, Any]) -> bool:
        """Set a JSON value in Redis with async support"""
        try:
            json_str = json.dumps(data, default=str)
            result = await asyncio.to_thread(self.client.set, key, json_str)
            
            if result and self.enable_disk_fallback:
                await self._persist_to_disk(key, data)
                
            return bool(result)
        except Exception as e:
            logger.error(f"Redis JSON set failed for key {key}: {e}")
            
            _memory_cache[key] = data
            
            if self.enable_disk_fallback:
                await self._persist_to_disk(key, data)
                
            return False
    
    @with_fallback
    async def json_get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a JSON value from Redis with async support"""
        try:
            # Get string and deserialize from JSON
            json_str = await asyncio.to_thread(self.client.get, key)
            if json_str:
                return json.loads(json_str)
            return None
        except Exception as e:
            logger.error(f"Redis JSON get failed for key {key}: {e}")
            
            # Try to load from disk if disk fallback is enabled
            if self.enable_disk_fallback:
                disk_data = await self._load_from_disk(key)
                if disk_data:
                    return disk_data
                    
            return None
    
    @with_fallback
    async def json_delete(self, key: str) -> bool:
        """Delete a JSON value from Redis with async support"""
        try:
            # Use regular delete for JSON keys
            result = await asyncio.to_thread(self.client.delete, key)
            
            # Remove from disk if fallback enabled
            if self.enable_disk_fallback:
                await self._remove_from_disk(key)
                
            # Remove from memory cache
            if key in _memory_cache:
                del _memory_cache[key]
                
            return bool(result)
        except Exception as e:
            logger.error(f"Redis JSON delete failed for key {key}: {e}")
            
            # Remove from memory cache
            if key in _memory_cache:
                del _memory_cache[key]
                
            # Remove from disk if fallback enabled
            if self.enable_disk_fallback:
                await self._remove_from_disk(key)
                
            return False
    
    @with_fallback
    async def stream_add(self, stream_key: str, data: Dict[str, Any]) -> Optional[str]:
        """Add data to a Redis stream with async support"""
        try:
            # Convert dict values to strings for stream storage
            string_data = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                          for k, v in data.items()}
            message_id = await asyncio.to_thread(self.client.xadd, stream_key, string_data)
            
            # Update memory cache
            if stream_key not in _memory_cache:
                _memory_cache[stream_key] = []
            _memory_cache[stream_key].append({"id": message_id, "data": data})
            
            return message_id
        except Exception as e:
            logger.error(f"Redis stream add failed for {stream_key}: {e}")
            
            # Store in memory fallback
            if stream_key not in _memory_cache:
                _memory_cache[stream_key] = []
            mem_id = f"mem_{len(_memory_cache[stream_key])}"
            _memory_cache[stream_key].append({"id": mem_id, "data": data})
            
            return None
    
    @with_fallback
    async def stream_read(self, stream_key: str, count: int = 100) -> List[Dict[str, Any]]:
        """Read from a Redis stream with async support"""
        try:
            messages = await asyncio.to_thread(self.client.xread, {stream_key: "0"}, count=count)
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
            
            # Return from memory cache if available
            if stream_key in _memory_cache:
                return _memory_cache[stream_key]
                
            return []
    
    @with_fallback
    async def list_push(self, key: str, value: str) -> bool:
        """Push a value to a Redis list with async support"""
        try:
            result = await asyncio.to_thread(self.client.lpush, key, value)
            
            # Update memory cache
            if key not in _memory_cache:
                _memory_cache[key] = []
            _memory_cache[key].append(value)
            
            return bool(result)
        except Exception as e:
            logger.error(f"Redis list push failed for key {key}: {e}")
            
            # Store in memory fallback
            if key not in _memory_cache:
                _memory_cache[key] = []
            _memory_cache[key].append(value)
            
            return False
    
    @with_fallback
    async def list_get(self, key: str) -> List[str]:
        """Get all values from a Redis list with async support"""
        try:
            return await asyncio.to_thread(self.client.lrange, key, 0, -1)
        except Exception as e:
            logger.error(f"Redis list get failed for key {key}: {e}")
            
            # Return from memory cache if available
            if key in _memory_cache:
                return _memory_cache[key]
                
            return []
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis with async support"""
        try:
            return bool(await asyncio.to_thread(self.client.exists, key))
        except Exception as e:
            logger.error(f"Redis exists check failed for key {key}: {e}")
            
            # Check memory cache
            return key in _memory_cache
    
    @with_fallback
    async def delete(self, key: str) -> bool:
        """Delete a key from Redis with async support"""
        try:
            result = await asyncio.to_thread(self.client.delete, key)
            
            # Remove from memory cache
            if key in _memory_cache:
                del _memory_cache[key]
                
            # Remove from disk if fallback enabled
            if self.enable_disk_fallback:
                await self._remove_from_disk(key)
                
            return bool(result)
        except Exception as e:
            logger.error(f"Redis delete failed for key {key}: {e}")
            
            # Remove from memory cache
            if key in _memory_cache:
                del _memory_cache[key]
                
            # Remove from disk if fallback enabled
            if self.enable_disk_fallback:
                await self._remove_from_disk(key)
                
            return False
    
    @with_fallback
    async def scan_keys(self, pattern: str) -> List[str]:
        """Scan for keys matching a pattern with async support"""
        try:
            # Need to convert iterator to list for async operation
            keys = []
            cursor = 0
            while True:
                cursor, partial_keys = await asyncio.to_thread(self.client.scan, cursor, match=pattern)
                keys.extend(partial_keys)
                if cursor == 0:
                    break
            return keys
        except Exception as e:
            logger.error(f"Redis scan failed for pattern {pattern}: {e}")
            
            # If memory cache is used, try to find matching keys there
            if _memory_cache:
                import fnmatch
                memory_keys = [k for k in _memory_cache.keys() if fnmatch.fnmatch(k, pattern)]
                return memory_keys
                
            return []
    
    async def _persist_to_disk(self, key: str, data: Any):
        """Persist data to disk as fallback"""
        if not self.enable_disk_fallback:
            return
            
        try:
            fallback_dir = os.environ.get("FALLBACK_STORAGE_DIR", "./fallback_storage")
            os.makedirs(fallback_dir, exist_ok=True)
            
            # Create a sanitized filename
            sanitized_key = key.replace(":", "_").replace("/", "_")
            filepath = os.path.join(fallback_dir, f"{sanitized_key}.json")
            
            # Save the data
            with open(filepath, 'w') as f:
                json.dump(data, f, default=str)
        except Exception as e:
            logger.error(f"Failed to persist {key} to disk: {e}")
    
    async def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load data from disk fallback"""
        if not self.enable_disk_fallback:
            return None
            
        try:
            fallback_dir = os.environ.get("FALLBACK_STORAGE_DIR", "./fallback_storage")
            sanitized_key = key.replace(":", "_").replace("/", "_")
            filepath = os.path.join(fallback_dir, f"{sanitized_key}.json")
            
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Failed to load {key} from disk: {e}")
            return None
    
    async def _remove_from_disk(self, key: str) -> bool:
        """Remove data from disk fallback"""
        if not self.enable_disk_fallback:
            return True
            
        try:
            fallback_dir = os.environ.get("FALLBACK_STORAGE_DIR", "./fallback_storage")
            sanitized_key = key.replace(":", "_").replace("/", "_")
            filepath = os.path.join(fallback_dir, f"{sanitized_key}.json")
            
            if os.path.exists(filepath):
                os.remove(filepath)
            return True
        except Exception as e:
            logger.error(f"Failed to remove {key} from disk: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Redis health and return status information"""
        health = {
            "connected": False,
            "memory_cache_size": len(_memory_cache),
            "disk_fallback_enabled": self.enable_disk_fallback
        }
        
        try:
            health["connected"] = await self.ping()
            if health["connected"]:
                # Get some info about the Redis server
                info = await asyncio.to_thread(self.client.info)
                health["version"] = info.get("redis_version", "unknown")
                health["memory_used"] = info.get("used_memory_human", "unknown")
                health["clients_connected"] = info.get("connected_clients", 0)
                
                # If we reconnected after being disconnected, try to recover
                if not self.connected and health["connected"]:
                    logger.info("Redis connection restored, attempting recovery")
                    await self._recover_from_disk()
                    self.connected = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            health["error"] = str(e)
            
        return health
    
    async def set_expiration(self, key: str, seconds: int) -> bool:
        """Set expiration time on a key"""
        try:
            result = await asyncio.to_thread(self.client.expire, key, seconds)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to set expiration for key {key}: {e}")
            return False

# Initialize a single Redis client instance
redis_client = RedisClient()