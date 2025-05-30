import uuid
from datetime import datetime
from typing import Any, Dict, Optional
import json

def generate_id() -> str:
    """Generate a unique ID"""
    return str(uuid.uuid4())

def serialize_datetime(dt: datetime) -> str:
    """Serialize datetime to ISO string"""
    return dt.isoformat()

def deserialize_datetime(dt_str: str) -> datetime:
    """Deserialize ISO string to datetime"""
    return datetime.fromisoformat(dt_str)

def safe_json_loads(data: str, default: Any = None) -> Any:
    """Safely load JSON data"""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(data: Any, default: str = "{}") -> str:
    """Safely dump data to JSON"""
    try:
        return json.dumps(data)
    except (TypeError, ValueError):
        return default

def validate_uuid(uuid_str: str) -> bool:
    """Validate if string is a valid UUID"""
    try:
        uuid.UUID(uuid_str)
        return True
    except ValueError:
        return False