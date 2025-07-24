from http.client import HTTPException
import requests
from core.crpypt_utils import decrypt_payload

from fastapi import Depends, Request

async def get_current_user(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = auth.replace("Bearer ", "")
    return get_user_from_token(token)

def get_user_from_token(token: str) -> dict:
    response = requests.post(
        "https://morderera.com/api/v1/auth/introspect",
        json={"token": token}
    )
    if response.status_code != 200:
        raise Exception("Invalid or expired token")

    encrypted_data = response.json()["data"]
    user_info = decrypt_payload(encrypted_data)
    return user_info
