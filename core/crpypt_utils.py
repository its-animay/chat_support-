from cryptography.fernet import Fernet

# Use a pre-generated key and share it across all services securely
SHARED_SECRET_KEY = b'5IyViD8_gZ3Mrq2pw3sZvXB6_6pUMfvMco34cjk3kBQ='

fernet = Fernet(SHARED_SECRET_KEY)

def encrypt_payload(payload: dict) -> str:
    import json
    data = json.dumps(payload).encode()
    return fernet.encrypt(data).decode()

def decrypt_payload(token: str) -> dict:
    import json
    data = fernet.decrypt(token.encode())
    return json.loads(data)
