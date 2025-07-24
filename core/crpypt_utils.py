from cryptography.fernet import Fernet

# Use a pre-generated key and share it across all services securely
SHARED_SECRET_KEY = b'my_super_secure_key_here___must_be_32_byte_base64'

fernet = Fernet(SHARED_SECRET_KEY)

def encrypt_payload(payload: dict) -> str:
    import json
    data = json.dumps(payload).encode()
    return fernet.encrypt(data).decode()

def decrypt_payload(token: str) -> dict:
    import json
    data = fernet.decrypt(token.encode())
    return json.loads(data)
