import hashlib
import secrets
from typing import Optional

def generate_key(length: int = 32) -> bytes:
    return secrets.token_bytes(length)

def hash_data(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

class Encryption:
    def __init__(self, key: Optional[bytes] = None):
        self.key = key or generate_key()
        
    def encrypt(self, data: str) -> bytes:
        # Заглушка для шифрования
        return data.encode()
        
    def decrypt(self, data: bytes) -> str:
        # Заглушка для дешифрования
        return data.decode() 