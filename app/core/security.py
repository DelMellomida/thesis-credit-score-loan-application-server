from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from typing import Optional, Dict, Any
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Validates that a password meets minimum length requirements
def is_valid_password(password: str) -> bool:
    return len(password) >= 8

# Hashes a password using bcrypt after validating its length
def hash_password(password: str) -> str:
    if not is_valid_password(password):
        raise ValueError("Password must be at least 8 characters long")
    
    try:
        return pwd_context.hash(password)
    except Exception as e:
        raise ValueError("Failed to hash password") from e

# Verifies a plain password against its hashed version
def verify_password(password: str, hashed_password: str) -> bool:
    try:
        return pwd_context.verify(password, hashed_password)
    except Exception as e:
        print(f"Password verification error: {e}")
        return False

# Creates a JWT access token with configurable expiration time
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    try:
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
        return encoded_jwt
    except Exception as e:
        raise ValueError("Failed to create access token") from e

# Creates a JWT refresh token with longer expiration time
def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    try:
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + (expires_delta or timedelta(days=7))
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
        return encoded_jwt
    except Exception as e:
        raise ValueError("Failed to create refresh token") from e

# Decodes and validates a JWT token returning its payload
def decode_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        return None