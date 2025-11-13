from fastapi import HTTPException, status
from app.database.models import User
from app.schemas import UserCreate
from app.core import hash_password, verify_password, create_access_token, is_valid_password
from typing import Dict, Optional
from datetime import datetime
import logging

class AuthService:
    # Register a new user with email and password validation
    @staticmethod
    async def register_user(user_data: UserCreate) -> Dict:
        existing_user = await User.find_one(User.email == user_data.email)
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        if not is_valid_password(user_data.password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters long"
            )
        
        try:
            hashed_password = hash_password(user_data.password)
            logging.getLogger(__name__).debug("Hashed password created (redacted): %s...", hashed_password[:12])
        except ValueError as e:
            logging.getLogger(__name__).warning("Password hashing failed")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid password format"
            )
        
        new_user = User(
            email=user_data.email,
            full_name=user_data.full_name,
            hashed_password=hashed_password,
            created_at=datetime.utcnow()
        )
        
        try:
            await new_user.insert()
            logging.getLogger(__name__).debug("User saved with ID: %s", new_user.id)
        except Exception as e:
            logging.getLogger(__name__).error("User save failed: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="User registration failed"
            )
        
        return {
            "id": str(new_user.id),
            "email": user_data.email,
            "full_name": user_data.full_name,
            "message": "User registered successfully"
        }
    
    # Authenticate user and generate access token
    @staticmethod
    async def login_user(form_data) -> Dict:
        logging.getLogger(__name__).debug("Login attempt for email: %s", form_data.email)
        
        user = await User.find_one(User.email == form_data.email)
        
        if not user:
            logging.getLogger(__name__).warning("User not found for email: %s", form_data.email)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        logging.getLogger(__name__).debug("User found: %s", user.email)
        # Log a short prefix of the stored hash for correlation only; do NOT log full hash or plaintext password
        logging.getLogger(__name__).debug("Stored hash (prefix): %s...", (user.hashed_password or '')[:12])

        # Do NOT log the incoming plaintext password

        password_valid = verify_password(form_data.password, user.hashed_password)
        logging.getLogger(__name__).debug("Password verification result: %s", password_valid)
        
        if not password_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        try:
            access_token = create_access_token(data={"sub": user.email})
            # Never log the full access token; log that one was created and the subject
            logging.getLogger(__name__).debug("Created JWT access token for sub: %s", user.email)
        except ValueError as e:
            logging.getLogger(__name__).error("Token creation failed: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create access token"
            )
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": str(user.id),
                "email": user.email,
                "full_name": user.full_name
            }
        }
    
    # Retrieve user information by email address
    @staticmethod
    async def get_user_by_email(email: str) -> Optional[Dict]:
        user = await User.find_one(User.email == email)
        if not user:
            return None
        
        return {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name
        }
    
    # Generate a new access token for existing user
    @staticmethod
    async def refresh_user_token(email: str) -> Dict:
        user = await User.find_one(User.email == email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        try:
            access_token = create_access_token(data={"sub": email})
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create access token"
            )
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }

auth_service = AuthService()