from fastapi import HTTPException, status
from app.database.models import User
from app.schemas import UserCreate
from app.core import hash_password, verify_password, create_access_token, is_valid_password
from typing import Dict, Optional
from datetime import datetime

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
            print(f"DEBUG: Hashed password created: {hashed_password[:20]}...")
        except ValueError as e:
            print(f"DEBUG: Password hashing failed: {e}")
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
            print(f"DEBUG: User saved with ID: {new_user.id}")
        except Exception as e:
            print(f"DEBUG: User save failed: {e}")
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
        print(f"DEBUG: Login attempt for email: {form_data.email}")
        
        user = await User.find_one(User.email == form_data.email)
        
        if not user:
            print(f"DEBUG: User not found for email: {form_data.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        print(f"DEBUG: User found: {user.email}")
        print(f"DEBUG: Stored hash: {user.hashed_password[:20]}...")
        print(f"DEBUG: Input password: {form_data.password}")
        
        password_valid = verify_password(form_data.password, user.hashed_password)
        print(f"DEBUG: Password verification result: {password_valid}")
        
        if not password_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        try:
            access_token = create_access_token(data={"sub": user.email})
            print(f"DEBUG: Created JWT access token: {access_token}")
        except ValueError as e:
            print(f"DEBUG: Token creation failed: {e}")
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