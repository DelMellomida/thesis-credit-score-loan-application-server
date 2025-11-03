from fastapi import HTTPException, status
from app.database.models import User
from app.schemas import UserCreate
from app.core import hash_password, verify_password, create_access_token, is_valid_password
from typing import Dict, Optional
from datetime import datetime

class AuthService:
    @staticmethod
    async def register_user(user_data: UserCreate) -> Dict:
        # Check if email already exists using Beanie
        existing_user = await User.find_one(User.email == user_data.email)
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Validate password
        if not is_valid_password(user_data.password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters long"
            )
        
        try:
            hashed_password = hash_password(user_data.password)
            print(f"DEBUG: Hashed password created: {hashed_password[:20]}...")  # Debug log
        except ValueError as e:
            print(f"DEBUG: Password hashing failed: {e}")  # Debug log
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid password format"
            )
        
        # Create new user using Beanie
        new_user = User(
            email=user_data.email,
            full_name=user_data.full_name,
            hashed_password=hashed_password,
            created_at=datetime.utcnow()
        )
        
        # Save user to database using Beanie
        try:
            await new_user.insert()
            print(f"DEBUG: User saved with ID: {new_user.id}")  # Debug log
        except Exception as e:
            print(f"DEBUG: User save failed: {e}")  # Debug log
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="User registration failed"
            )
        
        # Return user data (excluding sensitive information)
        return {
            "id": str(new_user.id),
            "email": user_data.email,
            "full_name": user_data.full_name,
            "message": "User registered successfully"
        }
    
    @staticmethod
    async def login_user(form_data) -> Dict:
        print(f"DEBUG: Login attempt for email: {form_data.email}")  # Debug log
        
        # Find user by email using Beanie
        user = await User.find_one(User.email == form_data.email)
        
        if not user:
            print(f"DEBUG: User not found for email: {form_data.email}")  # Debug log
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        print(f"DEBUG: User found: {user.email}")  # Debug log
        print(f"DEBUG: Stored hash: {user.hashed_password[:20]}...")  # Debug log
        print(f"DEBUG: Input password: {form_data.password}")  # Debug log
        
        # Verify password
        password_valid = verify_password(form_data.password, user.hashed_password)
        print(f"DEBUG: Password verification result: {password_valid}")  # Debug log
        
        if not password_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Create access token
        try:
            access_token = create_access_token(data={"sub": user.email})
            print(f"DEBUG: Created JWT access token: {access_token}")  # Debug log
        except ValueError as e:
            print(f"DEBUG: Token creation failed: {e}")  # Debug log
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
    
    @staticmethod
    async def get_user_by_email(email: str) -> Optional[Dict]:
        """Helper method to get user by email"""
        user = await User.find_one(User.email == email)
        if not user:
            return None
        
        return {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name
        }
    
    @staticmethod
    async def refresh_user_token(email: str) -> Dict:
        """Refresh access token for user"""
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