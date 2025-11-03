# app/routes/auth_routes.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import Dict

from pydantic import EmailStr

from app.services.auth_service import auth_service
from app.schemas import UserCreate, UserResponse, Token
from app.core.auth_dependencies import get_current_user

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)

@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def signup_user(user_data: UserCreate) -> UserResponse:
    try:
        created_user = await auth_service.register_user(user_data)
        return UserResponse(**created_user)
    except HTTPException:
        # Re-raise HTTPExceptions from the service layer
        raise
    except Exception as e:
        # Log the actual error for debugging but don't expose it
        print(f"Unexpected error during signup: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during registration"
        )

@router.post("/signup-demo", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def signup_user_demo(
    email: EmailStr,
    full_name: str,
    password: str
) -> UserResponse:
    user_data = UserCreate(email=email, full_name=full_name, password=password)

    try:
        created_user = await auth_service.register_user(user_data)
        return UserResponse(**created_user)
    except HTTPException:
        # Re-raise HTTPExceptions from the service layer
        raise
    except Exception as e:
        # Log the actual error for debugging but don't expose it
        print(f"Unexpected error during signup: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during registration"
        )

@router.post("/login", response_model=Token, status_code=status.HTTP_200_OK)
async def login_user(form_data: OAuth2PasswordRequestForm = Depends()) -> Token:
    try:
        # Create a simple data class for the login data
        class LoginData:
            def __init__(self, email: str, password: str):
                self.email = email
                self.password = password
        
        login_data = LoginData(email=form_data.username, password=form_data.password)
        token_data = await auth_service.login_user(login_data)
        
        return Token(
            access_token=token_data["access_token"],
            token_type=token_data["token_type"]
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error during login: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during login"
        )

@router.get("/me", response_model=UserResponse, status_code=status.HTTP_200_OK)
async def get_current_user_info(current_user: dict = Depends(get_current_user)) -> UserResponse:
    try:
        return UserResponse(**current_user)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error fetching user info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while fetching user information"
        )

@router.post("/refresh", response_model=Token, status_code=status.HTTP_200_OK)
async def refresh_token(current_user: dict = Depends(get_current_user)) -> Token:
    try:
        token_data = await auth_service.refresh_user_token(current_user["email"])
        return Token(
            access_token=token_data["access_token"],
            token_type=token_data["token_type"]
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error during token refresh: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during token refresh"
        )

# # Add a simple test endpoint
# @router.get("/test")
# async def test_endpoint():
#     """Simple test endpoint to verify router is working"""
#     return {"message": "Auth router is working!"}

# # Add a debug endpoint
# @router.get("/debug")
# async def debug_auth():
#     return {
#         "message": "Auth debug endpoint",
#         "available_routes": [
#             "/auth/signup (POST)",
#             "/auth/login (POST)", 
#             "/auth/me (GET)",
#             "/auth/refresh (POST)",
#             "/auth/test (GET)",
#             "/auth/debug (GET)"
#         ]
#     }