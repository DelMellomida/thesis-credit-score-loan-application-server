from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import Dict

from pydantic import EmailStr

from app.services.auth_service import auth_service
from app.schemas import UserCreate, UserResponse, Token
from app.core.auth_dependencies import get_current_user
from app.services.audit_service import audit_service
import logging

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)

# Registers a new user account with email and password
@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def signup_user(user_data: UserCreate) -> UserResponse:
    try:
        created_user = await auth_service.register_user(user_data)
        # Audit: successful signup
        try:
            await audit_service.create_audit(action="signup", actor=created_user.get("email") or created_user.get("id"), acted=created_user.get("id"), status="successful")
        except Exception:
            logging.getLogger(__name__).exception("Failed to write signup audit log")
        return UserResponse(**created_user)
    except HTTPException:
        # Audit: failed signup attempt
        try:
            await audit_service.create_audit(action="signup", actor=user_data.email if hasattr(user_data, 'email') else None, acted=None, status="failed")
        except Exception:
            logging.getLogger(__name__).exception("Failed to write failed signup audit log")
        raise
    except Exception as e:
        print(f"Unexpected error during signup: {str(e)}")
        try:
            await audit_service.create_audit(action="signup", actor=user_data.email if hasattr(user_data, 'email') else None, acted=None, status="failed")
        except Exception:
            logging.getLogger(__name__).exception("Failed to write failed signup audit log")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during registration"
        )

# Demo signup endpoint accepting individual parameters instead of a model
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
        raise
    except Exception as e:
        print(f"Unexpected error during signup: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during registration"
        )

# Authenticates user credentials and returns an access token
@router.post("/login", response_model=Token, status_code=status.HTTP_200_OK)
async def login_user(form_data: OAuth2PasswordRequestForm = Depends()) -> Token:
    try:
        class LoginData:
            def __init__(self, email: str, password: str):
                self.email = email
                self.password = password
        
        login_data = LoginData(email=form_data.username, password=form_data.password)
        token_data = await auth_service.login_user(login_data)

        # Audit: successful login
        try:
            await audit_service.create_audit(action="login", actor=login_data.email, acted=None, status="successful")
        except Exception:
            logging.getLogger(__name__).exception("Failed to write login audit log")

        return Token(
            access_token=token_data["access_token"],
            token_type=token_data["token_type"]
        )
    except HTTPException:
        # Audit: failed login
        try:
            await audit_service.create_audit(action="login", actor=form_data.username if hasattr(form_data, 'username') else None, acted=None, status="failed")
        except Exception:
            logging.getLogger(__name__).exception("Failed to write failed login audit log")
        raise
    except Exception as e:
        print(f"Unexpected error during login: {str(e)}")
        try:
            await audit_service.create_audit(action="login", actor=form_data.username if hasattr(form_data, 'username') else None, acted=None, status="failed")
        except Exception:
            logging.getLogger(__name__).exception("Failed to write failed login audit log")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during login"
        )

# Retrieves the authenticated user's profile information
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

# Generates a new access token for the authenticated user
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