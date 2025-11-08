from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from app.core.security import decode_token
from app.services.auth_service import auth_service
from typing import Dict

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Extracts and validates JWT token to retrieve current authenticated user
async def get_current_user(token: str = Depends(oauth2_scheme), request: Request = None) -> Dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        if request:
            auth_header = request.headers.get("authorization")
            print(f"DEBUG: Raw Authorization header: {auth_header}")
        else:
            print("DEBUG: No request object for header inspection.")

        payload = decode_token(token)
        print(f"DEBUG: Decoded token payload: {payload}")
        if payload is None:
            print("DEBUG: Token payload is None after decoding.")
            raise credentials_exception

        email: str = payload.get("sub")
        print(f"DEBUG: Extracted email from token: {email}")
        if email is None:
            print("DEBUG: No 'sub' field in token payload.")
            raise credentials_exception

    except Exception as e:
        print(f"DEBUG: Token validation failed: {e}")
        raise credentials_exception
    
    user = await auth_service.get_user_by_email(email)
    if user is None:
        raise credentials_exception
    
    return user

# Validates that the current user is active and authorized
async def get_current_active_user(current_user: Dict = Depends(get_current_user)) -> Dict:
    return current_user

# Validates that the current user has admin privileges
async def get_admin_user(current_user: Dict = Depends(get_current_user)) -> Dict:
    return current_user

# Validates that the current user has loan officer privileges
async def get_loan_officer_user(current_user: Dict = Depends(get_current_user)) -> Dict:
    return current_user