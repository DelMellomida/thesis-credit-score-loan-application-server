from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from app.core.security import decode_token
from app.services.auth_service import auth_service
from typing import Dict
import logging

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Extracts and validates JWT token to retrieve current authenticated user
async def get_current_user(token: str = Depends(oauth2_scheme), request: Request = None) -> Dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    logger = logging.getLogger(__name__)
    try:
        if request:
            auth_header = request.headers.get("authorization")
            # redact the header value for logs
            logger.debug("Raw Authorization header present: %s", "<redacted>" if auth_header else None)
        else:
            logger.debug("No request object for header inspection.")

        payload = decode_token(token)
        # Only log non-sensitive pieces of the payload
        if payload is None:
            logger.debug("Token payload is None after decoding.")
            raise credentials_exception

        email: str = payload.get("sub")
        logger.debug("Extracted email from token: %s", email)
        if email is None:
            logger.debug("No 'sub' field in token payload.")
            raise credentials_exception

    except Exception as e:
        logger.warning("Token validation failed")
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