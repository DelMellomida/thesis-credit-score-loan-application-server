# schemas.py
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class UserCreate(BaseModel):
    email: EmailStr = Field(..., description="Email address of the user")
    full_name: str = Field(..., description="Full name of the user")
    password: str = Field(..., min_length=8, description="Password for the user account")

class UserResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the user")
    email: EmailStr = Field(..., description="Email address of the user")
    full_name: str = Field(..., description="Full name of the user")
    message: Optional[str] = None

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str = Field(..., description="Access token for the user")
    token_type: str = Field(default="bearer", description="Type of the token")

class TokenData(BaseModel):
    email: Optional[EmailStr] = None