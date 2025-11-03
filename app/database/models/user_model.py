from beanie import Document
from pydantic import EmailStr, Field
from datetime import datetime
from typing import Optional
from bson import ObjectId

class User(Document):
    email: EmailStr = Field(..., description="Email address of the user")
    full_name: str = Field(..., description="Full name of the user")
    hashed_password: str = Field(..., description="Hashed password for the user account")
    is_active: bool = Field(default=True, description="Indicates if the user account is active")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the user was created")
    updated_at: Optional[datetime] = Field(None, description="Timestamp when the user was last updated")

    class Settings:
        name = "users"  # Collection name in MongoDB
        
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }