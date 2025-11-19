from beanie import Document
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class AuditLog(Document):
    action: str = Field(..., description="Action performed (e.g. 'login', 'create_application')")
    actor: Optional[str] = Field(None, description="Identifier of the actor who performed the action")
    acted: Optional[str] = Field(None, description="Identifier of the entity acted upon (application id, user id, etc.)")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the action occurred")
    status: str = Field(..., description="Result status: 'successful' or 'failed'")

    class Settings:
        name = "audit_logs"

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
