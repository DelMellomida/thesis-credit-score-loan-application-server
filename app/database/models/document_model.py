from beanie import Document
from typing import Optional, Dict
from datetime import datetime
from pydantic import Field
import uuid

class ApplicationDocument(Document):
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    application_id: str
    profile_photo_url: Optional[str] = None
    valid_id_url: Optional[str] = None
    brgy_cert_url: Optional[str] = None
    e_signature_personal_url: Optional[str] = None
    payslip_url: Optional[str] = None
    company_id_url: Optional[str] = None
    proof_of_billing_url: Optional[str] = None
    e_signature_comaker_url: Optional[str] = None
    file_metadata: Optional[Dict[str, Dict[str, str]]] = None  # {field: {filename, size}}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str
    updated_by: str

    class Settings:
        name = "application_documents"