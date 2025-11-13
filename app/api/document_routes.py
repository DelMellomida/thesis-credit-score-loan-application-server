from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, Header
from typing import Optional, List
from datetime import datetime
from app.services.document_service import DocumentService
from app.core.idempotency import get_idempotency_store
import json
from app.schemas.document_schema import DocumentUploadRequest, DocumentResponse
from app.core.auth_dependencies import get_current_active_user

router = APIRouter(prefix="/documents", tags=["Documents"])
service = DocumentService()
_idempotency = get_idempotency_store()

# Creates document records with uploaded files for an application
@router.post("/", response_model=DocumentResponse)
async def create_documents(
    application_id: str,
    profilePhoto: Optional[UploadFile] = File(None),
    validId: Optional[UploadFile] = File(None),
    brgyCert: Optional[UploadFile] = File(None),
    eSignaturePersonal: Optional[UploadFile] = File(None),
    payslip: Optional[UploadFile] = File(None),
    companyId: Optional[UploadFile] = File(None),
    proofOfBilling: Optional[UploadFile] = File(None),
    eSignatureCoMaker: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_active_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key")
):
    files = {
        "profile_photo": profilePhoto,
        "valid_id": validId,
        "brgy_cert": brgyCert,
        "e_signature_personal": eSignaturePersonal,
        "payslip": payslip,
        "company_id": companyId,
        "proof_of_billing": proofOfBilling,
        "e_signature_comaker": eSignatureCoMaker,
    }
    files = {k: v for k, v in files.items() if v}

    # If client supplied an idempotency key, check store for previous result
    if idempotency_key:
        try:
            prev = await _idempotency.get(idempotency_key)
            if prev:
                return prev
        except Exception:
            # best-effort: if idempotency store fails, continue processing
            pass

    doc = await service.create_documents(application_id, files, current_user["email"])

    # Persist response under idempotency key for future dedupe (best-effort)
    if idempotency_key:
        try:
            await _idempotency.set(idempotency_key, doc, ttl_seconds=24 * 3600)
        except Exception:
            pass

    return doc

# Retrieves a document record by its ID
@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, current_user: dict = Depends(get_current_active_user)):
    doc = await service.get_document_by_id(document_id)
    return doc

# Updates existing document fields with new uploaded files
@router.put("/{document_id}", response_model=DocumentResponse)
async def update_documents(
    document_id: str,
    profilePhoto: Optional[UploadFile] = File(None),
    validId: Optional[UploadFile] = File(None),
    brgyCert: Optional[UploadFile] = File(None),
    eSignaturePersonal: Optional[UploadFile] = File(None),
    payslip: Optional[UploadFile] = File(None),
    companyId: Optional[UploadFile] = File(None),
    proofOfBilling: Optional[UploadFile] = File(None),
    eSignatureCoMaker: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_active_user)
):
    files = {
        "profile_photo": profilePhoto,
        "valid_id": validId,
        "brgy_cert": brgyCert,
        "e_signature_personal": eSignaturePersonal,
        "payslip": payslip,
        "company_id": companyId,
        "proof_of_billing": proofOfBilling,
        "e_signature_comaker": eSignatureCoMaker,
    }
    files = {k: v for k, v in files.items() if v}
    doc = await service.update_documents(document_id, files, current_user["email"])
    return doc

# Deletes a document and all its associated files
@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_documents(document_id: str, current_user: dict = Depends(get_current_active_user)):
    await service.delete_documents(document_id)
    return {"detail": "Document deleted"}

# Deletes a specific file field from an application's documents
@router.delete("/application/{application_id}/file/{field}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_application_file(
    application_id: str, 
    field: str, 
    current_user: dict = Depends(get_current_active_user)
):
    doc = await service.get_documents_by_application_id(application_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Application documents not found")

    current_url = getattr(doc, field, None)
    if not current_url:
        return {"detail": f"No {field} to delete"}

    file_path = service._extract_path_from_url(current_url)
    await service._delete_file_from_supabase(file_path)

    setattr(doc, field, None)
    if doc.file_metadata and field in doc.file_metadata:
        del doc.file_metadata[field]
    doc.updated_at = datetime.utcnow()
    doc.updated_by = current_user["email"]
    await doc.save()

    return {"detail": f"{field} deleted"}

# Refreshes all signed URLs for a document by ID
@router.get("/{document_id}/refresh", response_model=DocumentResponse)
async def refresh_signed_urls(document_id: str, current_user: dict = Depends(get_current_active_user)):
    doc = await service.get_document_by_id(document_id)
    return doc

# Uploads or updates documents for an application without creating duplicates
@router.post("/", response_model=DocumentResponse)
async def create_or_update_documents(
    application_id: str,
    profilePhoto: Optional[UploadFile] = File(None),
    validId: Optional[UploadFile] = File(None),
    brgyCert: Optional[UploadFile] = File(None),
    eSignaturePersonal: Optional[UploadFile] = File(None),
    payslip: Optional[UploadFile] = File(None),
    companyId: Optional[UploadFile] = File(None),
    proofOfBilling: Optional[UploadFile] = File(None),
    eSignatureCoMaker: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_active_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key")
):
    files = {
        "profile_photo": profilePhoto,
        "valid_id": validId,
        "brgy_cert": brgyCert,
        "e_signature_personal": eSignaturePersonal,
        "payslip": payslip,
        "company_id": companyId,
        "proof_of_billing": proofOfBilling,
        "e_signature_comaker": eSignatureCoMaker,
    }
    files = {k: v for k, v in files.items() if v}

    # Check idempotency store first
    if idempotency_key:
        try:
            prev = await _idempotency.get(idempotency_key)
            if prev:
                return prev
        except Exception:
            pass

    doc = await service.update_or_create_documents(application_id, files, current_user["email"])

    if idempotency_key:
        try:
            await _idempotency.set(idempotency_key, doc, ttl_seconds=24 * 3600)
        except Exception:
            pass

    return doc

# Refreshes signed URLs for specific or all documents with enhanced cache control
@router.get("/application/{application_id}/refresh-urls")
async def refresh_application_document_urls(
    application_id: str, 
    document_types: Optional[str] = None,
    t: Optional[str] = None,
    nonce: Optional[str] = None,
    v: Optional[str] = None,
    current_user: dict = Depends(get_current_active_user)
):
    from fastapi.responses import JSONResponse
    from fastapi.encoders import jsonable_encoder
    from datetime import datetime, timedelta
    
    requested_types = document_types.split(',') if document_types else None
    
    timestamp = datetime.utcnow()
    response_version = v or timestamp.timestamp()
    
    doc = await service.get_documents_by_application_id(application_id)
    
    if not doc:
        empty_response = {
            "brgy_cert_url": None,
            "e_signature_personal_url": None,
            "payslip_url": None,
            "company_id_url": None,
            "proof_of_billing_url": None,
            "e_signature_comaker_url": None,
            "profile_photo_url": None,
            "valid_id_url": None,
            "_meta": {
                "timestamp": timestamp.isoformat(),
                "version": response_version,
                "status": "no_documents"
            }
        }
        
        headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0, private",
            "Pragma": "no-cache",
            "Expires": "0",
            "Last-Modified": timestamp.strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "ETag": f'W/"{nonce or response_version}"',
            "X-Response-Time": timestamp.isoformat(),
            "X-Cache-Status": "MISS"
        }
        
        return JSONResponse(
            content=jsonable_encoder(empty_response),
            headers=headers
        )
    
    field_mapping = {
        "brgy_cert": "brgy_cert_url",
        "e_signature_personal": "e_signature_personal_url",
        "payslip": "payslip_url",
        "company_id": "company_id_url",
        "proof_of_billing": "proof_of_billing_url",
        "e_signature_comaker": "e_signature_comaker_url",
        "profile_photo": "profile_photo_url",
        "valid_id": "valid_id_url"
    }
    
    response = {}
    
    if requested_types:
        for doc_type in requested_types:
            field_name = field_mapping.get(doc_type)
            if field_name and hasattr(doc, field_name):
                url = getattr(doc, field_name)
                if url:
                    response[field_name] = url
    else:
        for field_name in field_mapping.values():
            if hasattr(doc, field_name):
                url = getattr(doc, field_name)
                if url:
                    response[field_name] = url

    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    
    return JSONResponse(
        content=jsonable_encoder(response),
        headers=headers
    )

# Retrieves all document URLs for an application with cache control
@router.get("/application/{application_id}")
async def get_documents_by_application(application_id: str, current_user: dict = Depends(get_current_active_user)):
    from fastapi.responses import JSONResponse
    from fastapi.encoders import jsonable_encoder

    doc = await service.get_documents_by_application_id(application_id)
    
    if not doc:
        response_data = {
            "brgy_cert_url": None,
            "e_signature_personal_url": None,
            "payslip_url": None,
            "company_id_url": None,
            "proof_of_billing_url": None,
            "e_signature_comaker_url": None,
            "profile_photo_url": None,
            "valid_id_url": None
        }
    else:
        response_data = {
            "brgy_cert_url": doc.brgy_cert_url,
            "e_signature_personal_url": doc.e_signature_personal_url,
            "payslip_url": doc.payslip_url,
            "company_id_url": doc.company_id_url,
            "proof_of_billing_url": doc.proof_of_billing_url,
            "e_signature_comaker_url": doc.e_signature_comaker_url,
            "profile_photo_url": doc.profile_photo_url,
            "valid_id_url": doc.valid_id_url
        }

    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    
    return JSONResponse(
        content=jsonable_encoder(response_data),
        headers=headers
    )