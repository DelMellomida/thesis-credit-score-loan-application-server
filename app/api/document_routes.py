from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from typing import Optional, List
from datetime import datetime
from app.services.document_service import DocumentService
from app.schemas.document_schema import DocumentUploadRequest, DocumentResponse
from app.core.auth_dependencies import get_current_active_user

router = APIRouter(prefix="/documents", tags=["Documents"])
service = DocumentService()

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
    doc = await service.create_documents(application_id, files, current_user["email"])
    return doc

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, current_user: dict = Depends(get_current_active_user)):
    doc = await service.get_document_by_id(document_id)
    return doc

# @router.get("/application/{application_id}")
# async def get_documents_by_application(application_id: str, current_user: dict = Depends(get_current_active_user)):
#     # Add cache control headers for signed URLs
#     from fastapi.responses import JSONResponse
#     from fastapi.encoders import jsonable_encoder

#     doc = await service.get_documents_by_application_id(application_id)
#     response_data = {
#         "brgy_cert_url": doc.brgy_cert_url,
#         "e_signature_personal_url": doc.e_signature_personal_url,
#         "payslip_url": doc.payslip_url,
#         "company_id_url": doc.company_id_url,
#         "proof_of_billing_url": doc.proof_of_billing_url,
#         "e_signature_comaker_url": doc.e_signature_comaker_url,
#         "profile_photo_url": doc.profile_photo_url,
#         "valid_id_url": doc.valid_id_url
#     }

#     headers = {
#         "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
#         "Pragma": "no-cache",
#         "Expires": "0"
#     }
    
#     return JSONResponse(
#         content=jsonable_encoder(response_data),
#         headers=headers
#     )

# @router.get("/application/{application_id}/refresh-urls")
# async def refresh_application_document_urls(
#     application_id: str, 
#     document_types: Optional[List[str]] = None,
#     current_user: dict = Depends(get_current_active_user)
# ):
#     """
#     Refresh signed URLs for an application's documents.
#     Optionally specify document types to refresh only specific URLs.
#     """
#     from fastapi.responses import JSONResponse
#     from fastapi.encoders import jsonable_encoder
    
#     doc = await service.get_documents_by_application_id(application_id)
#     # If document_types specified, only return those URLs
#     response = {}
#     all_fields = [
#         "brgy_cert_url", "e_signature_personal_url", "payslip_url",
#         "company_id_url", "proof_of_billing_url", "e_signature_comaker_url",
#         "profile_photo_url", "valid_id_url"
#     ]
    
#     fields_to_refresh = document_types if document_types else all_fields
    
#     # Validate requested document types
#     if document_types:
#         invalid_types = [t for t in document_types if f"{t}_url" not in all_fields]
#         if invalid_types:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Invalid document types: {', '.join(invalid_types)}"
#             )
    
#     # Build response with only requested/available URLs
#     for field in fields_to_refresh:
#         url_field = f"{field}_url" if not field.endswith("_url") else field
#         if hasattr(doc, url_field):
#             response[url_field] = getattr(doc, url_field)

#     # Add cache control headers
#     headers = {
#         "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
#         "Pragma": "no-cache",
#         "Expires": "0"
#     }
    
#     return JSONResponse(
#         content=jsonable_encoder(response),
#         headers=headers
#     )

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

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_documents(document_id: str, current_user: dict = Depends(get_current_active_user)):
    await service.delete_documents(document_id)
    return {"detail": "Document deleted"}

@router.delete("/application/{application_id}/file/{field}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_application_file(
    application_id: str, 
    field: str, 
    current_user: dict = Depends(get_current_active_user)
):
    """Delete a specific file from an application's documents"""
    # Get the document record for this application
    doc = await service.get_documents_by_application_id(application_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Application documents not found")

    # Get the current file path from the URL
    current_url = getattr(doc, field, None)
    if not current_url:
        # If no file exists, just return success
        return {"detail": f"No {field} to delete"}

    # Delete the file from storage
    file_path = service._extract_path_from_url(current_url)
    await service._delete_file_from_supabase(file_path)

    # Update the document record
    setattr(doc, field, None)
    if doc.file_metadata and field in doc.file_metadata:
        del doc.file_metadata[field]
    doc.updated_at = datetime.utcnow()
    doc.updated_by = current_user["email"]
    await doc.save()

    return {"detail": f"{field} deleted"}

@router.get("/{document_id}/refresh", response_model=DocumentResponse)
async def refresh_signed_urls(document_id: str, current_user: dict = Depends(get_current_active_user)):
    doc = await service.get_document_by_id(document_id)
    return doc


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
    current_user: dict = Depends(get_current_active_user)
):
    """
    Upload or update documents for an application.
    Always updates existing document if found, never creates duplicates.
    """
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
    
    # Always try to update - the service will create if needed
    doc = await service.update_or_create_documents(application_id, files, current_user["email"])
    
    return doc

@router.get("/application/{application_id}/refresh-urls")
async def refresh_application_document_urls(
    application_id: str, 
    document_types: Optional[str] = None,  # Changed to string to handle query param properly
    current_user: dict = Depends(get_current_active_user)
):
    """
    Refresh signed URLs for an application's documents.
    Optionally specify document types (comma-separated) to refresh only specific URLs.
    """
    from fastapi.responses import JSONResponse
    from fastapi.encoders import jsonable_encoder
    
    # Parse document_types if provided
    requested_types = document_types.split(',') if document_types else None
    
    doc = await service.get_documents_by_application_id(application_id)
    
    if not doc:
        # Return empty URLs if no document exists
        return JSONResponse(
            content={
                "brgy_cert_url": None,
                "e_signature_personal_url": None,
                "payslip_url": None,
                "company_id_url": None,
                "proof_of_billing_url": None,
                "e_signature_comaker_url": None,
                "profile_photo_url": None,
                "valid_id_url": None
            },
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    
    # Map of document types to field names
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
    
    # Build response with only requested/available URLs
    response = {}
    
    if requested_types:
        # Only return requested document types
        for doc_type in requested_types:
            field_name = field_mapping.get(doc_type)
            if field_name and hasattr(doc, field_name):
                url = getattr(doc, field_name)
                if url:
                    response[field_name] = url
    else:
        # Return all URLs
        for field_name in field_mapping.values():
            if hasattr(doc, field_name):
                url = getattr(doc, field_name)
                if url:
                    response[field_name] = url

    # Add cache control headers
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    
    return JSONResponse(
        content=jsonable_encoder(response),
        headers=headers
    )

@router.get("/application/{application_id}")
async def get_documents_by_application(application_id: str, current_user: dict = Depends(get_current_active_user)):
    """Get all document URLs for an application"""
    from fastapi.responses import JSONResponse
    from fastapi.encoders import jsonable_encoder

    doc = await service.get_documents_by_application_id(application_id)
    
    # Return empty URLs if no document exists
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