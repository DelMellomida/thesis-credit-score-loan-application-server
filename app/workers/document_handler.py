import logging
import json
from typing import Dict, Optional, Any
from fastapi import UploadFile
from app.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

# Initialize Supabase client once for document operations
supabase_client = get_supabase_client()

async def process_document_updates(
    application_id: str,
    document_files: Dict[str, UploadFile],
    current_user: Dict[str, Any]
) -> Dict[str, Optional[str]]:
    """
    Process document file updates for a loan application.
    Uploads files to Supabase storage and returns the URLs.
    """
    try:
        logger.info(f"Processing document updates for application {application_id}")
        
        document_urls = {}
        storage_client = supabase_client.storage.from_('documents')
        
        for doc_type, file in document_files.items():
            if not file:
                continue
                
            try:
                # Read file content
                file_content = await file.read()
                
                # Generate storage path
                file_name = f"{application_id}/{doc_type}/{file.filename}"
                
                # Upload to Supabase
                response = storage_client.upload(
                    path=file_name,
                    file=file_content,
                    file_options={"content-type": file.content_type}
                )
                
                # Get public URL
                url = storage_client.get_public_url(file_name)
                
                # Map storage field names to database field names
                field_mapping = {
                    'profile_photo': 'profile_photo_url',
                    'valid_id': 'valid_id_url',
                    'brgy_cert': 'brgy_cert_url',
                    'e_signature_personal': 'e_signature_personal_url',
                    'payslip': 'payslip_url',
                    'company_id': 'company_id_url',
                    'proof_of_billing': 'proof_of_billing_url',
                    'e_signature_comaker': 'e_signature_comaker_url'
                }
                
                # Store URL in results
                document_urls[field_mapping[doc_type]] = url
                
                # Store metadata
                document_urls.setdefault('file_metadata', {})
                document_urls['file_metadata'][doc_type] = {
                    'original_name': file.filename,
                    'content_type': file.content_type,
                    'size': len(file_content),
                    'uploaded_by': current_user.get('email'),
                    'upload_timestamp': str(response.get('timeCreated', ''))
                }
                
                logger.info(f"Successfully uploaded {doc_type} for application {application_id}")
                
            except Exception as e:
                logger.error(f"Error uploading {doc_type} for application {application_id}: {e}")
                document_urls[field_mapping[doc_type]] = None
                
            finally:
                await file.seek(0)  # Reset file pointer
        
        return document_urls
        
    except Exception as e:
        logger.error(f"Error processing document updates: {e}")
        return {}