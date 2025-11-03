import logging
import uuid
import os
import asyncio
from datetime import datetime
from typing import Dict, Optional
from fastapi import UploadFile, HTTPException
from app.core.supabase_client import get_supabase_client
from app.database.models.document_model import ApplicationDocument

logger = logging.getLogger(__name__)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
MAX_SIZE = 5 * 1024 * 1024
BUCKET_NAME = "application-documents"

class DocumentService:
    def __init__(self):
        try:
            self.supabase = get_supabase_client()
            if not self.supabase:
                logger.error("Failed to initialize Supabase client")
                raise Exception("Supabase client initialization failed")
                
            # Verify Supabase connection and storage access
            try:
                # Test storage access by trying to create/ensure bucket exists
                try:
                    # First try to create the bucket
                    logger.info(f"Attempting to create storage bucket: {BUCKET_NAME}")
                    self.supabase.storage.create_bucket(BUCKET_NAME, {'public': False})
                    logger.info(f"Successfully created new bucket: {BUCKET_NAME}")
                except Exception as bucket_error:
                    # If bucket already exists, this is fine
                    if "already exists" in str(bucket_error).lower():
                        logger.info(f"Bucket {BUCKET_NAME} already exists")
                    else:
                        # For other errors, log but don't fail yet
                        logger.warning(f"Note: Bucket creation returned: {str(bucket_error)}")
                
                # Verify we can list the bucket contents
                try:
                    # Test if we can list files (even if empty)
                    self.supabase.storage.from_(BUCKET_NAME).list()
                    logger.info(f"Successfully verified access to bucket: {BUCKET_NAME}")
                except Exception as list_error:
                    logger.error(f"Cannot access bucket {BUCKET_NAME}: {str(list_error)}")
                    raise Exception(f"Cannot access storage bucket: {str(list_error)}")
                    
            except Exception as storage_error:
                logger.error(f"Failed to verify Supabase storage access: {str(storage_error)}")
                raise Exception(f"Supabase storage configuration error: {str(storage_error)}")
                
        except Exception as e:
            logger.error(f"Error initializing document service: {str(e)}")
            raise

    async def create_documents(self, application_id: str, files: Dict[str, UploadFile], user_email: str) -> ApplicationDocument:
        logger.info(f"Creating documents for application {application_id} by {user_email}")

        # CRITICAL: Check if document already exists first
        existing_doc = await ApplicationDocument.find_one(ApplicationDocument.application_id == application_id)
        if existing_doc:
            logger.warning(f"Document already exists for application {application_id}, updating instead of creating")
            # Call update instead to avoid duplicates
            file_metadata = existing_doc.file_metadata or {}
            
            for field, file in files.items():
                # Delete old file if it exists
                old_url = getattr(existing_doc, f"{field}_url", None)
                if old_url:
                    try:
                        path = self._extract_path_from_url(old_url)
                        await self._delete_file_from_supabase(path)
                        logger.info(f"Deleted old file for {field}: {path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete old file for {field}: {str(e)}")
                
                # Upload new file
                safe_field = field.replace(" ", "_").lower()
                filename = self._generate_unique_filename(file.filename)
                file_path = f"{existing_doc.application_id}/{safe_field}_{filename}"
                
                uploaded_path, size = await self._upload_file_to_supabase(file, file_path)
                signed_url = await self._generate_signed_url(uploaded_path)
                
                # Update document with new URL
                setattr(existing_doc, f"{field}_url", signed_url)
                file_metadata[safe_field] = {
                    "filename": file.filename,
                    "size": str(size),
                    "content_type": file.content_type,
                    "uploaded_at": datetime.utcnow().isoformat()
                }
                logger.info(f"Updated {field} for application {application_id} by {user_email} (size={size}, content_type={file.content_type})")
            
            # Save updated document
            existing_doc.updated_at = datetime.utcnow()
            existing_doc.updated_by = user_email
            existing_doc.file_metadata = file_metadata
            
            try:
                await existing_doc.save()
                logger.info(f"Successfully saved document updates for application {application_id}")
            except Exception as e:
                logger.error(f"Failed to save document updates: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to save document updates: {str(e)}")
            
            return existing_doc
        
        # Input validation with detailed logging
        if not application_id:
            logger.error("Invalid application_id: empty or None")
            raise HTTPException(status_code=400, detail="Invalid application ID")
            
        if not files:
            logger.error("No files provided for document creation")
            raise HTTPException(status_code=400, detail="No files provided")
            
        # Log file details for debugging
        logger.info(f"Received {len(files)} files for upload:")
        for field, file in files.items():
            logger.info(f"  - {field}: filename='{file.filename}', content_type='{file.content_type}'")
            
        # Verify Supabase client
        if not self.supabase:
            logger.error("Supabase client is not initialized")
            raise HTTPException(status_code=500, detail="Storage service not available")

        uploaded_paths = {}  # Store file paths for cleanup
        file_urls = {}  # Store signed URLs for database
        file_metadata = {}
        
        try:
            # Pre-validate all files before uploading
            for field, file in files.items():
                try:
                    await self._validate_file(file)
                except HTTPException as e:
                    logger.error(f"File validation failed for {field}: {e.detail}")
                    raise

            # Upload files to Supabase with retry logic
            logger.info(f"Starting file uploads for application {application_id}")
            total_files = len(files)
            current_file = 0
            
            for field, file in files.items():
                current_file += 1
                max_upload_retries = 3
                base_delay = 1.0  # seconds
                
                logger.info(f"Processing file {current_file}/{total_files}: {field} ({file.filename})")
                
                for attempt in range(max_upload_retries):
                    try:
                        # Generate safe filename and path
                        safe_field = field.replace(" ", "_").lower()
                        filename = self._generate_unique_filename(file.filename)
                        file_path = f"{application_id}/{safe_field}_{filename}"
                        
                        logger.debug(f"Generated storage path for {field}: {file_path}")
                        
                        # Ensure the file is ready for upload
                        await file.seek(0)
                        
                        # Attempt upload - returns (file_path, size)
                        uploaded_path, size = await self._upload_file_to_supabase(file, file_path)
                        
                        if not uploaded_path:
                            raise HTTPException(
                                status_code=500, 
                                detail="Upload succeeded but no path returned"
                            )
                        
                        # Store the uploaded path for potential cleanup
                        uploaded_paths[safe_field] = uploaded_path
                        
                        # Generate signed URL for the uploaded file
                        signed_url = await self._generate_signed_url(uploaded_path)
                        if not signed_url:
                            raise HTTPException(
                                status_code=500,
                                detail="Failed to generate signed URL"
                            )
                        
                        # Store URL and metadata
                        url_field = f"{safe_field}_url"
                        file_urls[url_field] = signed_url
                        file_metadata[safe_field] = {
                            "filename": file.filename,
                            "size": str(size),
                            "content_type": file.content_type,
                            "uploaded_at": datetime.utcnow().isoformat()
                        }
                        
                        logger.info(
                            f"Successfully uploaded {field} ({size} bytes, {file.content_type}) "
                            f"to {uploaded_path} for application {application_id}"
                        )
                        break  # Success - exit retry loop
                        
                    except Exception as e:
                        delay = base_delay * (2 ** attempt)
                        
                        if attempt == max_upload_retries - 1:  # Final attempt failed
                            logger.error(
                                f"All upload attempts failed for {field} in application "
                                f"{application_id}: {str(e)}"
                            )
                            # Cleanup already uploaded files
                            await self._cleanup_uploaded_files(uploaded_paths)
                            raise HTTPException(
                                status_code=500,
                                detail=f"Failed to upload {field} after {max_upload_retries} attempts: {str(e)}"
                            )
                            
                        logger.warning(
                            f"Upload attempt {attempt + 1}/{max_upload_retries} failed for {field} "
                            f"in application {application_id}: {str(e)}. Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)

            # Validate required fields are present
            if not any(file_urls.values()):
                raise ValueError("No valid file URLs found after upload")

            # Create the document with explicit None values for missing fields
            doc_data = {
                "application_id": application_id,
                "brgy_cert_url": file_urls.get("brgy_cert_url"),
                "e_signature_personal_url": file_urls.get("e_signature_personal_url"),
                "payslip_url": file_urls.get("payslip_url"),
                "company_id_url": file_urls.get("company_id_url"),
                "proof_of_billing_url": file_urls.get("proof_of_billing_url"),
                "e_signature_comaker_url": file_urls.get("e_signature_comaker_url"),
                "profile_photo_url": file_urls.get("profile_photo_url"),
                "valid_id_url": file_urls.get("valid_id_url"),
                "file_metadata": file_metadata,
                "created_by": user_email,
                "updated_by": user_email,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            # Log the document data for debugging
            logger.debug(f"Creating document with data: {doc_data}")
            
            # Log document creation attempt
            logger.info(
                f"Attempting to create document record for application {application_id} "
                f"with {len(file_urls)} files: {', '.join(file_urls.keys())}"
            )

            try:
                # Create and validate document instance
                doc = ApplicationDocument(**doc_data)
                
                # Attempt to insert with improved retry logic
                max_retries = 3
                base_delay = 1.0  # seconds
                
                for attempt in range(max_retries):
                    try:
                        await doc.insert()
                        logger.info(
                            f"Document created successfully for application {application_id} "
                            f"with {len(file_urls)} files"
                        )
                        return doc
                        
                    except Exception as insert_error:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        
                        if attempt == max_retries - 1:  # Last attempt
                            logger.error(
                                f"Final document insert attempt failed for {application_id}: "
                                f"{str(insert_error)}"
                            )
                            # Cleanup uploaded files since we can't save metadata
                            await self._cleanup_uploaded_files(uploaded_paths)
                            raise HTTPException(
                                status_code=500,
                                detail=f"Failed to save document metadata: {str(insert_error)}"
                            )
                            
                        logger.warning(
                            f"Document insert attempt {attempt + 1}/{max_retries} failed "
                            f"for {application_id}: {str(insert_error)}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                        
            except Exception as e:
                logger.error(f"Failed to create document record: {str(e)}")
                await self._cleanup_uploaded_files(uploaded_paths)
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to create document record: {str(e)}"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in document creation: {str(e)}")
            await self._cleanup_uploaded_files(uploaded_paths)
            raise HTTPException(
                status_code=500, 
                detail=f"Document creation failed: {str(e)}"
            )

    async def get_documents_by_application_id(self, application_id: str) -> Optional[ApplicationDocument]:
        """Get documents by application_id with refreshed signed URLs"""
        doc = await ApplicationDocument.find_one(ApplicationDocument.application_id == application_id)
        if not doc:
            # Return None instead of creating empty document
            logger.info(f"No documents found for application {application_id}")
            return None
            
        # Refresh signed URLs for existing documents
        for field in ["brgy_cert_url", "e_signature_personal_url", "payslip_url", "company_id_url", "proof_of_billing_url", "e_signature_comaker_url", "profile_photo_url", "valid_id_url"]:
            url = getattr(doc, field)
            if url:
                try:
                    path = self._extract_path_from_url(url)
                    signed_url = await self._generate_signed_url(path)
                    setattr(doc, field, signed_url)
                except Exception as e:
                    logger.warning(f"Failed to refresh signed URL for {field}: {str(e)}")
        return doc

    async def get_document_by_id(self, document_id: str) -> Optional[ApplicationDocument]:
        doc = await ApplicationDocument.find_one(ApplicationDocument.document_id == document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        # Refresh signed URLs
        for field in ["brgy_cert_url", "e_signature_personal_url", "payslip_url", "company_id_url", "proof_of_billing_url", "e_signature_comaker_url", "profile_photo_url", "valid_id_url"]:
            url = getattr(doc, field)
            if url:
                try:
                    path = self._extract_path_from_url(url)
                    signed_url = await self._generate_signed_url(path)
                    setattr(doc, field, signed_url)
                except Exception as e:
                    logger.warning(f"Failed to refresh signed URL for {field}: {str(e)}")
        return doc

    async def update_documents(self, document_id: str, files: Dict[str, UploadFile], user_email: str) -> ApplicationDocument:
        doc = await ApplicationDocument.find_one(ApplicationDocument.document_id == document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        file_metadata = doc.file_metadata or {}
        
        for field, file in files.items():
            old_url = getattr(doc, f"{field}_url", None)
            if old_url:
                # Extract path from URL before deleting
                try:
                    path = self._extract_path_from_url(old_url)
                    await self._delete_file_from_supabase(path)
                except Exception as e:
                    logger.warning(f"Failed to delete old file for {field}: {str(e)}")
            
            # Upload new file
            safe_field = field.replace(" ", "_").lower()
            filename = self._generate_unique_filename(file.filename)
            file_path = f"{doc.application_id}/{safe_field}_{filename}"
            
            uploaded_path, size = await self._upload_file_to_supabase(file, file_path)
            signed_url = await self._generate_signed_url(uploaded_path)
            
            setattr(doc, f"{field}_url", signed_url)
            file_metadata[field] = {
                "filename": file.filename,
                "size": str(size),
                "content_type": file.content_type,
                "uploaded_at": datetime.utcnow().isoformat()
            }
            logger.info(f"Updated {field} for document {document_id} by {user_email} (size={size}, content_type={file.content_type})")
        
        doc.updated_at = datetime.utcnow()
        doc.updated_by = user_email
        doc.file_metadata = file_metadata
        await doc.save()
        return doc

    async def delete_documents(self, document_id: str):
        doc = await ApplicationDocument.find_one(ApplicationDocument.document_id == document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        for field in ["brgy_cert_url", "e_signature_personal_url", "payslip_url", "company_id_url", "proof_of_billing_url", "e_signature_comaker_url"]:
            url = getattr(doc, field)
            if url:
                try:
                    path = self._extract_path_from_url(url)
                    await self._delete_file_from_supabase(path)
                except Exception as e:
                    logger.warning(f"Failed to delete file for {field}: {str(e)}")
        
        await doc.delete()
        logger.info(f"Deleted document {document_id}")

    async def delete_single_file(self, document_id: str, field_name: str):
        doc = await ApplicationDocument.find_one(ApplicationDocument.document_id == document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        url = getattr(doc, f"{field_name}_url", None)
        if not url:
            raise HTTPException(status_code=404, detail="File not found")
        
        path = self._extract_path_from_url(url)
        await self._delete_file_from_supabase(path)
        
        setattr(doc, f"{field_name}_url", None)
        file_metadata = doc.file_metadata or {}
        file_metadata.pop(field_name, None)
        doc.file_metadata = file_metadata
        doc.updated_at = datetime.utcnow()
        await doc.save()
        logger.info(f"Deleted {field_name} from document {document_id}")

    async def _upload_file_to_supabase(self, file: UploadFile, file_path: str) -> tuple:
        """Upload the file and return (file_path, size).

        This reads the UploadFile contents once, infers MIME type from magic bytes if necessary,
        and returns the uploaded path and the byte size.
        """
        await file.seek(0)
        contents = await file.read()
        size = len(contents)
            
        logger.debug(
            f"Preparing to upload file: name='{file.filename}', "
            f"size={size}, content_type='{file.content_type}'"
        )
            
        if size == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
                
        # Resolve content type: prefer provided, else infer from magic bytes or extension
        content_type = file.content_type or "application/octet-stream"
            
        # Log Supabase client status
        if not self.supabase:
            logger.error("Supabase client is None during upload attempt")
            raise HTTPException(status_code=500, detail="Storage service unavailable")

        # Magic bytes detection for common image types
        if content_type == "text/plain" or content_type == "application/octet-stream":
            header = contents[:12]
            if header.startswith(b"\xff\xd8\xff"):
                content_type = "image/jpeg"
            elif header.startswith(b"\x89PNG\r\n\x1a\n"):
                content_type = "image/png"
            elif header[0:4] == b"RIFF" and header[8:12] == b"WEBP":
                content_type = "image/webp"
            else:
                # fallback to extension based detection
                ext = os.path.splitext(file.filename)[1].lower()
                if ext in [".jpg", ".jpeg"]:
                    content_type = "image/jpeg"
                elif ext == ".png":
                    content_type = "image/png"
                elif ext == ".webp":
                    content_type = "image/webp"
                else:
                    content_type = "application/octet-stream"

        # Validate resolved content type
        if content_type not in ALLOWED_TYPES:
            logger.warning(f"Resolved invalid file type for {file.filename}: {content_type}")
            raise HTTPException(status_code=400, detail=f"Invalid file type: {content_type}")
        
        try:
            # Try to pass the content-type explicitly so Supabase accepts the file
            from io import BytesIO
            
            # Verify bucket exists
            try:
                bucket_info = self.supabase.storage.get_bucket(BUCKET_NAME)
                logger.debug(f"Found bucket: {BUCKET_NAME}")
            except Exception as e:
                logger.error(f"Error verifying bucket {BUCKET_NAME}: {str(e)}")
                try:
                    # Attempt to create bucket if it doesn't exist
                    self.supabase.storage.create_bucket(BUCKET_NAME, {'public': False})
                    logger.info(f"Created new bucket: {BUCKET_NAME}")
                except Exception as create_error:
                    logger.error(f"Failed to create bucket: {str(create_error)}")
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Storage bucket configuration error: {str(create_error)}"
                    )

            # Try different upload methods with explicit error handling
            upload_methods = [
                {
                    "method": lambda: self.supabase.storage.from_(BUCKET_NAME).upload(
                        file_path,
                        contents,
                        {"content-type": content_type}
                    ),
                    "description": "Direct upload with content-type"
                },
                {
                    "method": lambda: self.supabase.storage.from_(BUCKET_NAME).upload(
                        file_path,
                        BytesIO(contents),
                        {"content-type": content_type}
                    ),
                    "description": "BytesIO upload with content-type"
                },
                {
                    "method": lambda: self.supabase.storage.from_(BUCKET_NAME).upload(
                        file_path, 
                        contents
                    ),
                    "description": "Direct upload without content-type"
                },
                {
                    "method": lambda: self.supabase.storage.from_(BUCKET_NAME).upload(
                        file_path,
                        BytesIO(contents)
                    ),
                    "description": "BytesIO upload without content-type"
                }
            ]
            
            last_error = None
            for idx, upload_config in enumerate(upload_methods, 1):
                try:
                    logger.debug(
                        f"Trying upload method {idx}/{len(upload_methods)}: "
                        f"{upload_config['description']}"
                    )
                    res = upload_config["method"]()
                    
                    if res and not (isinstance(res, dict) and res.get("error")):
                        logger.info(
                            f"Successfully uploaded {file.filename} to {file_path} "
                            f"using method: {upload_config['description']}"
                        )
                        # IMPORTANT: Return here on success
                        return file_path, size
                    else:
                        error_msg = f"Upload failed with result: {res}"
                        logger.warning(
                            f"Method {idx} ({upload_config['description']}) "
                            f"returned unexpected result: {error_msg}"
                        )
                        last_error = error_msg
                except Exception as e:
                    error_msg = f"Upload attempt {idx} failed: {str(e)}"
                    logger.warning(error_msg)
                    last_error = error_msg
                    continue
                    
            # If we get here, all attempts failed
            error_msg = f"All upload attempts failed for {file.filename}. Last error: {last_error}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Supabase upload error for file {file.filename} (content_type={file.content_type}): {e}")
            raise HTTPException(status_code=500, detail=f"Supabase upload failed: {e}")

    async def _delete_file_from_supabase(self, file_path: str):
        """Delete a file from Supabase storage
        
        Args:
            file_path: The storage path (not URL) of the file to delete
        """
        try:
            result = self.supabase.storage.from_(BUCKET_NAME).remove([file_path])
            logger.info(f"Deleted file {file_path} from Supabase")
            return result
        except Exception as e:
            logger.error(f"Supabase delete error for {file_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

    async def _cleanup_uploaded_files(self, file_paths: Dict[str, str]):
        """Helper method to clean up uploaded files in case of failure
        
        Args:
            file_paths: Dictionary mapping field names to storage paths
        """
        if not file_paths:
            return
            
        logger.info(f"Cleaning up {len(file_paths)} uploaded files")
        for field, path in file_paths.items():
            try:
                await self._delete_file_from_supabase(path)
                logger.info(f"Cleaned up file for field {field}: {path}")
            except Exception as e:
                logger.error(f"Error cleaning up file {path} for field {field}: {str(e)}")
                # Continue cleanup even if one file fails

    async def _generate_signed_url(self, file_path: str) -> str:
        """Generate a signed URL for accessing a file with shorter expiry
        
        Args:
            file_path: The storage path of the file
            
        Returns:
            Signed URL string with cache control parameters
        """
        try:
            # Generate URL with 15-minute expiry
            res = self.supabase.storage.from_(BUCKET_NAME).create_signed_url(
                file_path, 
                900,  # 15 minutes in seconds
                {
                    # Add cache control to response headers
                    'response-cache-control': 'no-cache, no-store, must-revalidate, max-age=0',
                    'response-expires': '0',
                    'response-pragma': 'no-cache'
                }
            )
            signed_url = res.get("signedURL")
            
            if not signed_url:
                logger.error(f"No signed URL returned for path: {file_path}")
                raise HTTPException(status_code=500, detail="Failed to generate signed URL")

            # Add timestamp to URL to prevent caching
            url = signed_url + (
                "&" if "?" in signed_url else "?"
            ) + f"t={int(datetime.utcnow().timestamp())}"
                
            logger.debug(f"Generated cache-controlled signed URL for {file_path}")
            return url
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Supabase signed URL error for {file_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate signed URL: {str(e)}")

    def _extract_path_from_url(self, url: str) -> str:
        """Extract the file path from a Supabase URL
        
        Args:
            url: Full Supabase URL or just the path
            
        Returns:
            File path without URL parameters
        """
        # If it's already a path (not a full URL), return as-is
        if not url.startswith("http"):
            return url
            
        # Extract path from URL
        # Format: https://xxx.supabase.co/storage/v1/object/sign/bucket-name/path?token=xxx
        try:
            # Split by bucket name to get the path
            parts = url.split(f"/{BUCKET_NAME}/")
            if len(parts) > 1:
                # Remove query parameters
                path = parts[1].split("?")[0]
                return path
            else:
                # Fallback: just return the last part after 'sign/'
                path = url.split("/sign/")[-1].split("?")[0]
                # Remove bucket name if present at start
                if path.startswith(f"{BUCKET_NAME}/"):
                    path = path[len(BUCKET_NAME)+1:]
                return path
        except Exception as e:
            logger.warning(f"Could not extract path from URL {url}: {str(e)}")
            return url

    async def _validate_file(self, file: UploadFile):
        """Validate file type and size"""
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="Invalid file")
            
        # Reset cursor and read content for validation
        try:
            await file.seek(0)
            contents = await file.read()
            size = len(contents)
            
            # Reset cursor for future reads
            await file.seek(0)
            
            if size == 0:
                raise HTTPException(status_code=400, detail="Empty file uploaded")
                
            if size > MAX_SIZE:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File size {size} exceeds limit of {MAX_SIZE} bytes"
                )
                
            # Detect content type from magic bytes
            content_type = file.content_type or "application/octet-stream"
            if content_type == "application/octet-stream":
                header = contents[:12]
                if header.startswith(b"\xff\xd8\xff"):
                    content_type = "image/jpeg"
                elif header.startswith(b"\x89PNG\r\n\x1a\n"):
                    content_type = "image/png"
                elif header[0:4] == b"RIFF" and header[8:12] == b"WEBP":
                    content_type = "image/webp"
                else:
                    ext = os.path.splitext(file.filename)[1].lower()
                    if ext in [".jpg", ".jpeg"]:
                        content_type = "image/jpeg"
                    elif ext == ".png":
                        content_type = "image/png"
                    elif ext == ".webp":
                        content_type = "image/webp"
            
            if content_type not in ALLOWED_TYPES:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File type {content_type} not allowed. Allowed types: {', '.join(ALLOWED_TYPES)}"
                )
                
            # # Set the detected content type
            # file.content_type = content_type
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"File validation error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"File validation failed: {str(e)}")

    def _generate_unique_filename(self, original_filename: str) -> str:
        ext = os.path.splitext(original_filename)[1]
        return f"{uuid.uuid4()}{ext}"
    
    async def update_or_create_documents(self, application_id: str, files: Dict[str, UploadFile], user_email: str) -> ApplicationDocument:
        """
        Update existing document or create new one if it doesn't exist.
        This ensures we never create duplicate documents for the same application.
        """
        # Try to find existing document
        doc = await ApplicationDocument.find_one(ApplicationDocument.application_id == application_id)
        
        if doc:
            logger.info(f"Found existing document for application {application_id}, updating...")
            file_metadata = doc.file_metadata or {}
            
            for field, file in files.items():
                # Delete old file if it exists
                old_url = getattr(doc, f"{field}_url", None)
                if old_url:
                    try:
                        path = self._extract_path_from_url(old_url)
                        await self._delete_file_from_supabase(path)
                        logger.info(f"Deleted old file for {field}: {path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete old file for {field}: {str(e)}")
                
                # Upload new file
                safe_field = field.replace(" ", "_").lower()
                filename = self._generate_unique_filename(file.filename)
                file_path = f"{doc.application_id}/{safe_field}_{filename}"
                
                uploaded_path, size = await self._upload_file_to_supabase(file, file_path)
                signed_url = await self._generate_signed_url(uploaded_path)
                
                # Update document with new URL
                setattr(doc, f"{field}_url", signed_url)
                file_metadata[safe_field] = {
                    "filename": file.filename,
                    "size": str(size),
                    "content_type": file.content_type,
                    "uploaded_at": datetime.utcnow().isoformat()
                }
                logger.info(f"Updated {field} for application {application_id} by {user_email} (size={size}, content_type={file.content_type})")
            
            # Save updated document
            doc.updated_at = datetime.utcnow()
            doc.updated_by = user_email
            doc.file_metadata = file_metadata
            
            try:
                await doc.save()
                logger.info(f"Successfully saved document updates for application {application_id}")
            except Exception as e:
                logger.error(f"Failed to save document updates: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to save document updates: {str(e)}")
            
            return doc
        else:
            # No existing document, create new one
            logger.info(f"No existing document found for application {application_id}, creating new one")
            return await self.create_documents(application_id, files, user_email)