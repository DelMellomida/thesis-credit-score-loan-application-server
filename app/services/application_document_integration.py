import logging
import asyncio
from typing import Dict, Optional
from fastapi import UploadFile, HTTPException
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)

async def read_file_chunks(file: UploadFile) -> bytes:
    """Read file in chunks and return complete contents"""
    try:
        contents = b""
        chunk_size = 1024 * 1024  # 1MB chunks
        while chunk := await file.read(chunk_size):
            contents += chunk
        await file.seek(0)  # Reset file pointer
        return contents
    except Exception as e:
        logger.error(f"Error reading file {file.filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Could not read file {file.filename}")

async def validate_file(file: UploadFile) -> bool:
    """Validate a single file"""
    try:
        if not file.filename:
            raise ValueError("No filename provided")
            
        # Read a small chunk to verify file is readable
        await file.seek(0)
        chunk = await file.read(1024)
        if not chunk:
            raise ValueError("File is empty")
            
        await file.seek(0)
        return True
    except Exception as e:
        logger.error(f"File validation error for {file.filename}: {str(e)}")
        return False

async def handle_application_documents(application_id: str, files: Dict[str, Optional[UploadFile]], user_email: str):
    # Validate required parameters
    if not application_id:
        raise HTTPException(status_code=400, detail="Application ID is required")
    if not user_email:
        raise HTTPException(status_code=400, detail="User email is required")

    # Initialize services
    document_service = DocumentService()
    
    # Process and validate files
    valid_files = {}
    total_size = 0
    max_total_size = 25 * 1024 * 1024  # 25MB total limit
    
    for field, file in files.items():
        if not file:
            continue
            
        try:
            # Basic validation
            if not await validate_file(file):
                logger.warning(f"Skipping invalid file for field {field}")
                continue
                
            # Read and validate file contents
            contents = await read_file_chunks(file)
            size = len(contents)
            
            # Check individual and total size limits
            if size > 5 * 1024 * 1024:  # 5MB per file
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} exceeds 5MB limit"
                )
                
            total_size += size
            if total_size > max_total_size:
                raise HTTPException(
                    status_code=400,
                    detail="Total file size exceeds 25MB limit"
                )
                
            valid_files[field] = file
            logger.info(f"Validated file {file.filename} for field {field} ({size} bytes)")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing file {field}: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error processing file for {field}: {str(e)}"
            )
    
    if not valid_files:
        logger.info("No valid files to process")
        return None
    
    # FIXED: This section should NOT be indented under the return statement above
    # Upload and create documents with retries
    max_retries = 3
    retry_delay = 1  # seconds
    attempt = 0
    
    while True:
        try:
            document_result = await document_service.create_documents(
                application_id=application_id,
                files=valid_files,
                user_email=user_email
            )
            
            if not document_result:
                raise HTTPException(
                    status_code=500,
                    detail="Document creation succeeded but no result returned"
                )
                
            logger.info(
                f"Documents created successfully for application {application_id}: "
                f"{', '.join(f.filename for f in valid_files.values())}"
            )
            return document_result
            
        except HTTPException as http_exc:
            attempt += 1
            if attempt >= max_retries or http_exc.status_code < 500:
                # Don't retry client errors or if max retries reached
                logger.error(f"Document service error: {http_exc.detail}")
                raise
                
            logger.warning(
                f"Retrying document creation after error: {http_exc.detail} "
                f"(attempt {attempt}/{max_retries})"
            )
            await asyncio.sleep(retry_delay * attempt)
            
        except Exception as e:
            logger.error(f"Unexpected error in document creation: {str(e)}")
            raise HTTPException(status_code=500, detail="Document creation failed")