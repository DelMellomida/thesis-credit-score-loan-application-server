import logging
import asyncio
from typing import Dict, Optional
from fastapi import UploadFile, HTTPException
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)

# Read uploaded file in chunks and return complete contents
async def read_file_chunks(file: UploadFile) -> bytes:
    try:
        contents = b""
        chunk_size = 1024 * 1024
        while chunk := await file.read(chunk_size):
            contents += chunk
        await file.seek(0)
        return contents
    except Exception as e:
        logger.error(f"Error reading file {file.filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Could not read file {file.filename}")

# Validate that a file is readable and not empty
async def validate_file(file: UploadFile) -> bool:
    try:
        if not file.filename:
            raise ValueError("No filename provided")
            
        await file.seek(0)
        chunk = await file.read(1024)
        if not chunk:
            raise ValueError("File is empty")
            
        await file.seek(0)
        return True
    except Exception as e:
        logger.error(f"File validation error for {file.filename}: {str(e)}")
        return False

# Process and upload application documents with validation and retry logic
async def handle_application_documents(application_id: str, files: Dict[str, Optional[UploadFile]], user_email: str):
    if not application_id:
        raise HTTPException(status_code=400, detail="Application ID is required")
    if not user_email:
        raise HTTPException(status_code=400, detail="User email is required")

    document_service = DocumentService()
    
    valid_files = {}
    total_size = 0
    max_total_size = 25 * 1024 * 1024
    
    for field, file in files.items():
        if not file:
            continue
            
        try:
            if not await validate_file(file):
                logger.warning(f"Skipping invalid file for field {field}")
                continue
                
            contents = await read_file_chunks(file)
            size = len(contents)
            
            if size > 5 * 1024 * 1024:
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
    
    max_retries = 3
    retry_delay = 1
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