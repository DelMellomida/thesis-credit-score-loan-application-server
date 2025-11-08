import logging
import asyncio
from typing import Dict, Any, Optional
from app.database.models.loan_application_model import LoanApplication, ApplicantInfo as DbApplicantInfo, CoMakerInfo as DbCoMakerInfo, ModelInputData
from app.services.loan_service import LoanApplicationService
from app.services.application_document_integration import handle_application_documents
from app.utils.loan_application_utils import (
    convert_applicant_info,
    convert_comaker_info,
    convert_model_input_data
)
from app.helpers.response_builder import build_loan_application_response
from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)

async def process_loan_application(
    request_data,
    document_files: Dict[str, Optional[UploadFile]],
    current_user: Dict,
    service: LoanApplicationService
) -> Dict[str, Any]:
    """
    Orchestrates the full loan application creation process.
    """
    try:
        logger.info(f"Starting loan application creation process for user: {current_user['email']}")
        loan_officer_id = current_user["id"]
        
        # Convert data models with error handling
        try:
            applicant_info = convert_applicant_info(request_data.applicant_info)
            comaker_info = convert_comaker_info(request_data.comaker_info)
            model_input_data = convert_model_input_data(request_data.model_input_data)
        except Exception as conv_error:
            logger.error(f"Error converting data models: {conv_error}")
            raise HTTPException(
                status_code=422,
                detail=f"Error converting application data: {str(conv_error)}"
            )
            
        try:
            loan_application = LoanApplication(
                loan_officer_id=loan_officer_id,
                applicant_info=applicant_info,
                comaker_info=comaker_info,
                model_input_data=model_input_data
            )
        except Exception as create_error:
            logger.error(f"Error creating loan application: {create_error}")
            raise HTTPException(
                status_code=422,
                detail=f"Error creating loan application: {str(create_error)}"
            )
        
        # Run prediction
        prediction_result = await service._run_prediction(model_input_data)
        
        # Get recommendations
        if service.recommendation_service:
            logger.info("Attempting to get loan recommendations")
            try:
                logger.debug(f"Applicant info for recommendations: {applicant_info.model_dump()}")
                logger.debug(f"Model input data for recommendations: {model_input_data.model_dump()}")
                
                recommended_products = service.recommendation_service.get_loan_recommendations(
                    applicant_info=applicant_info,  # Use the converted model
                    model_input_data=model_input_data.model_dump()
                )
                
                # IMPORTANT: Update the prediction_result with recommendations
                prediction_result.loan_recommendation = recommended_products
                
                logger.info(f"Generated {len(recommended_products)} loan recommendations")
                if not recommended_products:
                    logger.warning("No loan recommendations were generated")
            except Exception as rec_error:
                logger.error(f"Error generating loan recommendations: {rec_error}", exc_info=True)
                # Don't raise the error, just log it as recommendations are optional
                prediction_result.loan_recommendation = []
        
        # Update the prediction result and AI explanation in loan application before saving
        loan_application.prediction_result = prediction_result

        # Generate AI explanation
        ai_explanation = await service._generate_and_save_explanation(loan_application)
        loan_application.ai_explanation = ai_explanation  # Add this line to save the AI explanation
        
        # Save loan application
        await loan_application.save()
        logger.info(f"Loan application created successfully with ID: {loan_application.application_id}, with {len(prediction_result.loan_recommendation)} recommendations")
        
        # Handle document upload - THIS IS THE CRITICAL PART
        document_result = None
        document_upload_failed = False
        document_error_message = None
        
        try:
            # Check if any files were actually provided
            valid_files = {k: v for k, v in document_files.items() if v is not None and v.filename}
            
            if valid_files:
                logger.info(f"Starting document processing for application {loan_application.application_id}")
                logger.info(f"Valid files to process: {list(valid_files.keys())}")
                
                # Log file details for debugging
                for field, file in valid_files.items():
                    logger.debug(f"File {field}: name={file.filename}, "
                               f"content_type={file.content_type}")
                
                # Process documents with retries
                max_retries = 3
                retry_count = 0
                last_error = None
                
                while retry_count < max_retries:
                    try:
                        logger.info(f"Attempting document upload (attempt {retry_count + 1}/{max_retries})")
                        
                        document_result = await handle_application_documents(
                            application_id=str(loan_application.application_id),
                            files=valid_files,
                            user_email=current_user["email"]
                        )
                        
                        if document_result:
                            logger.info(f"Documents processed successfully for application "
                                      f"{loan_application.application_id}")
                            logger.info(f"Document result: {document_result.document_id if hasattr(document_result, 'document_id') else 'success'}")
                        else:
                            logger.warning(f"Document processing returned None for application "
                                         f"{loan_application.application_id}")
                        break
                        
                    except HTTPException as doc_exc:
                        last_error = doc_exc
                        logger.error(f"Document upload HTTPException: {doc_exc.status_code} - {doc_exc.detail}")
                        
                        # Don't retry client errors (4xx)
                        if doc_exc.status_code < 500:
                            logger.error(f"Client error in document upload, not retrying: {doc_exc.detail}")
                            document_upload_failed = True
                            document_error_message = doc_exc.detail
                            break
                        
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error(f"Document processing failed after {max_retries} attempts")
                            document_upload_failed = True
                            document_error_message = doc_exc.detail
                            break
                            
                        logger.warning(f"Retrying document upload (attempt {retry_count + 1}/{max_retries})")
                        await asyncio.sleep(1 * retry_count)  # Exponential backoff
                        
                    except Exception as doc_error:
                        last_error = doc_error
                        logger.error(f"Unexpected error in document upload: {str(doc_error)}", exc_info=True)
                        
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error(f"Document processing failed after {max_retries} attempts")
                            document_upload_failed = True
                            document_error_message = str(doc_error)
                            break
                            
                        logger.warning(f"Retrying document upload after error (attempt {retry_count + 1}/{max_retries})")
                        await asyncio.sleep(1 * retry_count)
                
                # Check if upload ultimately failed
                if document_upload_failed:
                    logger.error(f"Document upload failed for application {loan_application.application_id}: {document_error_message}")
                    # Log the error but continue with the application
                    logger.warning(f"Document upload failed but continuing: {document_error_message}")
                    document_result = None  # Clear document result in case of failure
                    
            else:
                logger.warning("No valid documents provided for application")
                # Decide if you want to allow applications without documents
                # If documents are required, uncomment the following:
                # await loan_application.delete()
                # raise HTTPException(status_code=400, detail="At least one document is required")
                
        except HTTPException:
            # Re-raise HTTP exceptions (already handled above)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in document processing: {str(e)}", exc_info=True)
            # Delete the loan application since document upload failed
            await loan_application.delete()
            raise HTTPException(
                status_code=500,
                detail=f"Document processing failed: {str(e)}"
            )
        
        # Build and return response
        response = build_loan_application_response(
            loan_application,
            prediction_result,
            recommended_products,
            current_user,
            ai_explanation,
            document_result
        )
        
        logger.info(f"Successfully completed loan application process for {loan_application.application_id}")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in loan application worker: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Loan application processing failed: {str(e)}"
        )