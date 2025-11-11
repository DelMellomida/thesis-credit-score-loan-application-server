from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, Form
from fastapi import status
from typing import Dict, Any, List, Optional
import logging
import json
from uuid import UUID
from app.services.loan_service import LoanApplicationService, loan_application_service
from app.schemas.loan_schema import (
    FullLoanApplicationRequest,
    AIExplanation,
    FullLoanApplicationResponse,
    ApplicantInfo,
    CoMakerInfo,
    LoanApplicationRequest,
    EmploymentSectorEnum,
    SalaryFrequencyEnum,
    HousingStatusEnum,
    YesNoEnum,
    ComakerRelationshipEnum,
    OtherIncomeSourceEnum,
    DisasterPreparednessEnum,
    RecommendedProducts,
    PaluwaganParticipationEnum,
    CommunityRoleEnum,
    ApplicationStatusEnum
)
from app.database.models.loan_application_model import (
    LoanApplication, 
    ApplicantInfo as DbApplicantInfo, 
    CoMakerInfo as DbCoMakerInfo, 
    ModelInputData
)
from app.core.auth_dependencies import get_current_user, get_current_active_user

from app.workers.loan_application_worker import process_loan_application
from app.schemas.document_schema import DocumentUploadRequest

logger = logging.getLogger(__name__)

# Returns the loan application service instance or raises an error if unavailable
def get_loan_application_service() -> LoanApplicationService:
    if loan_application_service is None:
        logger.error("Loan application service is not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Loan application service is not initialized. Please contact system administrator."
        )
    
    return loan_application_service

router = APIRouter(prefix="/loans", tags=["Loan Applications"])

# Creates a new loan application with prediction, recommendations, and document uploads
@router.post("/applications", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_loan_application(
    request_data: str = Form(...),
    profilePhoto: Optional[UploadFile] = File(None),
    validId: Optional[UploadFile] = File(None),
    brgyCert: Optional[UploadFile] = File(None),
    eSignaturePersonal: Optional[UploadFile] = File(None),
    payslip: Optional[UploadFile] = File(None),
    companyId: Optional[UploadFile] = File(None),
    proofOfBilling: Optional[UploadFile] = File(None),
    eSignatureCoMaker: Optional[UploadFile] = File(None),
    current_user: Dict = Depends(get_current_active_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    try:
        try:
            data_dict = json.loads(request_data)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON format: {str(e)}"
            )
            
        # Allow any applicant job string to pass through; do not coerce specific job titles here.
                
        if "model_input_data" in data_dict and "Paluwagan_Participation" in data_dict["model_input_data"]:
            if data_dict["model_input_data"]["Paluwagan_Participation"] == "Rarel":
                data_dict["model_input_data"]["Paluwagan_Participation"] = "Rarely"

        try:
            parsed_request_data = FullLoanApplicationRequest(**data_dict)
        except Exception as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid request data format: {str(e)}"
            )

        document_files = {}
        if profilePhoto:
            document_files["profile_photo"] = profilePhoto
        if validId:
            document_files["valid_id"] = validId
        if brgyCert:
            document_files["brgy_cert"] = brgyCert
        if eSignaturePersonal:
            document_files["e_signature_personal"] = eSignaturePersonal
        if payslip:
            document_files["payslip"] = payslip
        if companyId:
            document_files["company_id"] = companyId
        if proofOfBilling:
            document_files["proof_of_billing"] = proofOfBilling
        if eSignatureCoMaker:
            document_files["e_signature_comaker"] = eSignatureCoMaker
        
        return await process_loan_application(
            parsed_request_data,
            document_files,
            current_user,
            service
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in loan application creation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the application"
        )

# Retrieves a loan application by its MongoDB ObjectId
@router.get("/applications/id/{mongo_id}", response_model=Dict[str, Any])
async def get_loan_application_by_id(
    mongo_id: str,
    include_ai_explanation: bool = Query(default=True, description="Include AI explanation in response"),
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    try:
        from bson import ObjectId
        loan_application = await LoanApplication.find_one({"_id": ObjectId(mongo_id)})
        if not loan_application:
            raise HTTPException(status_code=404, detail="Loan application not found")
            
        response = {
            "_id": {"$oid": str(loan_application.id)},
            "application_id": loan_application.application_id,
            "timestamp": loan_application.timestamp.isoformat(),
            "status": loan_application.status,
            "applicant_info": loan_application.applicant_info.model_dump(),
            "comaker_info": loan_application.comaker_info.model_dump(),
            "model_input_data": loan_application.model_input_data.model_dump(),
            "loan_officer_id": loan_application.loan_officer_id,
        }
        
        if loan_application.prediction_result:
            response["prediction_result"] = loan_application.prediction_result.model_dump()
            
        if include_ai_explanation and loan_application.ai_explanation:
            response["ai_explanation"] = loan_application.ai_explanation.model_dump()
            
        from app.database.models.document_model import ApplicationDocument
        documents = await ApplicationDocument.find_one(
            ApplicationDocument.application_id == str(loan_application.application_id)
        )
        if documents:
            response["documents"] = {
                "profile_photo_url": documents.profile_photo_url,
                "valid_id_url": documents.valid_id_url,
                "brgy_cert_url": documents.brgy_cert_url,
                "e_signature_personal_url": documents.e_signature_personal_url,
                "payslip_url": documents.payslip_url,
                "company_id_url": documents.company_id_url,
                "proof_of_billing_url": documents.proof_of_billing_url,
                "e_signature_comaker_url": documents.e_signature_comaker_url,
                "file_metadata": documents.file_metadata
            }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving loan application by MongoDB id {mongo_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving loan application: {str(e)}"
        )

# Retrieves a specific loan application by its UUID
@router.get("/applications/{application_id}", response_model=Dict[str, Any])
async def get_loan_application(
    application_id: UUID,
    include_ai_explanation: bool = Query(default=True, description="Include AI explanation in response"),
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    try:
        loan_application = await service.get_loan_application(application_id)
        if not loan_application:
            raise HTTPException(status_code=404, detail="Loan application not found")
            
        from app.database.models.document_model import ApplicationDocument
        documents = await ApplicationDocument.find_one(
            ApplicationDocument.application_id == str(application_id)
        )
        
        response = {
            "_id": {"$oid": str(loan_application.id)},
            "application_id": str(loan_application.application_id),
            "timestamp": loan_application.timestamp.isoformat(),
            "status": loan_application.status,
            "applicant_info": loan_application.applicant_info.model_dump(),
            "comaker_info": loan_application.comaker_info.model_dump(),
            "model_input_data": loan_application.model_input_data.model_dump(),
            "loan_officer_id": loan_application.loan_officer_id,
        }
        
        if loan_application.prediction_result:
            response["prediction_result"] = loan_application.prediction_result.model_dump()
            
        if include_ai_explanation and loan_application.ai_explanation:
            response["ai_explanation"] = loan_application.ai_explanation
            
        if documents:
            response["documents"] = {
                "profile_photo_url": documents.profile_photo_url,
                "valid_id_url": documents.valid_id_url,
                "brgy_cert_url": documents.brgy_cert_url,
                "e_signature_personal_url": documents.e_signature_personal_url,
                "payslip_url": documents.payslip_url,
                "company_id_url": documents.company_id_url,
                "proof_of_billing_url": documents.proof_of_billing_url,
                "e_signature_comaker_url": documents.e_signature_comaker_url,
                "file_metadata": documents.file_metadata
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error retrieving loan application")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving loan application"
        )
            
    return response

# Updates loan application data and documents with optional reassessment
@router.put("/applications/{application_id}", response_model=Dict[str, Any])
async def update_loan_application(
    application_id: str,
    request_data: str = Form(...),
    profilePhoto: Optional[UploadFile] = File(None),
    validId: Optional[UploadFile] = File(None),
    brgyCert: Optional[UploadFile] = File(None),
    eSignaturePersonal: Optional[UploadFile] = File(None),
    payslip: Optional[UploadFile] = File(None),
    companyId: Optional[UploadFile] = File(None),
    proofOfBilling: Optional[UploadFile] = File(None),
    eSignatureCoMaker: Optional[UploadFile] = File(None),
    current_user: Dict = Depends(get_current_active_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    try:
        try:
            data_dict = json.loads(request_data)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON format: {str(e)}"
            )
        
        application = await service.get_loan_application_by_mongo_id(application_id)
        if not application:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Loan application with ID {application_id} not found"
            )

        try:
            update_data = {}
            
            if 'applicant_info' in data_dict:
                current_applicant = application.applicant_info.model_dump()
                current_applicant.update(data_dict['applicant_info'])
                update_data['applicant_info'] = current_applicant

            if 'comaker_info' in data_dict:
                current_comaker = application.comaker_info.model_dump()
                current_comaker.update(data_dict['comaker_info'])
                update_data['comaker_info'] = current_comaker

            if 'model_input_data' in data_dict:
                current_model_input = application.model_input_data.model_dump()
                current_model_input.update(data_dict['model_input_data'])
                update_data['model_input_data'] = current_model_input

            document_files = {}
            if profilePhoto:
                document_files["profile_photo"] = profilePhoto
            if validId:
                document_files["valid_id"] = validId
            if brgyCert:
                document_files["brgy_cert"] = brgyCert
            if eSignaturePersonal:
                document_files["e_signature_personal"] = eSignaturePersonal
            if payslip:
                document_files["payslip"] = payslip
            if companyId:
                document_files["company_id"] = companyId
            if proofOfBilling:
                document_files["proof_of_billing"] = proofOfBilling
            if eSignatureCoMaker:
                document_files["e_signature_comaker"] = eSignatureCoMaker

            logger.info("Application data processed for update")
        except Exception as e:
            logger.error(f"Error processing update data: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid update data structure: {str(e)}"
            )

        if document_files:
            from app.workers.document_handler import process_document_updates
            document_urls = await process_document_updates(
                str(application.application_id),
                document_files,
                current_user
            )

            from app.database.models.document_model import ApplicationDocument
            doc = await ApplicationDocument.find_one(
                ApplicationDocument.application_id == str(application.application_id)
            )
            if doc:
                await doc.update({"$set": document_urls})

        if update_data:
            updated_application = await service.update_loan_application(
                application.application_id, 
                update_data,
                rerun_assessment=True
            )
        else:
            updated_application = application

        if not updated_application:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Loan application with ID {application_id} not found"
            )

        return {
            "message": "Application updated successfully",
            "application_id": str(application_id),
            "updated_at": updated_application.timestamp.isoformat(),
            "application": {
                "_id": {"$oid": str(updated_application.id)},
                "application_id": str(updated_application.application_id),
                "timestamp": updated_application.timestamp.isoformat(),
                "status": updated_application.status,
                "applicant_info": updated_application.applicant_info.model_dump(),
                "comaker_info": updated_application.comaker_info.model_dump(),
                "model_input_data": updated_application.model_input_data.model_dump(),
                "loan_officer_id": updated_application.loan_officer_id,
                "prediction_result": updated_application.prediction_result.model_dump() if updated_application.prediction_result else None,
                "ai_explanation": updated_application.ai_explanation.model_dump() if updated_application.ai_explanation else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving loan application {application_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error retrieving loan application: {str(e)}"
        )

# Retrieves paginated loan applications with optional filtering
@router.get("/applications", response_model=Dict[str, Any])
async def get_loan_applications(
    skip: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=6, ge=1, le=1000, description="Maximum number of records to return"),
    loan_officer_id: Optional[str] = Query(default=None, description="Filter by loan officer ID"),
    status: Optional[str] = Query(default=None, description="Filter by application status"),
    search: Optional[str] = Query(default=None, description="Search in applicant name, email, or loan details"),
    include_ai_explanation: bool = Query(default=False, description="Include AI explanations in response"),
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    try:
        if loan_officer_id is None:
            loan_officer_id = current_user.get("id")
            if not loan_officer_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No loan officer ID provided and couldn't get it from current user"
                )
        
        logger.info("Processing loan application request")
        result = await service.get_loan_applications(
            skip=skip,
            limit=limit,
            loan_officer_id=loan_officer_id,
            status=status,
            search=search
        )
        logger.info("Application data retrieved successfully")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error retrieving loan applications")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving applications"
        )

# Retrieves loan applications created by the authenticated user
@router.get("/my-applications", response_model=List[Dict[str, Any]])
async def get_my_loan_applications(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    include_ai_explanation: bool = Query(default=False),
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    try:
        applications = await service.get_loan_applications(
            skip=skip,
            limit=limit,
            loan_officer_id=current_user["id"]
        )
        
        response_data = []
        for app in applications:
            app_data = {
                "_id": {"$oid": str(app.id)},
                "application_id": str(app.application_id),
                "timestamp": app.timestamp.isoformat(),
                "applicant_name": app.applicant_info.full_name,
                "contact_number": app.applicant_info.contact_number,
                "status": app.prediction_result.status if app.prediction_result else "Unknown",
                "credit_score": app.prediction_result.final_credit_score if app.prediction_result else None,
                "recommendation_count": len(app.prediction_result.loan_recommendation or []) if app.prediction_result else 0
            }
            
            if include_ai_explanation and hasattr(app, 'ai_explanation') and app.ai_explanation:
                app_data["ai_explanation_summary"] = {
                    "has_explanation": True,
                    "recommendation": app.ai_explanation.recommendation if hasattr(app.ai_explanation, 'recommendation') else None
                }
            
            response_data.append(app_data)
        
        return response_data
        
    except Exception as e:
        logger.error("Error retrieving loan applications")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving applications"
        )

# Updates the status of a loan application with validation
@router.put("/applications/{application_id}/status")
async def update_application_status(
    application_id: str,
    status_update: Dict[str, str],
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    try:
        new_status = status_update.get("status")
        
        if not new_status:
            logger.error("No status provided in request")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Status is required"
            )
        
        try:
            logger.info("Validating requested status update")
            normalized_status = new_status.title()
            status_enum = ApplicationStatusEnum(normalized_status)
            logger.info("Status validation completed")
            new_status = status_enum.value
        except ValueError as e:
            logger.error(f"Invalid status value: {new_status}")
            logger.error(f"Validation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Valid statuses are: {', '.join([s.value for s in ApplicationStatusEnum])}"
            )
        
        updated_application = await service.update_application_status(
            application_id=application_id,
            new_status=new_status
        )
        
        if not updated_application:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Loan application with ID {application_id} not found"
            )
        
        return {
            "message": "Application status updated successfully",
            "application_id": str(application_id),
            "new_status": new_status,
            "updated_at": updated_application.timestamp.isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating application status")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while updating the application status"
        )

# Regenerates loan product recommendations for an existing application
@router.post("/applications/{application_id}/regenerate-recommendations")
async def regenerate_loan_recommendations(
    application_id: UUID,
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    try:
        recommendations = await service.regenerate_loan_recommendations(application_id)
        
        if recommendations is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Loan application with ID {application_id} not found or recommendation service unavailable"
            )
        
        return {
            "message": "Loan recommendations regenerated successfully",
            "application_id": str(application_id),
            "recommendation_count": len(recommendations),
            "recommendations": recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error regenerating loan recommendations")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while regenerating recommendations"
        )

# Deletes a loan application by its UUID
@router.delete("/applications/{application_id}")
async def delete_loan_application(
    application_id: UUID,
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    try:
        application = await service.get_loan_application(application_id)
        if not application:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Loan application with ID {application_id} not found"
            )
        
        deleted = await service.delete_loan_application(application_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Loan application with ID {application_id} not found"
            )
        
        return {
            "message": "Loan application deleted successfully",
            "application_id": str(application_id),
            "deleted_at": "now"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting loan application")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while deleting the application"
        )

# Returns the health status of the loan service
@router.get("/health", tags=["Health Check"])
async def health_check() -> Dict[str, str]:
    try:
        from datetime import datetime
        return {
            "status": "healthy",
            "service": "loan-api",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error("Health check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed"
        )

# Retrieves detailed status information about the loan application service
@router.get("/service-status", tags=["Health Check"])
async def get_service_status(
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
) -> Dict[str, Any]:
    try:
        logger.info("Retrieving loan service status")
        status_info = await service.get_service_status()
        
        logger.info("Status check completed")
        return status_info
        
    except Exception as e:
        logger.error("Error retrieving service status")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve service status"
        )