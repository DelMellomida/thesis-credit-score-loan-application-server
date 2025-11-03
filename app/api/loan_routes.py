from fastapi import APIRouter, HTTPException, status, Depends, Query, UploadFile, File, Form
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

# Configure logging
logger = logging.getLogger(__name__)

def get_loan_application_service() -> LoanApplicationService:
    if loan_application_service is None:
        logger.error("Loan application service is not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Loan application service is not initialized. Please contact system administrator."
        )
    
    return loan_application_service

router = APIRouter(prefix="/loans", tags=["Loan Applications"])

@router.post("/applications", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_loan_application(
    request_data: str = Form(...),
    profilePhoto: UploadFile = File(...),
    validId: UploadFile = File(...),
    brgyCert: UploadFile = File(...),
    eSignaturePersonal: UploadFile = File(...),
    payslip: UploadFile = File(...),
    companyId: UploadFile = File(...),
    proofOfBilling: UploadFile = File(...),
    eSignatureCoMaker: UploadFile = File(...),
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
            
        # Ensure job field is set to a valid enum value
        if "applicant_info" in data_dict and "job" in data_dict["applicant_info"]:
            if data_dict["applicant_info"]["job"] == "Government Employee":
                data_dict["applicant_info"]["job"] = "Others"
                
        # Ensure Paluwagan_Participation is correctly spelled
        if "model_input_data" in data_dict and "Paluwagan_Participation" in data_dict["model_input_data"]:
            if data_dict["model_input_data"]["Paluwagan_Participation"] == "Rarel":
                data_dict["model_input_data"]["Paluwagan_Participation"] = "Rarely"

        # Convert the dict to a Pydantic model
        try:
            parsed_request_data = FullLoanApplicationRequest(**data_dict)
        except Exception as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid request data format: {str(e)}"
            )

        document_files = {
            "profile_photo": profilePhoto,
            "valid_id": validId,
            "brgy_cert": brgyCert,
            "e_signature_personal": eSignaturePersonal,
            "payslip": payslip,
            "company_id": companyId,
            "proof_of_billing": proofOfBilling,
            "e_signature_comaker": eSignatureCoMaker
        }
        
        return await process_loan_application(
            parsed_request_data,
            document_files,
            current_user,
            service
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in loan application creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the application: {str(e)}"
        )

# @router.post("/applications/demo", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
# async def create_demo_loan_application(
#     # Applicant Info
#     applicant_name: str,
#     contact_number: str,
#     address: str,
#     salary: str,
#     job: str,
    
#     # Co-maker Info
#     comaker_name: str,
#     comaker_contact: str,
    
#     # Key Model Inputs
#     employment_sector: EmploymentSectorEnum,
#     employment_tenure_months: int,
#     net_salary_per_cutoff: float,
#     salary_frequency: SalaryFrequencyEnum,
#     housing_status: HousingStatusEnum,
#     years_at_current_address: float,
#     household_head: YesNoEnum,
#     number_of_dependents: int,
#     comaker_relationship: ComakerRelationshipEnum,
#     comaker_employment_tenure_months: int,
#     comaker_net_salary_per_cutoff: float,
#     has_community_role: CommunityRoleEnum,
#     paluwagan_participation: PaluwaganParticipationEnum,
#     other_income_source: OtherIncomeSourceEnum,
#     disaster_preparedness: DisasterPreparednessEnum,
#     is_renewing_client: int = 0,
#     grace_period_usage_rate: float = 0.0,
#     late_payment_count: int = 0,
#     had_special_consideration: int = 0,
    
#     current_user: Dict = Depends(get_current_active_user),
#     service: LoanApplicationService = Depends(get_loan_application_service)
# ):
#     """
#     Demo endpoint for loan application creation - all fields as individual parameters.
#     This will show each field separately in Swagger UI for easy testing and demo purposes.
#     Perfect for presentations where you want to fill out fields one by one.
#     """
#     try:
#         logger.info(f"Starting demo loan application creation for user: {current_user['email']}")
        
#         # Get loan officer ID from authenticated user
#         loan_officer_id = current_user["id"]
        
#         # Reconstruct the nested models from individual parameters
#         applicant_info = ApplicantInfo(
#             full_name=applicant_name,
#             contact_number=contact_number,
#             address=address,
#             salary=salary,
#             job=job
#         )
        
#         comaker_info = CoMakerInfo(
#             full_name=comaker_name,
#             contact_number=comaker_contact
#         )
        
#         model_input_data = LoanApplicationRequest(
#             Employment_Sector=employment_sector,
#             Employment_Tenure_Months=employment_tenure_months,
#             Net_Salary_Per_Cutoff=net_salary_per_cutoff,
#             Salary_Frequency=salary_frequency,
#             Housing_Status=housing_status,
#             Years_at_Current_Address=years_at_current_address,
#             Household_Head=household_head,
#             Number_of_Dependents=number_of_dependents,
#             Comaker_Relationship=comaker_relationship,
#             Comaker_Employment_Tenure_Months=comaker_employment_tenure_months,
#             Comaker_Net_Salary_Per_Cutoff=comaker_net_salary_per_cutoff,
#             Has_Community_Role=has_community_role,
#             Paluwagan_Participation=paluwagan_participation,
#             Other_Income_Source=other_income_source,
#             Disaster_Preparedness=disaster_preparedness,
#             Is_Renewing_Client=is_renewing_client,
#             Grace_Period_Usage_Rate=grace_period_usage_rate,
#             Late_Payment_Count=late_payment_count,
#             Had_Special_Consideration=had_special_consideration
#         )
        
#         # Create full request object
#         full_request = FullLoanApplicationRequest(
#             applicant_info=applicant_info,
#             comaker_info=comaker_info,
#             model_input_data=model_input_data
#         )
        
#         # Convert to database models
#         try:
#             db_applicant_info = DbApplicantInfo(**applicant_info.model_dump())
#             db_comaker_info = DbCoMakerInfo(
#                 full_name=comaker_info.full_name,
#                 contact_number=comaker_info.contact_number
#             )
#             db_model_input_data = ModelInputData(**model_input_data.model_dump())
#         except Exception as e:
#             logger.error(f"Error converting to database models: {e}")
#             raise ValueError(f"Error in data conversion: {e}")
        
#         # Create the loan application
#         try:
#             loan_application = LoanApplication(
#                 loan_officer_id=loan_officer_id,
#                 applicant_info=db_applicant_info,
#                 comaker_info=db_comaker_info,
#                 model_input_data=db_model_input_data
#             )
#         except Exception as e:
#             logger.error(f"Error creating loan application: {e}")
#             raise ValueError(f"Error creating loan application: {e}")
        
#         # Run prediction using the service
#         try:
#             logger.info("Running prediction for demo loan application")
#             prediction_result = await service._run_prediction(db_model_input_data)
#             loan_application.prediction_result = prediction_result
#         except Exception as e:
#             logger.error(f"Error during prediction: {e}")
#             raise RuntimeError(f"Prediction failed: {e}")
        
#         # Get loan recommendations using the recommendation service
#         try:
#             recommended_products = []
#             if service.recommendation_service:
#                 recommended_products = service.recommendation_service.get_loan_recommendations(
#                     applicant_info=applicant_info,
#                     model_input_data=model_input_data.model_dump()
#                 )
#                 logger.info("Recommended products are created")
#             else:
#                 logger.warning("Recommendation service not available")
#         except Exception as e:
#             logger.error(f"Error during recommending products: {e}")
#             raise ValueError(f"Recommending products failed: {e}")
        
#         try:
#             ai_explanation = await service._generate_and_save_explanation(loan_application)
#         except Exception as e:
#             logger.error(f"Error generating AI explanation: {e}")
#             raise RuntimeError(f"AI explanation generation failed: {e}")
        
#         # Save to database
#         try:
#             await loan_application.save()
#         except Exception as e:
#             logger.error(f"Error saving to database: {e}")
#             raise RuntimeError(f"Database error: {e}")
        
#         logger.info(f"Demo loan application created successfully with ID: {loan_application.application_id}")
        
#         return {
#             "message": "Demo loan application created successfully",
#             "application_id": str(loan_application.application_id),
#             "timestamp": loan_application.timestamp.isoformat(),
#             "status": "created",
#             "demo_mode": True,
#             "prediction_result": {
#                 "final_credit_score": prediction_result.final_credit_score,
#                 "default": prediction_result.default,
#                 "probability_of_default": prediction_result.probability_of_default,
#                 "status": prediction_result.status
#             },
#             "recommended_products": recommended_products,
#             "applicant_info": {
#                 "full_name": applicant_info.full_name,
#                 "contact_number": applicant_info.contact_number,
#                 "address": applicant_info.address,
#                 "salary": applicant_info.salary,
#                 "job": applicant_info.job
#             },
#             "loan_officer_id": loan_application.loan_officer_id,
#             "created_by": {
#                 "email": current_user["email"],
#                 "full_name": current_user["full_name"]
#             },
#             "ai_explanation": ai_explanation
#         }
        
#     except ValueError as e:
#         logger.error(f"Validation error during demo application creation: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=f"Invalid demo application data: {str(e)}"
#         )
#     except RuntimeError as e:
#         logger.error(f"Runtime error during demo application creation: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=str(e)
#         )
#     except Exception as e:
#         logger.error(f"Unexpected error during demo application creation: {e}")
#         import traceback
#         logger.error(f"Full traceback: {traceback.format_exc()}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Unexpected demo error: {str(e)}"
#         )

@router.get("/applications/id/{mongo_id}", response_model=Dict[str, Any])
async def get_loan_application_by_id(
    mongo_id: str,
    include_ai_explanation: bool = Query(default=True, description="Include AI explanation in response"),
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    """
    Get loan application by MongoDB _id
    """
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
        
        # Add prediction result if available
        if loan_application.prediction_result:
            response["prediction_result"] = loan_application.prediction_result.model_dump()
            
        # Add AI explanation if requested and available
        if include_ai_explanation and loan_application.ai_explanation:
            response["ai_explanation"] = loan_application.ai_explanation.model_dump()
            
        # Get and add documents if available
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

@router.get("/applications/{application_id}", response_model=Dict[str, Any])
async def get_loan_application(
    application_id: UUID,
    include_ai_explanation: bool = Query(default=True, description="Include AI explanation in response"),
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    """
    Retrieve a specific loan application by ID.
    Only accessible to authenticated users.
    """
    try:
        loan_application = await service.get_loan_application(application_id)
        if not loan_application:
            raise HTTPException(status_code=404, detail="Loan application not found")
            
        # Get associated documents
        from app.database.models.document_model import ApplicationDocument
        documents = await ApplicationDocument.find_one(
            ApplicationDocument.application_id == str(application_id)
        )
        
        # Create the base response
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
        
        # Include prediction result if available
        if loan_application.prediction_result:
            response["prediction_result"] = loan_application.prediction_result.model_dump()
            
        # Include AI explanation if requested and available
        if include_ai_explanation and loan_application.ai_explanation:
            response["ai_explanation"] = loan_application.ai_explanation
            
        # Include document URLs if available
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
        logger.error(f"Error retrieving loan application {application_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving loan application: {str(e)}"
        )
            
    return response

@router.put("/applications/{application_id}", response_model=Dict[str, Any])
async def update_loan_application(
    application_id: str,  # Changed from UUID to str for MongoDB ObjectId
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
    """
    Update loan application data and handle document updates.
    """
    try:
        try:
            data_dict = json.loads(request_data)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON format: {str(e)}"
            )
        
        # Get existing application
        application = await service.get_loan_application_by_mongo_id(application_id)
        if not application:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Loan application with ID {application_id} not found"
            )

        # Update document files if provided
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

        # If there are document updates, process them
        if document_files:
            from app.workers.document_handler import process_document_updates
            document_urls = await process_document_updates(
                str(application.application_id),
                document_files,
                current_user
            )

            # Update document URLs in database
            from app.database.models.document_model import ApplicationDocument
            await ApplicationDocument.find_one(
                ApplicationDocument.application_id == str(application.application_id)
            ).update({"$set": document_urls})

        # Update application data and run reassessment
        updated_application = await service.update_loan_application(
            application.application_id, 
            data_dict,
            rerun_assessment=True
        )

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

@router.get("/applications", response_model=Dict[str, Any])
async def get_loan_applications(
    skip: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=6, ge=1, le=1000, description="Maximum number of records to return"),
    loan_officer_id: Optional[str] = Query(default=None, description="Filter by loan officer ID"),
    include_ai_explanation: bool = Query(default=False, description="Include AI explanations in response"),
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    try:
        # Use current user's ID if no specific loan officer ID is provided
        if loan_officer_id is None:
            loan_officer_id = current_user.get("id")
            if not loan_officer_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No loan officer ID provided and couldn't get it from current user"
                )
        
        logger.info(f"Fetching applications for officer {loan_officer_id} (skip={skip}, limit={limit})")
        try:
            result = await service.get_loan_applications(
                skip=skip,
                limit=limit,
                loan_officer_id=loan_officer_id
            )
            logger.info(f"Successfully retrieved {len(result.get('data', []))} applications")
            return result
            
        except Exception as e:
            logger.error(f"Service error in get_loan_applications: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error retrieving applications: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_loan_applications: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving applications"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving loan applications: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving applications"
        )

@router.get("/my-applications", response_model=List[Dict[str, Any]])
async def get_my_loan_applications(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    include_ai_explanation: bool = Query(default=False),
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    """
    Retrieve loan applications created by the authenticated user.
    """
    try:
        applications = await service.get_loan_applications(
            skip=skip,
            limit=limit,
            loan_officer_id=current_user["id"]
        )
        
        # Format response data
        response_data = []
        for app in applications:
            app_data = {
                "_id": {"$oid": str(app.id)},  # Include MongoDB ObjectId
                "application_id": str(app.application_id),
                "timestamp": app.timestamp.isoformat(),
                "applicant_name": app.applicant_info.full_name,
                "contact_number": app.applicant_info.contact_number,
                "status": app.prediction_result.status if app.prediction_result else "Unknown",
                "credit_score": app.prediction_result.final_credit_score if app.prediction_result else None,
                "recommendation_count": len(app.prediction_result.loan_recommendation or []) if app.prediction_result else 0
            }
            
            # Add AI explanation if requested and available
            if include_ai_explanation and hasattr(app, 'ai_explanation') and app.ai_explanation:
                app_data["ai_explanation_summary"] = {
                    "has_explanation": True,
                    "recommendation": app.ai_explanation.recommendation if hasattr(app.ai_explanation, 'recommendation') else None
                }
            
            response_data.append(app_data)
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error retrieving user's loan applications: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving your applications"
        )

@router.put("/applications/{application_id}/status")
async def update_application_status(
    application_id: str,  # Changed from UUID to str to accept MongoDB ObjectId
    status_update: Dict[str, str],
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    """
    Update the status of a loan application.
    Only accessible to authenticated users.
    """
    try:
        logger.info(f"Received status update request: {status_update}")
        new_status = status_update.get("status")
        logger.info(f"Extracted status: {new_status}")
        
        if not new_status:
            logger.error("No status provided in request")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Status is required"
            )
        
        # Validate status value using the enum (case-insensitive)
        try:
            logger.info(f"Attempting to validate status: {new_status}")
            logger.info(f"Valid statuses are: {[s.value for s in ApplicationStatusEnum]}")
            # Convert input to title case to match enum values
            normalized_status = new_status.title()
            status_enum = ApplicationStatusEnum(normalized_status)
            logger.info(f"Status validated successfully as: {status_enum.value}")
            # Use the validated enum value
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
        logger.error(f"Error updating application status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while updating the application status"
        )

@router.post("/applications/{application_id}/regenerate-recommendations")
async def regenerate_loan_recommendations(
    application_id: UUID,
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    """
    Regenerate loan recommendations for an existing application.
    """
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
        logger.error(f"Error regenerating recommendations for application {application_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while regenerating recommendations"
        )

@router.delete("/applications/{application_id}")
async def delete_loan_application(
    application_id: UUID,
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    """
    Delete a loan application by ID.
    Only accessible to authenticated users.
    """
    try:
        # Optional: Check if user owns this application
        application = await service.get_loan_application(application_id)
        if not application:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Loan application with ID {application_id} not found"
            )
        
        # Optional authorization check
        # if application.loan_officer_id != current_user["id"]:
        #     raise HTTPException(
        #         status_code=status.HTTP_403_FORBIDDEN,
        #         detail="You can only delete your own loan applications"
        #     )
        
        deleted = await service.delete_loan_application(application_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Loan application with ID {application_id} not found"
            )
        
        return {
            "message": "Loan application deleted successfully",
            "application_id": str(application_id),
            "deleted_at": "now"  # You might want to add actual timestamp
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting loan application {application_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while deleting the application"
        )

@router.get("/health", tags=["Health Check"])
async def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint for the loan service.
    """
    try:
        from datetime import datetime
        return {
            "status": "healthy",
            "service": "loan-api",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed"
        )

@router.get("/service-status", tags=["Health Check"])
async def get_service_status(
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
) -> Dict[str, Any]:
    """
    Get the current status of the loan application service.
    Only accessible to authenticated users.
    """
    try:
        logger.info(f"Retrieving loan service status for user: {current_user['email']}")
        status_info = await service.get_service_status()
        
        logger.info("Loan service status retrieved successfully")
        return status_info
        
    except Exception as e:
        logger.error(f"Error retrieving service status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve service status"
        )