from app.database.models.loan_application_model import LoanApplication, PredictionResult, AIExplanation
import logging
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime

from app.schemas.loan_schema import FullLoanApplicationRequest, RecommendedProducts
from app.services.prediction_service import PredictionService, prediction_service
from app.services.ai_service import ai_service
from app.services.loan_recommendation_service import LoanRecommendationService, loan_recommendation_service

logger = logging.getLogger(__name__)


class LoanApplicationService:
    """
    Service class for handling loan application operations.
    """
    
    def __init__(self, 
                 prediction_service: PredictionService, 
                 recommendation_service: Optional[LoanRecommendationService] = None):
        self.prediction_service = prediction_service
        self.ai_service = ai_service
        self.recommendation_service = recommendation_service
        logger.info("LoanApplicationService initialized")

    async def create_loan_application(
        self, 
        request_data: FullLoanApplicationRequest, 
        loan_officer_id: str
    ) -> Dict[str, Any]:
        try:
            logger.info(f"Creating loan application for loan officer: {loan_officer_id}")
            
            # Validate input data
            self._validate_loan_application_data(request_data)
            
            # Run prediction using the prediction service
            try:
                prediction_result = await self._run_prediction(request_data.model_input_data)
                if not prediction_result:
                    raise RuntimeError("Prediction service returned no result")
            except Exception as e:
                logger.error(f"Failed to get prediction result: {e}")
                # Create a failed prediction result
                prediction_result = PredictionResult(
                    final_credit_score=0,
                    default=1,
                    probability_of_default=1.0,
                    loan_recommendation=[],
                    status="Failed",
                )
            
            # Generate loan recommendations if service is available
            if self.recommendation_service:
                loan_recommendations = self.recommendation_service.get_loan_recommendations(
                    applicant_info=request_data.applicant_info,
                    model_input_data=request_data.model_input_data.model_dump()
                )
                prediction_result.loan_recommendation = loan_recommendations
                logger.info(f"Generated {len(loan_recommendations)} loan recommendations")
            else:
                logger.warning("Loan recommendation service not available")
                prediction_result.loan_recommendation = []
            
            # Create loan application record
            new_application = LoanApplication(
                loan_officer_id=loan_officer_id,
                applicant_info=request_data.applicant_info,
                co_maker_info=request_data.comaker_info,
                model_input_data=request_data.model_input_data,
                prediction_result=prediction_result,
                status="Pending"  # Set initial status
            )
            
            # Save to database
            await new_application.insert()
            
            # Generate and save AI explanation asynchronously
            await self._generate_and_save_explanation(new_application)
            
            logger.info(f"Loan application created successfully with ID: {new_application.application_id}")
            return {
                "application": new_application,
                "prediction_result": prediction_result
            }
            
        except ValueError as e:
            logger.error(f"Validation error in loan application creation: {e}")
            raise ValueError(f"Invalid loan application data: {str(e)}")
        except Exception as e:
            logger.error(f"Error creating loan application: {e}")
            raise RuntimeError(f"Failed to create loan application: {str(e)}")
        
    async def _generate_and_save_explanation(self, application: LoanApplication) -> Optional[AIExplanation]:
        try:
            if not self.ai_service:
                logger.warning("AIExplainabilityService is not available, skipping explanation generation")
                return None
            
            logger.info(f"Generating AI explanation for application ID: {application.application_id}")
            
            # Check if we have a valid prediction result
            if not application.prediction_result:
                error_msg = "No prediction result available for AI explanation generation"
                logger.error(error_msg)
                application.ai_explanation_status = "failed"
                application.ai_explanation_error = error_msg
                await application.save()
                return None

            if not hasattr(application.prediction_result, "model_dump"):
                logger.error("Invalid prediction result format")
                application.ai_explanation_status = "failed"
                application.ai_explanation_error = "Invalid prediction result format"
                await application.save()
                return None
                
            prediction_result_dict = application.prediction_result.model_dump()
            
            # Call the AI service asynchronously
            try:
                explanation_dict = await self.ai_service.generate_loan_explanation_async(
                    application_data=application.model_input_data,
                    prediction_results=prediction_result_dict
                )

                # Handle potential failure from the AI service
                if not explanation_dict or "technical_explanation" not in explanation_dict:
                    error_msg = f"AI service failed to return a valid explanation dict. Got: {explanation_dict}"
                    logger.error(error_msg)
                    # Set an error status in the application
                    application.ai_explanation_status = "failed"
                    application.ai_explanation_error = error_msg
                    await application.save()
                    return None
            except Exception as e:
                error_msg = f"AI service failed to generate explanation: {str(e)}"
                logger.error(error_msg)
                # Set an error status in the application
                application.ai_explanation_status = "failed"  
                application.ai_explanation_error = error_msg
                await application.save()
                return None

            ai_explanation = AIExplanation(**explanation_dict)
            application.ai_explanation = ai_explanation
            application.ai_explanation_status = "success"
            application.ai_explanation_error = None
            await application.save()
            logger.info(f"AI explanation generated and saved successfully for application ID: {application.application_id}")
            return ai_explanation
        except Exception as e:
            logger.error(f"Error generating AI explanation for application {application.application_id}: {e}", exc_info=True)
            return None

    async def get_loan_application(self, application_id: UUID) -> Optional[LoanApplication]:
        try:
            logger.info(f"Retrieving loan application with application_id: {application_id}")
            
            # Use find_one to search by application_id field instead of _id
            application = await LoanApplication.find_one(LoanApplication.application_id == application_id)
            
            if application:
                logger.info(f"Loan application {application_id} found")
            else:
                logger.warning(f"Loan application {application_id} not found")
                
            return application
            
        except Exception as e:
            logger.error(f"Error retrieving loan application {application_id}: {e}")
            raise RuntimeError(f"Failed to retrieve loan application: {str(e)}")
    
    async def get_loan_applications(
        self, 
        skip: int = 0, 
        limit: int = 100, 
        loan_officer_id: Optional[str] = None,
        status: Optional[str] = None,
        search: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            logger.info(f"Retrieving loan applications (skip: {skip}, limit: {limit}, officer: {loan_officer_id}, status: {status}, search: {search})")
            
            # Build base query
            query = {}
            if loan_officer_id:
                query["loan_officer_id"] = loan_officer_id
                
            # Add status filter if provided
            if status and status.lower() != 'all':
                query["status"] = status.title()  # Convert to title case to match enum values
                
            # Add search filter if provided
            if search:
                # Use MongoDB $or to search across multiple fields
                search_regex = {"$regex": search, "$options": "i"}  # case-insensitive search
                query["$or"] = [
                    {"applicant_info.full_name": search_regex},
                    {"applicant_info.email": search_regex},
                    {"applicant_info.job": search_regex},
                    {"model_input_data.Employment_Sector": search_regex}
                ]
            
            try:
                # Get total count
                total = await LoanApplication.find(query).count()
                logger.info(f"Total count: {total}")
                
                # Get paginated results with sorting by timestamp in descending order (newest first)
                applications = await LoanApplication.find(query).sort([("timestamp", -1)]).skip(skip).limit(limit).to_list()
                logger.info(f"Retrieved {len(applications)} applications")
                
            except Exception as e:
                logger.error(f"Database query error: {e}")
                raise RuntimeError(f"Database query failed: {str(e)}")
            
            # Format the applications for response
            formatted_applications = []
            try:
                for app in applications:
                    app_data = {
                        "_id": {"$oid": str(app.id)},  # Include MongoDB ObjectId
                        "application_id": str(app.application_id),
                        "timestamp": app.timestamp.isoformat(),
                        "loan_officer_id": app.loan_officer_id,
                        "status": app.status or "Pending",
                        "applicant_info": app.applicant_info.model_dump() if app.applicant_info else {},
                        "comaker_info": app.comaker_info.model_dump() if app.comaker_info else {},
                    }
                    
                    if app.prediction_result:
                        try:
                            pred_result = {
                                "final_credit_score": app.prediction_result.final_credit_score,
                                "default": app.prediction_result.default,
                                "probability_of_default": app.prediction_result.probability_of_default,
                                "status": app.prediction_result.status,
                            }
                            
                            # Handle loan recommendations
                            if hasattr(app.prediction_result, 'loan_recommendation') and app.prediction_result.loan_recommendation:
                                pred_result["loan_recommendation"] = [
                                    rec if isinstance(rec, dict) else rec.model_dump()
                                    for rec in app.prediction_result.loan_recommendation
                                ]
                                pred_result["recommendation_count"] = len(pred_result["loan_recommendation"])
                            else:
                                pred_result["loan_recommendation"] = []
                                pred_result["recommendation_count"] = 0
                                
                            app_data["prediction_result"] = pred_result
                        except Exception as e:
                            logger.error(f"Error formatting prediction result: {e}")
                            app_data["prediction_result"] = None
                    
                    formatted_applications.append(app_data)
            except Exception as e:
                logger.error(f"Error formatting application data: {e}")
                raise RuntimeError(f"Data formatting failed: {str(e)}")
            
            logger.info(f"Successfully formatted {len(formatted_applications)} applications")
            # Calculate counts using aggregation pipeline for better performance
            base_query = {"loan_officer_id": loan_officer_id} if loan_officer_id else {}
            pipeline = [
                {"$match": base_query},
                {"$group": {
                    "_id": "$status",
                    "count": {"$sum": 1}
                }}
            ]
            
            status_counts = {
                "total": 0,
                "approved": 0,
                "denied": 0,
                "cancelled": 0,
                "pending": 0
            }
            
            status_result = await LoanApplication.aggregate(pipeline).to_list()
            for result in status_result:
                status = result["_id"] or "Pending"  # Default to Pending if status is None
                count = result["count"]
                status_counts[status.lower()] = count
                status_counts["total"] += count
            
            return {
                "data": formatted_applications,
                "total": total,
                "page": skip // limit + 1 if limit > 0 else 1,
                "pages": (total + limit - 1) // limit if limit > 0 else 1,
                "counts": status_counts
            }
            
        except Exception as e:
            logger.error(f"Error retrieving loan applications: {e}")
            raise RuntimeError(f"Failed to retrieve loan applications: {str(e)}")

    async def update_application_status(
        self, 
        application_id: str,  # Changed from UUID to str
        new_status: str
    ) -> Optional[LoanApplication]:
        try:
            logger.info(f"Updating status for application {application_id} to: {new_status}")
            
            # Convert string ID to ObjectId and search by _id
            from bson import ObjectId
            try:
                object_id = ObjectId(application_id)
                logger.info(f"Converted to ObjectId: {object_id}")
            except Exception as e:
                logger.error(f"Invalid ObjectId format: {e}")
                return None
                
            application = await LoanApplication.find_one({"_id": object_id})
            if not application:
                logger.warning(f"Application {application_id} not found for status update")
                return None
            
            # Update the status field
            application.status = new_status
            
            # Also update the status in prediction_result if it exists
            if application.prediction_result:
                application.prediction_result.status = new_status
            
            await application.save()
            logger.info(f"Status updated successfully for application {application_id}")
                
            return application
            
        except Exception as e:
            logger.error(f"Error updating application status: {e}")
            raise RuntimeError(f"Failed to update application status: {str(e)}")

    async def get_loan_application_by_mongo_id(self, mongo_id: str) -> Optional[LoanApplication]:
        try:
            from bson import ObjectId
            logger.info(f"Retrieving loan application with mongo ID: {mongo_id}")
            
            application = await LoanApplication.find_one({"_id": ObjectId(mongo_id)})
            
            if application:
                logger.info(f"Loan application {mongo_id} found")
            else:
                logger.warning(f"Loan application {mongo_id} not found")
                
            return application
            
        except Exception as e:
            logger.error(f"Error retrieving loan application {mongo_id}: {e}")
            raise RuntimeError(f"Failed to retrieve loan application: {str(e)}")

    async def update_loan_application(
        self, 
        application_id: UUID, 
        update_data: Dict[str, Any],
        rerun_assessment: bool = False
    ) -> Optional[LoanApplication]:
        """Update an existing loan application with new data."""
        try:
            logger.info(f"Updating loan application {application_id} with new data")
            
            # Get the existing application
            application = await LoanApplication.find_one(LoanApplication.application_id == application_id)
            if not application:
                logger.warning(f"Application {application_id} not found")
                return None

            # Update each section if provided in update_data
            if 'applicant_info' in update_data:
                for key, value in update_data['applicant_info'].items():
                    setattr(application.applicant_info, key, value)

            if 'comaker_info' in update_data:
                for key, value in update_data['comaker_info'].items():
                    setattr(application.comaker_info, key, value)

            if 'model_input_data' in update_data:
                for key, value in update_data['model_input_data'].items():
                    setattr(application.model_input_data, key, value)

                if rerun_assessment:
                    # Re-run prediction with updated model input data
                    new_prediction = await self._run_prediction(application.model_input_data)
                    application.prediction_result = new_prediction

                    # Update recommendations if available
                    if self.recommendation_service:
                        new_recommendations = self.recommendation_service.get_loan_recommendations(
                            applicant_info=application.applicant_info,
                            model_input_data=application.model_input_data.model_dump()
                        )
                        application.prediction_result.loan_recommendation = new_recommendations

                    # Generate new AI explanation
                    if self.ai_service:
                        await self._generate_and_save_explanation(application)

            # Save updates
            await application.save()
            
            # Generate new AI explanation if model inputs changed
            if 'model_input_data' in update_data:
                await self._generate_and_save_explanation(application)
            
            logger.info(f"Application {application_id} updated successfully")
            return application

        except ValueError as e:
            logger.error(f"Validation error updating application {application_id}: {e}")
            raise ValueError(f"Invalid update data: {str(e)}")
        except Exception as e:
            logger.error(f"Error updating application {application_id}: {e}")
            raise RuntimeError(f"Failed to update application: {str(e)}")

    async def delete_loan_application(self, application_id: UUID) -> bool:
        try:
            logger.info(f"Deleting loan application with ID: {application_id}")
            
            # Use find_one to search by application_id field instead of _id
            application = await LoanApplication.find_one(LoanApplication.application_id == application_id)
            if not application:
                logger.warning(f"Application {application_id} not found for deletion")
                return False
            
            await application.delete()
            logger.info(f"Loan application {application_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting loan application {application_id}: {e}")
            raise RuntimeError(f"Failed to delete loan application: {str(e)}")

    async def regenerate_loan_recommendations(
        self, 
        application_id: UUID
    ) -> Optional[List[RecommendedProducts]]:
        try:
            logger.info(f"Regenerating loan recommendations for application: {application_id}")
            
            if not self.recommendation_service:
                logger.error("Loan recommendation service not available")
                return None
            
            # Use find_one to search by application_id field instead of _id
            application = await LoanApplication.find_one(LoanApplication.application_id == application_id)
            if not application:
                logger.warning(f"Application {application_id} not found")
                return None
            
            # Generate new recommendations
            new_recommendations = self.recommendation_service.get_loan_recommendations(
                applicant_info=application.applicant_info,
                model_input_data=application.model_input_data.model_dump()
            )
            
            # Update the application
            application.prediction_result.loan_recommendation = new_recommendations
            await application.save()
            
            logger.info(f"Regenerated {len(new_recommendations)} loan recommendations for application {application_id}")
            return new_recommendations
            
        except Exception as e:
            logger.error(f"Error regenerating loan recommendations: {e}")
            return None

    async def get_service_status(self) -> Dict[str, Any]:
        try:
            # Check if prediction service is available
            prediction_service_status = "healthy" if self.prediction_service else "unavailable"
            
            # Check if recommendation service is available
            recommendation_service_status = "healthy" if self.recommendation_service else "unavailable"
            
            # Get basic service info
            status_info = {
                "service": "loan-application-service",
                "status": "healthy",
                "prediction_service_status": prediction_service_status,
                "recommendation_service_status": recommendation_service_status,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
            
            # Add prediction service status if available
            if self.prediction_service:
                try:
                    pred_status = self.prediction_service.get_service_status()
                    status_info["prediction_service_details"] = pred_status
                except Exception as e:
                    logger.warning(f"Could not get prediction service status: {e}")
                    status_info["prediction_service_details"] = {"error": str(e)}
            
            # Add recommendation service status if available
            if self.recommendation_service:
                try:
                    rec_status = self.recommendation_service.get_service_status()
                    status_info["recommendation_service_details"] = rec_status
                except Exception as e:
                    logger.warning(f"Could not get recommendation service status: {e}")
                    status_info["recommendation_service_details"] = {"error": str(e)}
            
            return status_info
            
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            raise RuntimeError(f"Failed to get service status: {str(e)}")

    def _validate_loan_application_data(self, request_data: FullLoanApplicationRequest) -> None:
        if not request_data.applicant_info.full_name.strip():
            raise ValueError("Applicant full name is required")
        
        if not request_data.applicant_info.contact_number.strip():
            raise ValueError("Applicant contact number is required")
        
        if not request_data.applicant_info.address.strip():
            raise ValueError("Applicant address is required")
        
        if not request_data.comaker_info.full_name.strip():
            raise ValueError("Co-maker full name is required")
        
        if not request_data.comaker_info.contact_number.strip():
            raise ValueError("Co-maker contact number is required")
        
        model_data = request_data.model_input_data
        
        if model_data.Employment_Tenure_Months <= 0:
            raise ValueError("Employment tenure must be greater than 0")
        
        if model_data.Net_Salary_Per_Cutoff <= 0:
            raise ValueError("Net salary must be greater than 0")
        
        if model_data.Years_at_Current_Address < 0:
            raise ValueError("Years at current address cannot be negative")
        
        if model_data.Number_of_Dependents < 0:
            raise ValueError("Number of dependents cannot be negative")
        
        if model_data.Comaker_Employment_Tenure_Months <= 0:
            raise ValueError("Co-maker employment tenure must be greater than 0")
        
        if model_data.Comaker_Net_Salary_Per_Cutoff <= 0:
            raise ValueError("Co-maker net salary must be greater than 0")
        
        if not (0 <= model_data.Grace_Period_Usage_Rate <= 1):
            raise ValueError("Grace period usage rate must be between 0 and 1")
        
        if model_data.Late_Payment_Count < 0:
            raise ValueError("Late payment count cannot be negative")

    async def _run_prediction(self, model_input_data) -> PredictionResult:
        try:
            logger.info("Running prediction for loan application")
            
            # Check if prediction service is available
            if not self.prediction_service:
                raise RuntimeError("Prediction service is not available")
            
            # Create LoanApplicationRequest instance from the input data
            from app.schemas.loan_schema import LoanApplicationRequest
            
            # Convert model_input_data to dictionary if it's a Pydantic model
            input_dict = model_input_data.model_dump() if hasattr(model_input_data, 'model_dump') else dict(model_input_data)
            
            # Create LoanApplicationRequest instance
            loan_request = LoanApplicationRequest(**input_dict)
            
            # Run prediction with proper input format
            pod_result = self.prediction_service.predict(loan_request)
            pod = pod_result.get("probability_of_default")
            default = pod_result.get("default_prediction")
            
            if pod is None:
                raise RuntimeError("Prediction service returned invalid result")
            
            # Transform to credit score
            credit_score = self.prediction_service.transform_pod_to_credit_score(pod)
            
            # Create prediction result
            prediction_result = PredictionResult(
                final_credit_score=credit_score,
                default=default,
                probability_of_default=pod,
                loan_recommendation=[],  # Will be populated later
                status="Success"
            )
            
            logger.info(f"Prediction completed successfully. Credit Score: {credit_score}, POD: {pod:.4f}")
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")


def initialize_loan_application_service() -> Optional[LoanApplicationService]:
    try:
        logger.info("Initializing LoanApplicationService...")
        
        if not prediction_service:
            logger.error("Prediction service is not available for loan service initialization")
            return None
        
        if not ai_service:
            logger.warning("AIExplainabilityService is not available for loan service initialization")
        
        if not loan_recommendation_service:
            logger.warning("LoanRecommendationService is not available for loan service initialization")
        
        service = LoanApplicationService(
            prediction_service=prediction_service,
            recommendation_service=loan_recommendation_service
        )
        logger.info("LoanApplicationService initialized successfully")
        return service
        
    except Exception as e:
        logger.error(f"Failed to initialize LoanApplicationService: {e}")
        return None


# Initialize the service
loan_application_service = initialize_loan_application_service()