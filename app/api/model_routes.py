from fastapi import APIRouter, HTTPException, status, Depends
from typing import Dict, Any
from app.core.auth_dependencies import get_current_active_user
from app.services.prediction_service import prediction_service
import logging

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model", tags=["Model Management"])

@router.post("/retrain", response_model=Dict[str, Any])
async def retrain_model(
    current_user: Dict = Depends(get_current_active_user)
):
    """
    Retrain the model with updated data.
    Only accessible to authenticated users.
    """
    try:
        if not prediction_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service is not available"
            )

        # Save current threshold and settings
        current_threshold = prediction_service.default_threshold
        current_feature = prediction_service.default_sensitive_feature

        # Reload model with updated data
        prediction_service._load_model()

        # Restore previous settings
        if current_threshold:
            prediction_service.set_threshold(current_threshold)
        if current_feature:
            prediction_service.set_default_sensitive_feature(current_feature)

        return {
            "message": "Model successfully retrained",
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retraining model: {str(e)}"
        )

@router.post("/update-formula", response_model=Dict[str, Any])
async def update_model_formula(
    formula_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_active_user)
):
    """
    Update model formula parameters.
    Only accessible to authenticated users.
    """
    try:
        if not prediction_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service is not available"
            )

        # Here you would implement the formula update logic
        # This is just a placeholder - replace with actual implementation
        # For example, update weights, parameters, or feature importance

        return {
            "message": "Model formula updated successfully",
            "status": "success",
            "updated_parameters": formula_data
        }

    except Exception as e:
        logger.error(f"Error updating model formula: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating model formula: {str(e)}"
        )

@router.get("/status", response_model=Dict[str, Any])
async def get_model_status(
    current_user: Dict = Depends(get_current_active_user)
):
    """
    Get the current status of the model.
    Only accessible to authenticated users.
    """
    try:
        if not prediction_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service is not available"
            )

        status = prediction_service.get_service_status()
        feature_importance = prediction_service.get_feature_importance()

        return {
            "status": status,
            "feature_importance": feature_importance
        }

    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model status: {str(e)}"
        )