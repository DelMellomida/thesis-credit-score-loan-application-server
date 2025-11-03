from beanie.odm.fields import PydanticObjectId

def convert_objectid(obj):
    """Convert PydanticObjectId fields to strings."""
    if isinstance(obj, dict):
        return {key: convert_objectid(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid(item) for item in obj]
    elif isinstance(obj, PydanticObjectId):
        return str(obj)
    return obj

def build_loan_application_response(
    loan_application,
    prediction_result,
    recommended_products,
    current_user,
    ai_explanation,
    document_result
):
    # Convert any model data to dict first
    prediction_result_dict = prediction_result.model_dump() if prediction_result else None
    ai_explanation_dict = ai_explanation.model_dump() if ai_explanation else None
    document_result_dict = document_result.model_dump() if document_result else None
    
    response = {
        "message": "Loan application created successfully",
        "application_id": str(loan_application.application_id),
        "_id": str(loan_application.id) if hasattr(loan_application, 'id') else None,
        "timestamp": loan_application.timestamp.isoformat(),
        "status": "created",
        "prediction_result": prediction_result_dict,
        "recommended_products": convert_objectid(recommended_products),
        "applicant_info": {
            "full_name": loan_application.applicant_info.full_name,
            "contact_number": loan_application.applicant_info.contact_number,
            "address": loan_application.applicant_info.address,
            "salary": loan_application.applicant_info.salary,
            "job": loan_application.applicant_info.job
        },
        "loan_officer_id": str(loan_application.loan_officer_id) if isinstance(loan_application.loan_officer_id, PydanticObjectId) else loan_application.loan_officer_id,
        "created_by": {
            "email": current_user["email"],
            "full_name": current_user["full_name"]
        },
        "ai_explanation": convert_objectid(ai_explanation_dict),
        "document": convert_objectid(document_result_dict)
    }
    
    # Clean up any None values for cleaner response
    response = {k: v for k, v in response.items() if v is not None}
    return response
