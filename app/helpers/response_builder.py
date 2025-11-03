def build_loan_application_response(
    loan_application,
    prediction_result,
    recommended_products,
    current_user,
    ai_explanation,
    document_result
):
    return {
        "message": "Loan application created successfully",
        "application_id": str(loan_application.application_id),
        "timestamp": loan_application.timestamp.isoformat(),
        "status": "created",
        "prediction_result": {
            "final_credit_score": prediction_result.final_credit_score,
            "default": prediction_result.default,
            "probability_of_default": prediction_result.probability_of_default,
            "status": prediction_result.status
        },
        "recommended_products": recommended_products,
        "applicant_info": {
            "full_name": loan_application.applicant_info.full_name,
            "contact_number": loan_application.applicant_info.contact_number,
            "address": loan_application.applicant_info.address,
            "salary": loan_application.applicant_info.salary,
            "job": loan_application.applicant_info.job
        },
        "loan_officer_id": loan_application.loan_officer_id,
        "created_by": {
            "email": current_user["email"],
            "full_name": current_user["full_name"]
        },
        "ai_explanation": ai_explanation,
        "document": document_result.model_dump() if document_result else None
    }
