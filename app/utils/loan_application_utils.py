from app.database.models.loan_application_model import (
    ApplicantInfo as DbApplicantInfo,
    CoMakerInfo as DbCoMakerInfo,
    ModelInputData as DbModelInputData
)

def convert_applicant_info(applicant_info):
    return DbApplicantInfo(**applicant_info.model_dump())

def convert_comaker_info(comaker_info):
    return DbCoMakerInfo(**comaker_info.model_dump())

def convert_model_input_data(model_input_data):
    return DbModelInputData(**model_input_data.model_dump())
