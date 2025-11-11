from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Dict, Any, Optional

class AIExplanation(BaseModel):
    technical_explanation: str = Field(..., description="Technical explanation of the AI model's prediction")
    business_explanation: str = Field(..., description="Business explanation of the AI model's prediction")
    customer_explanation: str = Field(..., description="Customer-friendly explanation of the AI model's prediction")
    risk_factors: str = Field(..., description="List of risk factors identified by the AI model")
    recommendations: str = Field(..., description="Recommendations based on the AI model's prediction")

class CulturalComponentScores(BaseModel):
    cultural_composite: float = Field(..., description="Overall cultural composite score")
    disaster_preparedness: float = Field(..., description="Disaster preparedness score")
    other_income: float = Field(..., description="Other income source score")
    comaker_relationship: float = Field(..., description="Co-maker relationship score")
    salary_frequency: float = Field(..., description="Salary frequency score")
    community_role: float = Field(..., description="Community role score")
    paluwagan: float = Field(..., description="Paluwagan participation score")
    dependents_impact: float = Field(..., description="Number of dependents impact score")

class EmploymentSectorEnum(str, Enum):
    public = "Public"
    private = "Private"

# Job is now an open string to allow any position value supplied by the client

class SalaryFrequencyEnum(str, Enum):
    monthly = "Monthly"
    bimonthly = "Bimonthly"
    biweekly = "Biweekly"
    weekly = "Weekly"

class HousingStatusEnum(str, Enum):
    owned = "Owned"
    rented = "Rented"

class ComakerRelationshipEnum(str, Enum):
    spouse = "Spouse"
    sibling = "Sibling"
    parent = "Parent"
    friend = "Friend"

class YesNoEnum(str, Enum):
    yes = "Yes"
    no = "No"

class OtherIncomeSourceEnum(str, Enum):
    none = "None"
    ofw_remittance = "OFW Remittance"
    freelance = "Freelance"
    business = "Business"

class DisasterPreparednessEnum(str, Enum):
    none = "None"
    savings = "Savings"
    insurance = "Insurance"
    community_plan = "Community Plan"

class PaluwaganParticipationEnum(str, Enum):
    never = "Never"
    rarely = "Rarely"
    sometimes = "Sometimes"
    frequently = "Frequently"

class CommunityRoleEnum(str, Enum):
    none = "None"
    member = "Member"
    leader = "Leader"
    multiple_leader = "Multiple Leader"

class ApplicationStatusEnum(str, Enum):
    pending = "Pending"
    approved = "Approved"
    denied = "Denied"
    cancelled = "Cancelled"

class LoanApplicationRequest(BaseModel):
    """Defines all the fields a loan officer must submit for a prediction."""
    Employment_Sector: EmploymentSectorEnum
    Employment_Tenure_Months: int = Field(..., gt=0)
    Net_Salary_Per_Cutoff: float = Field(..., gt=0)
    Salary_Frequency: SalaryFrequencyEnum
    Housing_Status: HousingStatusEnum
    Years_at_Current_Address: float = Field(..., ge=0)
    Household_Head: YesNoEnum
    Number_of_Dependents: int = Field(..., ge=0)
    Comaker_Relationship: ComakerRelationshipEnum
    Comaker_Employment_Tenure_Months: int = Field(..., gt=0)
    Comaker_Net_Salary_Per_Cutoff: float = Field(..., gt=0)
    Has_Community_Role: CommunityRoleEnum
    Paluwagan_Participation: PaluwaganParticipationEnum
    Other_Income_Source: OtherIncomeSourceEnum
    Disaster_Preparedness: DisasterPreparednessEnum
    Is_Renewing_Client: int = Field(0, ge=0, le=1)
    Grace_Period_Usage_Rate: float = Field(0.0, ge=0.0, le=1.0)
    Late_Payment_Count: int = Field(0, ge=0)
    Had_Special_Consideration: int = Field(0, ge=0, le=1)

class ApplicantInfo(BaseModel):
    """Schema for the applicant's personal info."""
    full_name: str
    contact_number: str
    address: str
    salary: str
    # Accept any job string (free-text) instead of limiting to an enum
    job: str

class CoMakerInfo(BaseModel):
    """Schema for the co-maker's personal info."""
    full_name: str
    contact_number: str

class FullLoanApplicationRequest(BaseModel):
    """The complete request body for creating a new loan application record."""
    applicant_info: ApplicantInfo
    comaker_info: CoMakerInfo
    model_input_data: LoanApplicationRequest

class RecommendedProducts(BaseModel):
    product_name: str
    is_top_recommendation: bool
    max_loanable_amount: float
    interest_rate_monthly: float
    term_in_months: int
    estimated_amortization_per_cutoff: float
    suitability_score: int

class PredictionResult(BaseModel):
    """Schema for the prediction result."""
    final_credit_score: int
    default: int = Field(ge=0, le=1, description="Default status of the applicant (0 for no, 1 for yes)")
    probability_of_default: float
    loan_recommendation: List[RecommendedProducts]
    status: ApplicationStatusEnum = Field(default=ApplicationStatusEnum.pending, description="Status of the prediction result")
    risk_level: Optional[str] = Field(None, description="Risk level assessment")
    threshold_used: Optional[float] = Field(None, description="Threshold used for binary classification")
    cultural_component_scores: Optional[CulturalComponentScores] = Field(None, description="Cultural component scores")
    detailed_cultural_analysis: Optional[Dict[str, Any]] = Field(None, description="Detailed cultural analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "final_credit_score": 750,
                "probability_of_default": 0.05,
                "loan_recommendation": "Approved",
                "status": "Pending"
            }
        }

class FullLoanApplicationResponse(BaseModel):
    """
    The complete response body after creating a new loan application.
    """
    message: str
    application_id: str
    timestamp: str
    status: ApplicationStatusEnum
    prediction_result: PredictionResult
    applicant_info: ApplicantInfo
    loan_officer_id: str
    ai_explanation: Optional[AIExplanation] = None

    recommended_products: List[RecommendedProducts] = []

    class Config:
        orm_mode = True