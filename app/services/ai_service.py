import logging
import json
from typing import Dict, Any, Optional

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from app.core import Settings
from app.schemas.loan_schema import LoanApplicationRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    if not Settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in the environment or settings.")
    genai.configure(api_key=Settings.GEMINI_API_KEY)
    _is_service_configured = True
    logger.info("Google AI SDK configured successfully.")
except (ValueError, AttributeError) as e:
    logger.critical(f"CRITICAL: Gemini API Key is not configured. AI Service will be disabled. Error: {e}")
    _is_service_configured = False


class AIExplainabilityService:
    """Service for generating AI-powered explanations of loan decisions."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """Initialize the AI explainability service."""
        try:
            if not _is_service_configured:
                raise RuntimeError("AI Service is not configured due to missing API key.")
            
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"AIExplainabilityService initialized successfully with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize AI service model: {e}", exc_info=True)
            raise RuntimeError(f"AI service initialization failed: {e}")

    async def generate_loan_explanation_async(
        self,
        application_data: LoanApplicationRequest,
        prediction_results: Dict[str, Any],
        feature_importance: Optional[Dict[str, float]] = None
    ) -> Dict[str, str]:
        """
        Generate comprehensive explanation of loan decision using AI asynchronously.
        Returns a dictionary of explanations.
        """
        try:
            analysis_data = self._prepare_analysis_data(
                application_data, prediction_results, feature_importance
            )
            explanations = {
                "technical_explanation": await self._call_ai_model(self._generate_technical_explanation(analysis_data)),
                "business_explanation": await self._call_ai_model(self._generate_business_explanation(analysis_data)),
                "customer_explanation": await self._call_ai_model(self._generate_customer_explanation(analysis_data)),
                "risk_factors": await self._call_ai_model(self._generate_risk_factors_explanation(analysis_data)),
                "recommendations": await self._call_ai_model(self._generate_recommendations(analysis_data)),
            }
            return explanations
        except Exception as e:
            logger.error(f"Failed to generate full loan explanation set: {e}", exc_info=True)
            error_msg = f"Error generating explanation: {e}"
            return {key: error_msg for key in ["technical_explanation", "business_explanation", "customer_explanation", "risk_factors", "recommendations"]}

    async def _call_ai_model(self, prompt: str) -> str:
        """Call the AI model with error handling."""
        try:
            generation_config = GenerationConfig(
                temperature=0.3
            )
            
            response = await self.model.generate_content_async(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            logger.error(f"AI model call failed: {e}", exc_info=True)
            return f"Error generating explanation: {str(e)}"

    def _prepare_analysis_data(
        self,
        application_data: LoanApplicationRequest,
        prediction_results: Dict[str, Any],
        feature_importance: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Prepare data for AI analysis."""
        app_dict = application_data.model_dump()
        monthly_salary = self._calculate_monthly_salary(
            app_dict.get("Net_Salary_Per_Cutoff", 0),
            app_dict.get("Salary_Frequency", "Monthly")
        )
        
        return {
            "application_data": app_dict,
            "prediction_results": prediction_results,
            "feature_importance": feature_importance or {},
            "derived_metrics": {
                "monthly_salary": monthly_salary,
            },
            "risk_assessment": self._assess_risk_factors(app_dict),
            "model_info": {
                "model_type": "Logistic Regression with Polynomial Features",
                "threshold": prediction_results.get("threshold_used", 0.5),
            }
        }

    def _generate_technical_explanation(self, analysis_data: Dict[str, Any]) -> str:
        return (
            "You are an expert data analyst. Provide a bulleted analysis of this credit score in technical terms, not exceeding 100 words. "
            "Instructions:\n"
            "- Create 4-5 key points.\n"
            "- **Start each point on a new line with the '•' character.**\n"
            "- Do not use any other bullet format (like '*' or '-').\n\n"
            "Include 4-5 key points focusing on:\n"
            "• Statistical model performance\n"
            "• Feature engineering insights\n"
            "• Key validation metrics\n"
            "• Model reliability indicators\n\n"
            "Use professional data science terminology. Ensure the entire response is a bulleted list and under 50 words.\n\n"
            f"Analysis Data: {json.dumps(analysis_data, default=str)}"
        )

    def _generate_business_explanation(self, analysis_data: Dict[str, Any]) -> str:
        return (
            "You are a senior loan officer. Provide a bulleted analysis of this credit score in business terms, not exceeding 100 words. "
            "Instructions:\n"
            "- Create 4-5 key points.\n"
            "- **Start each point on a new line with the '•' character.**\n"
            "- Do not use any other bullet format (like '*' or '-').\n\n"
            "Include 4-5 key points focusing on:\n"
            "• Operational impact on lending\n"
            "• Risk-adjusted pricing implications\n"
            "• Portfolio risk considerations\n"
            "• Business recommendations\n\n"
            "Use terminology suitable for banking executives. Ensure the entire response is a bulleted list and under 50 words.\n\n"
            f"Analysis Data: {json.dumps(analysis_data, default=str)}"
        )

    def _generate_customer_explanation(self, analysis_data: Dict[str, Any]) -> str:
        return (
            "You are explaining to a loan applicant. Provide a bulleted explanation of their credit score in simple terms, not exceeding 100 words. "
            "Instructions:\n"
            "- Create 4-5 key points.\n"
            "- **Start each point on a new line with the '•' character.**\n"
            "- Do not use any other bullet format (like '*' or '-').\n\n"
            "Include 4-5 key points focusing on:\n"
            "• What their score means\n"
            "• Main factors affecting their score\n"
            "• Positive aspects of their application\n"
            "• Areas for potential improvement\n\n"
            "Use friendly, supportive language. Ensure the entire response is a bulleted list and under 50 words.\n\n"
            f"Analysis Data: {json.dumps(analysis_data, default=str)}"
        )

    def _generate_risk_factors_explanation(self, analysis_data: Dict[str, Any]) -> str:
        return (
            "You are a risk assessment expert. Provide a bulleted analysis of the default risk factors, not exceeding 100 words. "
            "Instructions:\n"
            "- Create 4-5 key points.\n"
            "- **Start each point on a new line with the '•' character.**\n"
            "- Do not use any other bullet format (like '*' or '-').\n\n"
            "Include 4-5 key points focusing on:\n"
            "• Primary risk indicators\n"
            "• Cultural financial practices impact\n"
            "• Positive risk mitigants\n"
            "• Areas requiring attention\n\n"
            "Use professional risk management terminology. Ensure the entire response is a bulleted list and under 50 words.\n\n"
            f"Analysis Data: {json.dumps(analysis_data, default=str)}"
        )

    def _generate_recommendations(self, analysis_data: Dict[str, Any]) -> str:
        return (
            "You are a financial advisor. Provide bulleted recommendations to improve this credit score, not exceeding 100 words. "
            "Instructions:\n"
            "- Create 4-5 key points.\n"
            "- **Start each point on a new line with the '•' character.**\n"
            "- Do not use any other bullet format (like '*' or '-').\n\n"
            "Include 4-5 key points focusing on:\n"
            "• Immediate actions to take\n"
            "• Medium-term financial practices\n"
            "• Cultural financial strengths to leverage\n"
            "• Specific credit-building strategies\n\n"
            "Use encouraging, practical language. Ensure the entire response is a bulleted list and under 50 words.\n\n"
            f"Analysis Data: {json.dumps(analysis_data, default=str)}"
        )

    def _calculate_monthly_salary(self, salary_per_cutoff: float, frequency: str) -> float:
        multipliers = {"Monthly": 1, "Biweekly": 2, "Weekly": 4.33}
        return salary_per_cutoff * multipliers.get(frequency, 1)

    def _assess_risk_factors(self, app_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"employment_stability": "High" if app_data.get("Employment_Tenure_Months", 0) > 24 else "Medium"}

def initialize_ai_service() -> Optional[AIExplainabilityService]:
    """Initialize the AI explainability service."""
    if not _is_service_configured:
        logger.error("AI service will not be initialized because it was not configured.")
        return None
    try:
        return AIExplainabilityService()
    except Exception as e:
        logger.error(f"Failed to initialize AIExplainabilityService instance: {e}", exc_info=True)
        return None

ai_service = initialize_ai_service()