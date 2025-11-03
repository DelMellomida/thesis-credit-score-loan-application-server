import logging
from typing import List, Dict, Any, Optional
from app.schemas.loan_schema import RecommendedProducts, ApplicantInfo as ApplicantInfoSchema
from app.loan_product import LOAN_PRODUCTS_CATALOG

logger = logging.getLogger(__name__)


class LoanRecommendationService:
    def __init__(self):
        self.loan_products = LOAN_PRODUCTS_CATALOG
        logger.info("LoanRecommendationService initialized")

    def get_loan_recommendations(
        self,
        applicant_info: ApplicantInfoSchema,
        model_input_data: Dict[str, Any]
    ) -> List[RecommendedProducts]:
        try:
            logger.info(f"Generating loan recommendations for applicant: {applicant_info.full_name}")
            
            # Get eligible products based on rules
            eligible_products = self._filter_eligible_products(applicant_info, model_input_data)
            
            if not eligible_products:
                logger.warning("No eligible products found for applicant")
                return []
            
            # Calculate recommendations with financial analysis
            recommendations = self._calculate_recommendations(
                eligible_products, 
                applicant_info, 
                model_input_data
            )
            
            logger.info(f"Generated {len(recommendations)} loan recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating loan recommendations: {e}")
            return []

    def _filter_eligible_products(
        self, 
        applicant_info: ApplicantInfoSchema, 
        model_input_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        eligible_products = []
        is_renewing = model_input_data.get("Is_Renewing_Client") == 1
        employment_sector = model_input_data.get("Employment_Sector", "")
        
        logger.info(f"Starting product filtering with {len(self.loan_products)} available products")
        logger.info(f"Client type: {'Renewing' if is_renewing else 'New'}")
        logger.info(f"Employment sector: {employment_sector}")
        logger.info(f"Applicant job: {getattr(applicant_info, 'job', 'Not specified')}")
        
        logger.info(f"Filtering products for employment sector: {employment_sector}, is_renewing: {is_renewing}")
        
        for product in self.loan_products:
            rules = product["eligibility_rules"]
            
            logger.info(f"Evaluating product: {product['product_name']}")
            
            # Rule: Check if the product is for new or existing clients
            client_type_eligible = self._check_client_type_eligibility(rules, is_renewing)
            if not client_type_eligible:
                logger.info(f"Product {product['product_name']} not eligible due to client type requirement: "
                         f"new_client_eligible={rules.get('is_new_client_eligible', True)}, is_renewing={is_renewing}")
                continue
            
            # Rule: Check employment sector
            sector_eligible = self._check_employment_sector_eligibility(rules, employment_sector)
            if not sector_eligible:
                logger.info(f"Product {product['product_name']} not eligible due to sector requirement: "
                         f"allowed_sectors={rules.get('employment_sector', [])}, applicant_sector={employment_sector}")
                continue
            
            # Rule: Check specific job (if applicable)
            job_eligible = self._check_job_eligibility(rules, applicant_info)
            if not job_eligible:
                logger.info(f"Product {product['product_name']} not eligible due to job requirement: "
                         f"required_jobs={rules.get('job', [])}, applicant_job={getattr(applicant_info, 'job', None)}")
                continue
                
            logger.info(f"Product {product['product_name']} is eligible")
            
            eligible_products.append(product)
            logger.debug(f"Product {product['product_name']} is eligible")
        
        logger.info(f"Found {len(eligible_products)} eligible products")
        return eligible_products

    def _check_client_type_eligibility(self, rules: Dict[str, Any], is_renewing: bool) -> bool:
        return rules.get("is_new_client_eligible", True) or is_renewing

    def _check_employment_sector_eligibility(self, rules: Dict[str, Any], employment_sector: str) -> bool:

        allowed_sectors = rules.get("employment_sector", [])
        if not allowed_sectors:  # If no restriction, allow all
            return True
        return employment_sector in allowed_sectors

    def _check_job_eligibility(self, rules: Dict[str, Any], applicant_info: ApplicantInfoSchema) -> bool:
        required_jobs = rules.get("job")
        if not required_jobs:  # If no job requirement, allow all
            return True
        
        applicant_job = getattr(applicant_info, 'job', None)
        if not applicant_job:
            return False
        
        return applicant_job in required_jobs

    def _calculate_recommendations(
        self, 
        eligible_products: List[Dict[str, Any]], 
        applicant_info: ApplicantInfoSchema,
        model_input_data: Dict[str, Any]
    ) -> List[RecommendedProducts]:
        ranked_products = []
        net_salary_per_cutoff = model_input_data.get("Net_Salary_Per_Cutoff", 0)
        salary_frequency = model_input_data.get("Salary_Frequency", "Bimonthly")
        
        # Calculate maximum affordable amortization (50% of net salary)
        max_affordable_amortization = net_salary_per_cutoff * 0.50
        
        # Convert per-cutoff amortization to monthly
        cutoffs_per_month = 2 if salary_frequency in ["Biweekly", "Bimonthly"] else 1
        max_affordable_monthly_amortization = max_affordable_amortization * cutoffs_per_month
        
        logger.info(f"Max affordable amortization per cutoff: {max_affordable_amortization}")
        
        for product in eligible_products:
            # Calculate maximum loan principal for this product
            max_principal = self._calculate_max_loan_principal(
                max_amortization=max_affordable_monthly_amortization,
                monthly_interest_rate=product["interest_rate_monthly"],
                term_in_months=product["max_term_months"]
            )
            
            # Apply product limits
            final_loanable_amount = min(max_principal, product["max_loanable_amount"])
            final_loanable_amount = max(0, final_loanable_amount)
            
            # Calculate suitability score
            suitability_score = self._calculate_suitability_score(
                product, 
                final_loanable_amount, 
                applicant_info, 
                model_input_data
            )
            
            # Only include products with meaningful loan amounts
            if final_loanable_amount > 0:
                ranked_products.append({
                    "product_data": product,
                    "suitability_score": suitability_score,
                    "final_loanable_amount": round(final_loanable_amount, -2),  # Round to nearest 100
                    "estimated_amortization_per_cutoff": round(max_affordable_amortization, 2),
                })
        
        # Sort by score, highest first
        sorted_products = sorted(ranked_products, key=lambda x: x["suitability_score"], reverse=True)
        
        # Convert to RecommendedProducts schema
        recommendations = []
        for i, item in enumerate(sorted_products):
            prod_data = item["product_data"]
            
            recommendations.append(
                RecommendedProducts(
                    product_name=prod_data["product_name"],
                    is_top_recommendation=(i == 0),
                    max_loanable_amount=item["final_loanable_amount"],
                    interest_rate_monthly=prod_data["interest_rate_monthly"],
                    term_in_months=prod_data["max_term_months"],
                    estimated_amortization_per_cutoff=item["estimated_amortization_per_cutoff"],
                    suitability_score=item["suitability_score"]
                )
            )
        
        return recommendations

    def _calculate_max_loan_principal(
        self, 
        max_amortization: float, 
        monthly_interest_rate: float, 
        term_in_months: int
    ) -> float:
        rate = monthly_interest_rate / 100
        
        # Handle edge case where rate is 0
        if rate == 0:
            return max_amortization * term_in_months
        
        # Using the present value of annuity formula
        # PV = PMT * [(1 - (1 + r)^(-n)) / r]
        discount_factor = (1 - (1 + rate) ** (-term_in_months)) / rate
        principal = max_amortization * discount_factor
        
        return principal

    def _calculate_suitability_score(
        self, 
        product: Dict[str, Any], 
        final_loanable_amount: float,
        applicant_info: ApplicantInfoSchema,
        model_input_data: Dict[str, Any]
    ) -> int:
        score = 100
        is_renewing = model_input_data.get("Is_Renewing_Client") == 1
        
        # Lower interest rate increases score
        score -= product["interest_rate_monthly"] * 10
        
        # Higher potential loan amount increases score
        score += (final_loanable_amount / 10000)
        
        # Longer term increases score (more flexibility)
        score += product["max_term_months"] / 12
        
        # Major bonus for specialized loans that match the job
        rules = product["eligibility_rules"]
        if (rules.get("job") and 
            hasattr(applicant_info, 'job') and 
            applicant_info.job in rules["job"]):
            score += 20
        
        # Bonus for existing client products if applicable
        if not rules.get("is_new_client_eligible", True) and is_renewing:
            score += 10
        
        return max(0, int(score))  # Ensure non-negative score

    def get_service_status(self) -> Dict[str, Any]:
        try:
            return {
                "service": "loan-recommendation-service",
                "status": "healthy",
                "available_products": len(self.loan_products),
                "version": "1.0.0"
            }
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            return {
                "service": "loan-recommendation-service",
                "status": "error",
                "error": str(e)
            }

    def add_product(self, product: Dict[str, Any]) -> bool:
        try:
            # Validate product structure (basic validation)
            required_fields = ["product_name", "interest_rate_monthly", "max_term_months", 
                             "max_loanable_amount", "eligibility_rules"]
            
            if not all(field in product for field in required_fields):
                logger.error(f"Product missing required fields: {required_fields}")
                return False
            
            self.loan_products.append(product)
            logger.info(f"Added new product: {product['product_name']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding product: {e}")
            return False

    def remove_product(self, product_name: str) -> bool:
        try:
            original_length = len(self.loan_products)
            self.loan_products = [p for p in self.loan_products if p["product_name"] != product_name]
            
            if len(self.loan_products) < original_length:
                logger.info(f"Removed product: {product_name}")
                return True
            else:
                logger.warning(f"Product not found: {product_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing product: {e}")
            return False


def initialize_loan_recommendation_service() -> Optional[LoanRecommendationService]:
    try:
        logger.info("Initializing LoanRecommendationService...")
        
        service = LoanRecommendationService()
        logger.info("LoanRecommendationService initialized successfully")
        return service
        
    except Exception as e:
        logger.error(f"Failed to initialize LoanRecommendationService: {e}")
        return None


# Initialize the service
loan_recommendation_service = initialize_loan_recommendation_service()