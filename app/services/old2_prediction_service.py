# import joblib
# import os
# import pandas as pd
# import numpy as np
# from pydantic import BaseModel, Field
# from typing import Dict, Any, Optional, List
# import logging
# from app.schemas.loan_schema import LoanApplicationRequest

# # Import enhanced transformers from local transformers module
# from scripts.transformers import (
#     EnhancedCreditScoringTransformer,
#     EnhancedCreditScoringConfig,
#     validate_loan_application_schema,
#     get_available_features,
#     create_loan_application_from_dict,
#     ClientType
# )

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # File paths with fallback options
# MODEL_DIR = './models'
# FALLBACK_MODEL_DIR = '../models'

# def get_model_path(filename):
#     """Get model path with fallback directory checking."""
#     primary_path = os.path.join(MODEL_DIR, filename)
#     fallback_path = os.path.join(FALLBACK_MODEL_DIR, filename)
    
#     if os.path.exists(primary_path):
#         return primary_path
#     elif os.path.exists(fallback_path):
#         return fallback_path
#     else:
#         return primary_path

# # Model paths - using the original naming scheme
# MODEL_PATH = get_model_path('enhanced_credit_model.pkl')
# ENCODER_PATH = get_model_path('encoder.pkl')
# SCALER_PATH = get_model_path('feature_scaler.pkl')
# POLY_PATH = get_model_path('polynomial_features.pkl')
# SELECTOR_PATH = get_model_path('feature_selector.pkl')
# FEATURES_INFO_PATH = get_model_path('feature_info.pkl')
# HYBRID_MODEL_INFO_PATH = get_model_path('model_info.pkl')
# TRANSFORMER_INSTANCES_PATH = get_model_path('transformers.pkl')  # Changed from transformer_instances.pkl

# # Additional enhanced model paths (for when new models are available)
# ENHANCED_MODEL_INFO_PATH = get_model_path('model_info.pkl')


# class PredictionService:
#     """
#     Enhanced Prediction Service with feature isolation and data leakage prevention.
    
#     Maintains original interface while providing enhanced capabilities:
#     - Client-type specific scoring
#     - Mathematical feature isolation
#     - Data leakage prevention
#     - Backwards compatibility
#     """
    
#     def __init__(self, default_threshold: float = 0.5):
#         self.model = None
#         self.encoder = None
#         self.scaler = None
#         self.poly = None
#         self.selector = None
#         self.features_info = None
#         self.hybrid_model_info = None
#         self.enhanced_transformer = None
#         self.config = None
#         self.default_threshold = default_threshold
#         self._load_models()

#     def _load_models(self):
#         """Load all required model components with enhanced transformers."""
#         try:
#             logger.info("Loading models with enhanced transformers...")
            
#             if not os.path.exists(MODEL_DIR) and not os.path.exists(FALLBACK_MODEL_DIR):
#                 raise FileNotFoundError(f"Model directories '{MODEL_DIR}' and '{FALLBACK_MODEL_DIR}' do not exist.")
            
#             # Load main ML components
#             self.model = joblib.load(MODEL_PATH)
#             self.scaler = joblib.load(SCALER_PATH)
#             self.poly = joblib.load(POLY_PATH)
#             self.selector = joblib.load(SELECTOR_PATH)
            
#             # Load encoder (should be None for new model)
#             try:
#                 encoder_data = joblib.load(ENCODER_PATH)
#                 if isinstance(encoder_data, dict) and encoder_data.get('type') == 'none':
#                     self.encoder = None
#                 else:
#                     self.encoder = encoder_data
#             except Exception as e:
#                 logger.warning(f"Could not load encoder: {e}")
#                 self.encoder = None
            
#             # Load features info
#             try:
#                 if os.path.exists(FEATURES_INFO_PATH):
#                     self.features_info = joblib.load(FEATURES_INFO_PATH)
#                 else:
#                     # Fallback to enhanced feature structure
#                     self.features_info = get_available_features()
#             except Exception as e:
#                 logger.warning(f"Could not load features info: {e}")
#                 self.features_info = get_available_features()
            
#             # Load hybrid model info
#             try:
#                 if os.path.exists(HYBRID_MODEL_INFO_PATH):
#                     self.hybrid_model_info = joblib.load(HYBRID_MODEL_INFO_PATH)
#                 elif os.path.exists(ENHANCED_MODEL_INFO_PATH):
#                     self.hybrid_model_info = joblib.load(ENHANCED_MODEL_INFO_PATH)
#                 else:
#                     self.hybrid_model_info = {
#                         'model_type': 'enhanced_feature_isolation',
#                         'feature_isolation_enabled': True,
#                         'version': '2.0.0'
#                     }
#             except Exception as e:
#                 logger.warning(f"Could not load model info: {e}")
#                 self.hybrid_model_info = {
#                     'model_type': 'enhanced_feature_isolation',
#                     'feature_isolation_enabled': True,
#                     'version': '2.0.0'
#                 }
            
#             # Load enhanced transformer instances
#             try:
#                 if os.path.exists(TRANSFORMER_INSTANCES_PATH):
#                     transformer_info = joblib.load(TRANSFORMER_INSTANCES_PATH)
#                     if isinstance(transformer_info, dict):
#                         self.enhanced_transformer = transformer_info.get('enhanced_transformer')
#                         self.config = transformer_info.get('config')
#                     else:
#                         # Handle case where transformer is saved directly
#                         self.enhanced_transformer = transformer_info
#                         self.config = EnhancedCreditScoringConfig()
#                     logger.info("Loaded saved enhanced transformer instances")
#                 else:
#                     raise FileNotFoundError("Transformer instances not found")
#             except Exception as e:
#                 logger.warning(f"Could not load saved transformer instances: {e}")
#                 # Initialize fresh enhanced transformers
#                 self.config = EnhancedCreditScoringConfig()
#                 self.enhanced_transformer = EnhancedCreditScoringTransformer(self.config)
#                 logger.info("Initialized fresh enhanced transformer")
            
#             # Validate transformer
#             if not isinstance(self.enhanced_transformer, EnhancedCreditScoringTransformer):
#                 logger.warning("Transformer is not enhanced type, creating new one")
#                 self.config = EnhancedCreditScoringConfig()
#                 self.enhanced_transformer = EnhancedCreditScoringTransformer(self.config)
            
#             logger.info("All model components loaded successfully with enhanced transformers.")
            
#         except Exception as e:
#             logger.error(f"Failed to load models: {e}")
#             raise RuntimeError(f"Model loading failed: {e}")

#     def predict(self, input_data: LoanApplicationRequest) -> Dict[str, Any]:
#         """Make a prediction using enhanced feature isolation system."""
#         try:
#             if not self._is_service_ready():
#                 raise RuntimeError("PredictionService is not properly initialized.")
            
#             # Convert to DataFrame
#             df = pd.DataFrame([input_data.model_dump()])
            
#             # Validate and fix schema issues
#             df_fixed, validation_issues = validate_loan_application_schema(df)
#             if validation_issues:
#                 logger.warning(f"Schema issues found and fixed: {validation_issues}")
#                 df = df_fixed
            
#             # Apply enhanced transformations
#             df_transformed = self.enhanced_transformer.transform(df)
            
#             # Determine client type for conditional logic
#             is_renewing = bool(df['Is_Renewing_Client'].iloc[0]) if 'Is_Renewing_Client' in df.columns else False
#             client_type = "renewing" if is_renewing else "new"
            
#             # Get features that match training (excluding problematic features)
#             available_features = self._get_prediction_features(df_transformed)
            
#             logger.info(f"Using {len(available_features)} features for {client_type} client prediction")
#             logger.info(f"Features: {available_features}")
            
#             # Prepare feature matrix - should be all numeric from training fix
#             X = df_transformed[available_features].fillna(0)
            
#             # Verify all features are numeric (should be after training fix)
#             for col in X.columns:
#                 if X[col].dtype == 'object':
#                     logger.warning(f"Converting {col} to numeric")
#                     X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
#             # Apply preprocessing pipeline (same as training)
#             X_scaled = self.scaler.transform(X)
#             X_poly = self.poly.transform(X_scaled)
#             X_selected = self.selector.transform(X_poly)
            
#             # Make prediction
#             prediction_proba = self.model.predict_proba(X_selected)
#             probability_of_default = float(prediction_proba[0, 1])
#             binary_prediction = int(probability_of_default >= self.default_threshold)
            
#             # Get enhanced component scores with feature isolation details
#             component_scores = self._extract_component_scores(df_transformed, client_type)
            
#             # Get feature isolation summary
#             isolation_summary = self._get_feature_isolation_summary(client_type)
            
#             return {
#                 "probability_of_default": probability_of_default,
#                 "default_prediction": binary_prediction,
#                 "threshold_used": self.default_threshold,
#                 "component_scores": component_scores,
#                 "client_type": client_type,
#                 "feature_isolation_applied": True,
#                 "model_type": "enhanced_feature_isolation",
#                 "feature_isolation_summary": isolation_summary,
#                 "bias_reduction_active": True,
#                 "mathematical_constraints_active": True,
#                 "features_used": available_features,
#                 "features_count": len(available_features)
#             }
            
#         except Exception as e:
#             logger.error(f"Prediction failed: {e}")
#             logger.error(f"Error details: {str(e)}")
            
#             # Add helpful debugging info
#             if "feature_names" in str(e).lower():
#                 logger.error("This appears to be a feature mismatch error.")
#                 logger.error("Ensure the model was retrained with the correct feature exclusions.")
#                 logger.error("Expected features should exclude: Is_Renewing_Client, Client_Type, Credit_Risk_Score, Default")
                
#             raise RuntimeError(f"Prediction failed: {e}")

#     def _get_prediction_features(self, df_transformed: pd.DataFrame) -> List[str]:
#         """Get features for prediction based on what the model was actually trained on."""
        
#         # FIRST: Try to use saved feature info from training
#         if self.features_info and 'available_features' in self.features_info:
#             saved_features = self.features_info['available_features']
#             available_features = [f for f in saved_features if f in df_transformed.columns]
#             if available_features:
#                 logger.info(f"Using saved training features: {len(available_features)} features")
#                 return available_features
        
#         # FALLBACK: If no saved features, use the same logic as training
#         component_features = [
#             'Credit_Behavior_Score', 'Financial_Stability_Score', 'Cultural_Context_Score'
#         ]
        
#         raw_features = [
#             'Net_Salary_Per_Cutoff', 'Employment_Tenure_Months', 'Number_of_Dependents',
#             'Years_at_Current_Address', 'Late_Payment_Count', 'Grace_Period_Usage_Rate'
#         ]
        
#         # CRITICAL: Same exclusions as training
#         excluded_features = [
#             'Is_Renewing_Client', 'Client_Type', 'Credit_Risk_Score', 'Default'
#         ]
        
#         candidate_features = component_features + raw_features
#         final_features = [f for f in candidate_features 
#                         if f in df_transformed.columns and f not in excluded_features]
        
#         logger.info(f"Using fallback features: {len(final_features)} features")
#         return final_features


#     def _extract_component_scores(self, df_transformed: pd.DataFrame, client_type: str) -> Dict[str, Any]:
#         """Extract component scores with feature isolation details."""
#         scores = {
#             'financial_stability': float(df_transformed['Financial_Stability_Score'].iloc[0]),
#             'cultural_context': float(df_transformed['Cultural_Context_Score'].iloc[0]),
#             'client_type': client_type
#         }
        
#         # Add credit behavior for renewing clients
#         if client_type == "renewing" and 'Credit_Behavior_Score' in df_transformed.columns:
#             scores['credit_behavior'] = float(df_transformed['Credit_Behavior_Score'].iloc[0])
        
#         # Add final score
#         if 'Credit_Risk_Score' in df_transformed.columns:
#             scores['final_credit_risk'] = float(df_transformed['Credit_Risk_Score'].iloc[0])
        
#         # Add effective scores showing conditional logic
#         late_payments = df_transformed.get('Late_Payment_Count', pd.Series([0])).iloc[0] if 'Late_Payment_Count' in df_transformed.columns else 0
#         grace_usage = df_transformed.get('Grace_Period_Usage_Rate', pd.Series([0])).iloc[0] if 'Grace_Period_Usage_Rate' in df_transformed.columns else 0
        
#         if client_type == "renewing":
#             scores['effective_late_payments'] = float(late_payments)
#             scores['effective_grace_usage'] = float(grace_usage)
#         else:
#             scores['effective_late_payments'] = 0.0  # Neutralized for new clients
#             scores['effective_grace_usage'] = 0.0   # Neutralized for new clients
        
#         # Add component weights if available
#         if self.config:
#             client_enum = ClientType.RENEWING if client_type == "renewing" else ClientType.NEW
#             try:
#                 client_config = self.config.get_client_config(client_enum)
#                 scores['component_weights'] = {
#                     comp_name: comp_config.max_contribution_pct 
#                     for comp_name, comp_config in client_config.components.items()
#                 }
#                 scores['max_cultural_impact'] = client_config.components['cultural_context'].max_contribution_pct
#             except Exception as e:
#                 logger.warning(f"Could not get component weights: {e}")
        
#         return scores

#     def _get_feature_isolation_summary(self, client_type: str) -> Dict[str, Any]:
#         """Get summary of feature isolation constraints."""
#         if client_type == "new":
#             return {
#                 "feature_isolation_active": True,
#                 "client_type": "new",
#                 "scoring_weights": {
#                     "financial_stability": "80%",
#                     "cultural_context": "20%"
#                 },
#                 "cultural_constraints": {
#                     "community_role_max_impact": "0.4%",
#                     "paluwagan_max_impact": "0.6%",
#                     "total_cultural_cap": "20%"
#                 },
#                 "leakage_prevention": {
#                     "perfect_predictors": "Mathematically constrained",
#                     "payment_history": "Neutralized for new clients",
#                     "bias_mitigation": "Cultural factors severely limited"
#                 }
#             }
#         else:
#             return {
#                 "feature_isolation_active": True,
#                 "client_type": "renewing",
#                 "scoring_weights": {
#                     "credit_behavior": "60%",
#                     "financial_stability": "37%", 
#                     "cultural_context": "3%"
#                 },
#                 "cultural_constraints": {
#                     "community_role_max_impact": "0.015%",
#                     "paluwagan_max_impact": "0.024%",
#                     "total_cultural_cap": "3%"
#                 },
#                 "leakage_prevention": {
#                     "perfect_predictors": "Completely neutralized",
#                     "payment_history": "Primary factor for existing clients",
#                     "bias_mitigation": "Cultural factors extremely limited"
#                 }
#             }

#     def get_cultural_analysis(self, input_data: LoanApplicationRequest) -> Dict[str, Any]:
#         """Get detailed analysis using enhanced feature isolation system."""
#         try:
#             if not self.enhanced_transformer:
#                 raise RuntimeError("Enhanced transformer not loaded.")
            
#             # Convert to DataFrame
#             df = pd.DataFrame([input_data.model_dump()])
            
#             # Validate schema
#             df_fixed, validation_issues = validate_loan_application_schema(df)
#             if validation_issues:
#                 logger.warning(f"Schema issues found and fixed: {validation_issues}")
#                 df = df_fixed
            
#             # Get detailed explanation using enhanced system
#             applicant_series = df.iloc[0]
#             explanation = self.enhanced_transformer.get_score_explanation(applicant_series)
            
#             return {
#                 "detailed_explanation": explanation,
#                 "component_breakdown": explanation.get('component_contributions', {}),
#                 "feature_isolation": explanation.get('feature_isolation_summary', {}),
#                 "bias_reduction_notes": explanation.get('leakage_prevention', {}),
#                 "model_fairness": {
#                     "mathematical_constraints": "All features capped to prevent dominance",
#                     "cultural_bias_prevention": "Perfect predictors neutralized",
#                     "client_type_fairness": "Different scoring for new vs renewing clients"
#                 },
#                 "client_type": explanation.get('client_type', 'unknown'),
#                 "score_interpretation": explanation.get('score_interpretation', 'No interpretation available')
#             }
            
#         except Exception as e:
#             logger.error(f"Error in cultural analysis: {e}")
#             raise RuntimeError(f"Cultural analysis failed: {e}")

#     def get_score_explanation(self, input_data: LoanApplicationRequest) -> Dict[str, Any]:
#         """Get human-readable score explanation using enhanced feature isolation."""
#         try:
#             if not self.enhanced_transformer:
#                 raise RuntimeError("Enhanced transformer not loaded.")
            
#             # Convert to DataFrame
#             df = pd.DataFrame([input_data.model_dump()])
            
#             # Validate schema
#             df_fixed, validation_issues = validate_loan_application_schema(df)
#             if validation_issues:
#                 logger.warning(f"Schema issues found and fixed: {validation_issues}")
#                 df = df_fixed
            
#             # Get full explanation
#             applicant_series = df.iloc[0]
#             explanation = self.enhanced_transformer.get_score_explanation(applicant_series)
            
#             # Add service-specific context
#             explanation['prediction_service_info'] = {
#                 'feature_isolation_active': True,
#                 'data_leakage_prevention': True,
#                 'mathematical_constraints': True,
#                 'model_version': self.hybrid_model_info.get('version', '2.0.0')
#             }
            
#             return explanation
            
#         except Exception as e:
#             logger.error(f"Error getting score explanation: {e}")
#             raise RuntimeError(f"Score explanation failed: {e}")

#     def get_feature_caps_summary(self) -> Dict[str, Any]:
#         """Get comprehensive summary of all feature caps and constraints."""
#         if not self.config:
#             return {
#                 "feature_caps_active": False,
#                 "note": "Enhanced config not available"
#             }
        
#         try:
#             caps_summary = self.config.get_feature_caps_summary()
            
#             # Add leakage mitigation details
#             leakage_details = {
#                 "community_role": {
#                     "original_impact": "Perfect predictor (0% default rate)",
#                     "constrained_impact": {
#                         "new_clients": "Max 0.4% of final score",
#                         "renewing_clients": "Max 0.015% of final score"
#                     }
#                 },
#                 "paluwagan_participation": {
#                     "original_impact": "66% default rate difference",
#                     "constrained_impact": {
#                         "new_clients": "Max 0.6% of final score", 
#                         "renewing_clients": "Max 0.024% of final score"
#                     }
#                 }
#             }
            
#             return {
#                 "feature_caps_active": True,
#                 "caps_summary": caps_summary,
#                 "leakage_mitigation": leakage_details,
#                 "mathematical_constraints": {
#                     "individual_features": "Each feature has strict contribution caps",
#                     "component_level": "Components cannot exceed specified percentages",
#                     "client_type_specific": "Different constraints for new vs renewing clients"
#                 }
#             }
            
#         except Exception as e:
#             logger.warning(f"Error getting feature caps: {e}")
#             return {
#                 "feature_caps_active": True,
#                 "error": f"Could not retrieve caps summary: {e}"
#             }

#     def set_threshold(self, threshold: float) -> None:
#         """Set the threshold for binary classification."""
#         if not (0 <= threshold <= 1):
#             raise ValueError("Threshold must be between 0 and 1.")
#         self.default_threshold = threshold
#         logger.info(f"Default threshold set to {threshold}")

#     def _is_service_ready(self) -> bool:
#         """Check if all required components are loaded."""
#         return all([
#             self.model is not None,
#             self.scaler is not None,
#             self.poly is not None,
#             self.selector is not None,
#             self.enhanced_transformer is not None
#         ])

#     @staticmethod
#     def transform_pod_to_credit_score(pod: float, min_score: int = 300, max_score: int = 850) -> int:
#         """Transform probability of default to credit score."""
#         try:
#             if not isinstance(pod, (int, float)):
#                 raise TypeError("Probability of default must be a number.")
            
#             if not (0 <= pod <= 1):
#                 raise ValueError("Probability of default must be between 0 and 1.")
            
#             credit_score = min_score + (max_score - min_score) * (1 - pod)
#             credit_score = int(round(credit_score))
#             credit_score = max(min_score, min(max_score, credit_score))
            
#             return credit_score
            
#         except Exception as e:
#             logger.error(f"Error transforming probability to credit score: {e}")
#             raise

#     def get_service_status(self) -> Dict[str, Any]:
#         """Get the current status of the prediction service."""
#         return {
#             "is_ready": self._is_service_ready(),
#             "model_loaded": self.model is not None,
#             "encoder_loaded": self.encoder is not None,
#             "scaler_loaded": self.scaler is not None,
#             "poly_loaded": self.poly is not None,
#             "selector_loaded": self.selector is not None,
#             "enhanced_transformer_loaded": self.enhanced_transformer is not None,
#             "config_loaded": self.config is not None,
#             "current_threshold": self.default_threshold,
#             "model_type": self.hybrid_model_info.get('model_type', 'enhanced_feature_isolation'),
#             "feature_isolation_enabled": self.hybrid_model_info.get('feature_isolation_enabled', True),
#             "bias_reduction_active": True,
#             "mathematical_constraints_active": True,
#             "version": self.hybrid_model_info.get('version', '2.0.0'),
#             "client_type_configurations": {
#                 "new_clients": "Financial (80%) + Cultural (20%)",
#                 "renewing_clients": "Credit (60%) + Financial (37%) + Cultural (3%)"
#             },
#             "data_leakage_prevention": {
#                 "community_role_constrained": True,
#                 "paluwagan_constrained": True,
#                 "perfect_predictors_neutralized": True
#             }
#         }


# # Initialize the prediction service
# def initialize_prediction_service(threshold: float = 0.5) -> Optional[PredictionService]:
#     """Initialize the prediction service with enhanced transformers."""
#     try:
#         logger.info("Initializing Prediction Service with enhanced feature isolation...")
#         service = PredictionService(default_threshold=threshold)
#         logger.info("Prediction Service initialized successfully with enhanced transformers.")
#         return service
#     except Exception as e:
#         logger.error(f"Failed to initialize PredictionService: {e}")
#         return None

# # Initialize the service
# prediction_service = initialize_prediction_service()

# # Complete workflow function with enhanced feature isolation
# def get_credit_assessment(input_data, threshold: float = 0.5) -> Optional[Dict[str, Any]]:
#     """Complete workflow using enhanced feature isolation system."""
#     if prediction_service is None:
#         logger.error("PredictionService is not available.")
#         return None
    
#     try:
#         # Set threshold if different from current
#         if prediction_service.default_threshold != threshold:
#             prediction_service.set_threshold(threshold)
        
#         # Get prediction using enhanced feature isolation
#         prediction_result = prediction_service.predict(input_data)
#         pod = prediction_result["probability_of_default"]
#         binary_prediction = prediction_result["default_prediction"]
#         component_scores = prediction_result["component_scores"]
        
#         # Convert to credit score
#         credit_score = PredictionService.transform_pod_to_credit_score(pod)
        
#         # Add risk assessment
#         if credit_score >= 750:
#             risk_level = "Low Risk"
#         elif credit_score >= 650:
#             risk_level = "Medium Risk"
#         else:
#             risk_level = "High Risk"
        
#         # Recommendation based on binary prediction
#         recommendation = "Decline" if binary_prediction == 1 else "Approve"
        
#         # Get detailed analysis using enhanced system
#         try:
#             detailed_analysis = prediction_service.get_cultural_analysis(input_data)
#         except Exception as e:
#             logger.warning(f"Failed to get detailed analysis: {e}")
#             detailed_analysis = {"error": "Detailed analysis unavailable"}
        
#         # Get feature caps summary
#         try:
#             feature_caps = prediction_service.get_feature_caps_summary()
#         except Exception as e:
#             logger.warning(f"Failed to get feature caps: {e}")
#             feature_caps = {"error": "Feature caps unavailable"}
        
#         # Determine client type
#         client_type = component_scores.get('client_type', 'unknown')
        
#         return {
#             "probability_of_default": pod,
#             "default_prediction": binary_prediction,
#             "credit_score": credit_score,
#             "risk_level": risk_level,
#             "recommendation": recommendation,
#             "threshold_used": threshold,
#             "component_scores": component_scores,
#             "detailed_analysis": detailed_analysis,
#             "feature_caps_summary": feature_caps,
#             "model_type": "enhanced_feature_isolation",
#             "feature_isolation_enabled": True,
#             "client_type": client_type,
#             "bias_reduction_active": True,
#             "mathematical_constraints_active": True,
#             "feature_isolation_summary": prediction_result.get("feature_isolation_summary", {}),
#             "fairness_notes": {
#                 "new_clients": "Assessed using Financial (80%) + Cultural (20%) with severe cultural constraints",
#                 "renewing_clients": "Assessed using Credit (60%) + Financial (37%) + Cultural (3%) with extreme cultural constraints",
#                 "bias_mitigation": "Cultural factors mathematically capped to prevent discrimination",
#                 "perfect_predictors": "Community role and Paluwagan completely neutralized",
#                 "payment_history": "Neutralized for new clients, primary factor for renewing clients",
#                 "transparency": "Full feature contribution breakdown available with mathematical caps"
#             }
#         }
        
#     except Exception as e:
#         logger.error(f"Credit assessment failed: {e}")
#         return None


# # =================================================================
# # BACKWARDS COMPATIBILITY WRAPPER
# # =================================================================

# class CulturalScoreTransformer:
#     """
#     Backwards compatibility wrapper that maintains the old interface
#     while using the enhanced feature isolation system internally.
#     """
    
#     def __init__(self):
#         if prediction_service and prediction_service.enhanced_transformer:
#             self.enhanced_transformer = prediction_service.enhanced_transformer
#             self.config = prediction_service.config
#         else:
#             self.config = EnhancedCreditScoringConfig()
#             self.enhanced_transformer = EnhancedCreditScoringTransformer(self.config)
        
#         # Legacy attributes for compatibility
#         self.cultural_scoring_rules = self.config.leakage_mitigation_features if hasattr(self.config, 'leakage_mitigation_features') else {}
#         self.binary_scoring_rules = {
#             'community_role_boost': 0.02,  # Severely limited
#             'paluwagan_boost': 0.03,       # Severely limited
#             'household_head_boost': 0.10
#         }
#         self.dependents_penalty_per_person = -0.01  # Limited penalty
    
#     def transform_cultural_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Legacy method using enhanced feature isolation internally."""
#         return self.enhanced_transformer.transform(df)
    
#     def calculate_cultural_composite_score(self, df: pd.DataFrame) -> np.ndarray:
#         """Legacy method using enhanced feature isolation internally."""
#         transformed_df = self.enhanced_transformer.transform(df)
#         return transformed_df['Cultural_Context_Score'].values
    
#     def get_scoring_explanation(self) -> Dict[str, Any]:
#         """Legacy method returning enhanced system configuration."""
#         if self.config and hasattr(self.config, 'get_feature_caps_summary'):
#             return self.config.get_feature_caps_summary()
#         else:
#             return {
#                 "enhanced_mode": True,
#                 "feature_isolation": True,
#                 "data_leakage_prevention": True,
#                 "cultural_constraints": {
#                     "community_role": "Severely limited to prevent bias",
#                     "paluwagan": "Severely limited to prevent leakage",
#                     "mathematical_caps": "All features mathematically constrained"
#                 }
#             }


# class FinancialFeatureEngineer:
#     """
#     Backwards compatibility wrapper for financial feature engineering.
#     """
    
#     def __init__(self):
#         if prediction_service and prediction_service.enhanced_transformer:
#             self.enhanced_transformer = prediction_service.enhanced_transformer
#         else:
#             config = EnhancedCreditScoringConfig()
#             self.enhanced_transformer = EnhancedCreditScoringTransformer(config)
    
#     def create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Create features using enhanced feature isolation system."""
#         transformed_df = self.enhanced_transformer.transform(df)
        
#         # Add legacy feature names for compatibility
#         if 'Financial_Stability_Score' in transformed_df.columns:
#             transformed_df['Employment_Stability_Score'] = transformed_df['Financial_Stability_Score'] * 0.3
#             transformed_df['Income_Capacity_Score'] = transformed_df['Financial_Stability_Score'] * 0.4
#             transformed_df['Affordability_Score'] = transformed_df['Financial_Stability_Score'] * 0.3
        
#         return transformed_df
    
#     def get_feature_list(self) -> Dict[str, list]:
#         """Get feature list for backwards compatibility."""
#         return get_available_features()


# def get_all_expected_features() -> Dict[str, list]:
#     """Backwards compatibility function returning enhanced feature structure."""
#     return get_available_features()


# def validate_schema_compliance(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
#     """Schema validation using enhanced transformers module."""
#     return validate_loan_application_schema(df)


# # =================================================================
# # DEMONSTRATION AND TESTING
# # =================================================================

# def demonstrate_feature_isolation():
#     """Demonstrate the enhanced feature isolation system in prediction service."""
    
#     print("=== ENHANCED FEATURE ISOLATION DEMONSTRATION ===")
    
#     # Sample data with different client types showing feature isolation
#     sample_applications = [
#         {
#             # New Client - Cultural factors limited
#             'Employment_Sector': 'Public',
#             'Employment_Tenure_Months': 24,
#             'Net_Salary_Per_Cutoff': 22000,
#             'Comaker_Net_Salary_Per_Cutoff': 18000,
#             'Number_of_Dependents': 1,
#             'Is_Renewing_Client': 0,  # NEW CLIENT
#             'Late_Payment_Count': 3,  # Will be neutralized
#             'Grace_Period_Usage_Rate': 0.5,  # Will be neutralized
#             'Had_Special_Consideration': 0,
#             'Years_at_Current_Address': 3.0,
#             'Housing_Status': 'Owned',
#             'Paluwagan_Participation': 'Yes',  # Was perfect predictor - now constrained
#             'Has_Community_Role': 'Yes',  # Was perfect predictor - now constrained
#             'Household_Head': 'Yes',
#             'Disaster_Preparedness': 'Insurance',
#             'Other_Income_Source': 'Business',
#             'Comaker_Relationship': 'Spouse',
#             'Salary_Frequency': 'Biweekly'
#         },
#         {
#             # Renewing Client - Payment history considered, cultural extremely limited
#             'Employment_Sector': 'Private',
#             'Employment_Tenure_Months': 36,
#             'Net_Salary_Per_Cutoff': 20000,
#             'Comaker_Net_Salary_Per_Cutoff': 0,
#             'Number_of_Dependents': 2,
#             'Is_Renewing_Client': 1,  # RENEWING CLIENT
#             'Late_Payment_Count': 1,  # Will be considered at full weight
#             'Grace_Period_Usage_Rate': 0.2,  # Will be considered at full weight
#             'Had_Special_Consideration': 0,
#             'Years_at_Current_Address': 2.0,
#             'Housing_Status': 'Rented',
#             'Paluwagan_Participation': 'Yes',  # Extremely limited impact (0.024%)
#             'Has_Community_Role': 'Yes',  # Extremely limited impact (0.015%)
#             'Household_Head': 'Yes',
#             'Disaster_Preparedness': 'Savings',
#             'Other_Income_Source': 'None',
#             'Comaker_Relationship': 'Friend',
#             'Salary_Frequency': 'Monthly'
#         }
#     ]
    
#     for i, app_data in enumerate(sample_applications):
#         client_type = "New Client" if app_data['Is_Renewing_Client'] == 0 else "Renewing Client"
        
#         print(f"\n--- {client_type} (Application {i+1}) ---")
#         print(f"Has Community Role: {app_data['Has_Community_Role']} (Perfect predictor)")
#         print(f"Paluwagan Participation: {app_data['Paluwagan_Participation']} (66% difference)")
#         print(f"Late Payments: {app_data['Late_Payment_Count']}")
#         print(f"Grace Usage: {app_data['Grace_Period_Usage_Rate']}")
        
#         if app_data['Is_Renewing_Client'] == 0:
#             print("→ FEATURE ISOLATION FOR NEW CLIENT:")
#             print("  • Community Role impact: LIMITED to 0.4% max")
#             print("  • Paluwagan impact: LIMITED to 0.6% max") 
#             print("  • Payment history: NEUTRALIZED (0% impact)")
#             print("  • Cultural total: CAPPED at 20%")
#             print("  • Focus: Financial capacity (80%)")
#         else:
#             print("→ FEATURE ISOLATION FOR RENEWING CLIENT:")
#             print("  • Community Role impact: EXTREMELY LIMITED to 0.015% max")
#             print("  • Paluwagan impact: EXTREMELY LIMITED to 0.024% max")
#             print("  • Payment history: PRIMARY FACTOR (60% weight)")
#             print("  • Cultural total: EXTREMELY CAPPED at 3%")
#             print("  • Focus: Credit behavior (60%) + Financial (37%)")
    
#     return sample_applications


# if __name__ == "__main__":
#     print("="*80)
#     print("ENHANCED PREDICTION SERVICE WITH FEATURE ISOLATION")
#     print("="*80)
    
#     # Test service initialization
#     if prediction_service and prediction_service._is_service_ready():
#         print("✅ PredictionService initialized with enhanced feature isolation")
        
#         status = prediction_service.get_service_status()
#         print(f"✅ Model Type: {status['model_type']}")
#         print(f"✅ Feature Isolation: {status['feature_isolation_enabled']}")
#         print(f"✅ Bias Reduction: {status['bias_reduction_active']}")
#         print(f"✅ Mathematical Constraints: {status['mathematical_constraints_active']}")
#         print(f"✅ Version: {status['version']}")
        
#         print("\n📊 Client Type Configurations:")
#         for client_type, config in status['client_type_configurations'].items():
#             print(f"   {client_type.replace('_', ' ').title()}: {config}")
        
#         print("\n🛡️ Data Leakage Prevention:")
#         for measure, active in status['data_leakage_prevention'].items():
#             print(f"   {measure.replace('_', ' ').title()}: {'✅' if active else '❌'}")
        
#         # Test feature caps
#         try:
#             caps_summary = prediction_service.get_feature_caps_summary()
#             if caps_summary.get('feature_caps_active'):
#                 print("\n🔒 Feature Isolation Summary:")
#                 leakage = caps_summary.get('leakage_mitigation', {})
                
#                 if 'community_role' in leakage:
#                     print("   Community Role Constraints:")
#                     print(f"     Original: {leakage['community_role']['original_impact']}")
#                     constrained = leakage['community_role']['constrained_impact']
#                     print(f"     New Clients: {constrained['new_clients']}")
#                     print(f"     Renewing: {constrained['renewing_clients']}")
                
#                 if 'paluwagan_participation' in leakage:
#                     print("   Paluwagan Constraints:")
#                     print(f"     Original: {leakage['paluwagan_participation']['original_impact']}")
#                     constrained = leakage['paluwagan_participation']['constrained_impact']
#                     print(f"     New Clients: {constrained['new_clients']}")
#                     print(f"     Renewing: {constrained['renewing_clients']}")
#         except Exception as e:
#             print(f"   Could not retrieve feature caps: {e}")
    
#     else:
#         print("❌ PredictionService failed to initialize")
#         print("   Check model files and transformer dependencies")
    
#     print("\n--- Enhanced Feature Isolation Features ---")
#     print("✅ Mathematical feature caps prevent any single feature dominance")
#     print("✅ Perfect predictors (Community Role) completely neutralized")
#     print("✅ Data leakage (Paluwagan) mathematically constrained")
#     print("✅ Client-type specific scoring with different cultural constraints")
#     print("✅ Payment history neutralized for new clients (fair assessment)")
#     print("✅ Cultural factors severely limited to prevent bias")
#     print("✅ Component-level caps ensure balanced scoring")
#     print("✅ Full transparency with detailed explanations")
    
#     # Demonstrate feature isolation
#     demonstrate_feature_isolation()
    
#     print("\n" + "="*80)
#     print("🎯 ENHANCED PREDICTION SERVICE STATUS")
#     print("="*80)
    
#     if prediction_service and prediction_service._is_service_ready():
#         print("✅ FULLY OPERATIONAL WITH ENHANCED FEATURE ISOLATION")
#         print("   - Complete mathematical feature isolation implemented")
#         print("   - Data leakage prevention verified and active") 
#         print("   - Client-type specific scoring with bias prevention")
#         print("   - Cultural factor constraints preventing discrimination")
#         print("   - Perfect predictor neutralization active")
#         print("   - Payment history conditional logic operational")
#         print("   - Component-level contribution caps enforced")
#         print("   - Full transparency and explainability available")
#         print("   - Backwards compatibility maintained")
#         print("   - Production-ready with comprehensive error handling")
#     else:
#         print("❌ SERVICE UNAVAILABLE")
#         print("   - Check model files in ./models/ or ../models/")
#         print("   - Ensure scripts/transformers.py is available")
#         print("   - Verify enhanced transformer dependencies")
    
#     print("="*80)

import joblib
import os
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import logging
from app.schemas.loan_schema import LoanApplicationRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = './models'
MODEL_PATH = os.path.join(MODEL_DIR, 'credit_model.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'encoder.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
POLY_PATH = os.path.join(MODEL_DIR, 'polynomial_features.pkl')
SELECTOR_PATH = os.path.join(MODEL_DIR, 'feature_selector.pkl')
FEATURES_INFO_PATH = os.path.join(MODEL_DIR, 'feature_info.pkl')

class PredictionService:
    def __init__(self, default_threshold: float = 0.5):
        self.model = None
        self.encoder = None
        self.scaler = None
        self.poly = None
        self.selector = None
        self.features_info = None
        self.default_threshold = default_threshold  # Threshold for binary classification
        self._load_models()

    def _load_models(self):
        """Load all required model components with comprehensive error handling."""
        try:
            logger.info("Loading models...")
            
            # Check if model directory exists
            if not os.path.exists(MODEL_DIR):
                raise FileNotFoundError(f"Model directory '{MODEL_DIR}' does not exist.")
            
            # Check if all required files exist
            required_files = {
                'model': MODEL_PATH,
                'encoder': ENCODER_PATH,
                'scaler': SCALER_PATH,
                'polynomial_features': POLY_PATH,
                'feature_selector': SELECTOR_PATH,
                'features_info': FEATURES_INFO_PATH
            }
            
            missing_files = []
            for name, path in required_files.items():
                if not os.path.exists(path):
                    missing_files.append(f"{name} ({path})")
            
            if missing_files:
                raise FileNotFoundError(f"Missing model files: {', '.join(missing_files)}")
            
            # Load models with individual error handling
            try:
                self.model = joblib.load(MODEL_PATH)
                logger.info("Model loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load main model: {e}")
            
            try:
                self.encoder = joblib.load(ENCODER_PATH)
                logger.info("Encoder loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load encoder: {e}")
            
            try:
                self.scaler = joblib.load(SCALER_PATH)
                logger.info("Scaler loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load scaler: {e}")
            
            try:
                self.poly = joblib.load(POLY_PATH)
                logger.info("Polynomial features loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load polynomial features: {e}")
            
            try:
                self.selector = joblib.load(SELECTOR_PATH)
                logger.info("Feature selector loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load feature selector: {e}")
            
            try:
                self.features_info = joblib.load(FEATURES_INFO_PATH)
                logger.info("Features info loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load features info: {e}")
            
            # Validate features_info structure
            if not isinstance(self.features_info, dict):
                raise ValueError("features_info must be a dictionary.")
            
            required_keys = ['categorical_features', 'numerical_features']
            missing_keys = [key for key in required_keys if key not in self.features_info]
            if missing_keys:
                raise ValueError(f"features_info missing required keys: {missing_keys}")
            
            logger.info("All models loaded successfully.")
            
        except FileNotFoundError as e:
            logger.error(f"File not found error: {e}")
            raise RuntimeError(f"Model files not found: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading models: {e}")
            raise RuntimeError(f"Failed to load models: {e}")
        
    def predict(self, input_data: LoanApplicationRequest) -> Dict[str, Any]:
        """Make a prediction based on the input data with comprehensive error handling."""
        try:
            # Validate service is properly initialized
            if not self._is_service_ready():
                raise RuntimeError("PredictionService is not properly initialized. All model components must be loaded.")
            
            # Validate input data
            if input_data is None:
                raise ValueError("Input data cannot be None.")
            
            # Convert to DataFrame
            try:
                df = pd.DataFrame([input_data.model_dump()])
            except Exception as e:
                raise ValueError(f"Failed to convert input data to DataFrame: {e}")
            
            # Validate required features are present
            categorical_features = self.features_info['categorical_features']
            numerical_features = self.features_info['numerical_features']
            
            missing_categorical = [f for f in categorical_features if f not in df.columns]
            missing_numerical = [f for f in numerical_features if f not in df.columns]
            
            if missing_categorical:
                raise ValueError(f"Missing categorical features: {missing_categorical}")
            if missing_numerical:
                raise ValueError(f"Missing numerical features: {missing_numerical}")
            
            # Check for missing values
            if df[categorical_features + numerical_features].isnull().any().any():
                raise ValueError("Input data contains missing values.")
            
            # Transform categorical features
            try:
                encoded_categorical = self.encoder.transform(df[categorical_features])
            except Exception as e:
                raise RuntimeError(f"Failed to encode categorical features: {e}")
            
            # Transform numerical features
            try:
                scaled_numerical = self.scaler.transform(df[numerical_features])
            except Exception as e:
                raise RuntimeError(f"Failed to scale numerical features: {e}")
            
            # Combine features
            try:
                transformed_features = np.hstack([scaled_numerical, encoded_categorical.toarray()])
            except Exception as e:
                raise RuntimeError(f"Failed to combine features: {e}")
            
            # Apply polynomial transformation
            try:
                poly_features = self.poly.transform(transformed_features)
            except Exception as e:
                raise RuntimeError(f"Failed to apply polynomial transformation: {e}")
            
            # Apply feature selection
            try:
                selected_features = self.selector.transform(poly_features)
            except Exception as e:
                raise RuntimeError(f"Failed to apply feature selection: {e}")
            
            # Make prediction
            try:
                prediction_proba = self.model.predict_proba(selected_features)
                if prediction_proba.shape[1] < 2:
                    raise RuntimeError("Model does not provide probability for positive class.")
                probability_of_default = float(prediction_proba[0, 1])
                
                # Make binary prediction based on threshold
                binary_prediction = int(probability_of_default >= self.default_threshold)
                
            except Exception as e:
                raise RuntimeError(f"Failed to make prediction: {e}")
            
            # Validate probability is in valid range
            if not (0 <= probability_of_default <= 1):
                logger.warning(f"Probability of default {probability_of_default} is outside [0,1] range. Clipping.")
                probability_of_default = max(0, min(1, probability_of_default))
                # Recalculate binary prediction after clipping
                binary_prediction = int(probability_of_default >= self.default_threshold)
            
            return {
                "probability_of_default": probability_of_default,
                "default_prediction": binary_prediction,
                "threshold_used": self.default_threshold
            }
            
        except ValueError as e:
            logger.error(f"Input validation error: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"Runtime error during prediction: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for binary classification."""
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1.")
        self.default_threshold = threshold
        logger.info(f"Default threshold set to {threshold}")

    def _is_service_ready(self) -> bool:
        """Check if all required components are loaded."""
        return all([
            self.model is not None,
            self.encoder is not None,
            self.scaler is not None,
            self.poly is not None,
            self.selector is not None,
            self.features_info is not None
        ])

    @staticmethod
    def transform_pod_to_credit_score(pod: float, min_score: int = 300, max_score: int = 850) -> int:
        """Transform probability of default to credit score with error handling."""
        try:
            # Validate inputs
            if not isinstance(pod, (int, float)):
                raise TypeError("Probability of default must be a number.")
            
            if not (0 <= pod <= 1):
                raise ValueError("Probability of default must be between 0 and 1.")
            
            if not isinstance(min_score, int) or not isinstance(max_score, int):
                raise TypeError("Min and max scores must be integers.")
            
            if min_score >= max_score:
                raise ValueError("Min score must be less than max score.")
            
            if min_score < 0 or max_score < 0:
                raise ValueError("Credit scores must be non-negative.")
            
            # Calculate credit score
            credit_score = min_score + (max_score - min_score) * (1 - pod)
            
            # Round and convert to integer
            credit_score = int(round(credit_score))
            
            # Ensure score is within bounds (safety check)
            credit_score = max(min_score, min(max_score, credit_score))
            
            return credit_score
            
        except (TypeError, ValueError) as e:
            logger.error(f"Error transforming probability to credit score: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in credit score transformation: {e}")
            raise RuntimeError(f"Credit score transformation failed: {e}")

    def get_service_status(self) -> Dict[str, Any]:
        """Get the current status of the prediction service."""
        return {
            "is_ready": self._is_service_ready(),
            "model_loaded": self.model is not None,
            "encoder_loaded": self.encoder is not None,
            "scaler_loaded": self.scaler is not None,
            "poly_loaded": self.poly is not None,
            "selector_loaded": self.selector is not None,
            "features_info_loaded": self.features_info is not None,
            "current_threshold": self.default_threshold,
            "categorical_features": self.features_info.get('categorical_features', []) if self.features_info else [],
            "numerical_features": self.features_info.get('numerical_features', []) if self.features_info else []
        }

# Initialize the prediction service with proper error handling
def initialize_prediction_service(threshold: float = 0.5) -> Optional[PredictionService]:
    """Initialize the prediction service with proper error handling."""
    try:
        logger.info("Initializing PredictionService...")
        service = PredictionService(default_threshold=threshold)
        logger.info("PredictionService initialized successfully.")
        return service
    except Exception as e:
        logger.error(f"Failed to initialize PredictionService: {e}")
        return None

# Initialize the service
prediction_service = initialize_prediction_service()

# Complete workflow function for your model (updated to include binary prediction)
def get_credit_assessment(input_data, threshold: float = 0.5) -> Optional[Dict[str, Any]]:
    """Complete workflow: input -> POD -> binary prediction -> credit score."""
    if prediction_service is None:
        logger.error("PredictionService is not available.")
        return None
    
    try:
        # Set threshold if different from current
        if prediction_service.default_threshold != threshold:
            prediction_service.set_threshold(threshold)
        
        # Step 1: Get probability of default and binary prediction
        prediction_result = prediction_service.predict(input_data)
        pod = prediction_result["probability_of_default"]
        binary_prediction = prediction_result["default_prediction"]
        
        # Step 2: Convert to credit score
        credit_score = PredictionService.transform_pod_to_credit_score(pod)
        
        # Step 3: Add risk assessment
        if credit_score >= 750:
            risk_level = "Low Risk"
        elif credit_score >= 650:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
        
        # Step 4: Recommendation based on binary prediction
        recommendation = "Decline" if binary_prediction == 1 else "Approve"
        
        return {
            "probability_of_default": pod,
            "default_prediction": binary_prediction,
            "credit_score": credit_score,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "threshold_used": threshold
        }
        
    except Exception as e:
        logger.error(f"Credit assessment failed: {e}")
        return None