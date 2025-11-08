import pickle
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from enum import Enum
import logging
from app.schemas.loan_schema import LoanApplicationRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = './models'
FAIR_MODEL_PATH = os.path.join(MODEL_DIR, 'new_client_model_fair.pkl')
STANDARD_MODEL_PATH = os.path.join(MODEL_DIR, 'new_client_model.pkl')


class FairnessAwareModel:
    
    def __init__(self, base_model, fairness_postprocessors):
        self.base_model = base_model
        self.fairness_postprocessors = fairness_postprocessors
        
    def predict(self, X, apply_fairness=True, sensitive_feature_name=None):
        if not apply_fairness or not sensitive_feature_name:
            return self.base_model.predict(X)
        
        if sensitive_feature_name in self.fairness_postprocessors:
            postprocessor = self.fairness_postprocessors[sensitive_feature_name]
            return postprocessor.predict(
                X, 
                sensitive_features=X[sensitive_feature_name]
            )
        else:
            return self.base_model.predict(X)
            
    def predict_proba(self, X):
        return self.base_model.predict_proba(X)


class PredictionService:
    def __init__(self, default_threshold: float = 0.5, default_sensitive_feature: Optional[str] = None):
        self.model = None
        self.is_fairness_aware = False
        self.default_threshold = default_threshold
        self.default_sensitive_feature = default_sensitive_feature
        self.available_sensitive_features = []
        self._load_model()

    # Loads the trained model pipeline with support for fairness-aware models
    def _load_model(self):
        try:
            logger.info("Loading model...")
            
            if not os.path.exists(MODEL_DIR):
                raise FileNotFoundError(f"Model directory '{MODEL_DIR}' does not exist.")
            
            model_loaded = False
            if os.path.exists(FAIR_MODEL_PATH):
                try:
                    logger.info("Attempting to load fairness-aware model...")
                    
                    try:
                        with open(FAIR_MODEL_PATH, 'rb') as f:
                            import pickle
                            import sys
                            
                            import __main__
                            __main__.FairnessAwareModel = FairnessAwareModel
                            
                            self.model = pickle.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to load with custom unpickler: {e}")
                        raise e
                    
                    if hasattr(self.model, 'fairness_postprocessors'):
                        self.is_fairness_aware = True
                        self.available_sensitive_features = list(self.model.fairness_postprocessors.keys())
                        logger.info(f"Fairness-aware model loaded successfully. Available sensitive features: {self.available_sensitive_features}")
                        model_loaded = True
                    else:
                        logger.info("Model at fair path is not fairness-aware, treating as standard model.")
                        model_loaded = True
                        
                except Exception as e:
                    logger.warning(f"Failed to load fairness-aware model: {e}")
                    logger.info("Will attempt to reconstruct from components...")
                    
                    try:
                        self._attempt_fairness_model_reconstruction()
                        if self.model is not None:
                            model_loaded = True
                    except Exception as reconstruction_error:
                        logger.warning(f"Failed to reconstruct fairness model: {reconstruction_error}")
            
            if not model_loaded and os.path.exists(STANDARD_MODEL_PATH):
                try:
                    logger.info("Loading standard model...")
                    with open(STANDARD_MODEL_PATH, 'rb') as f:
                        self.model = pickle.load(f)
                    model_loaded = True
                    logger.info("Standard model loaded successfully.")
                except Exception as e:
                    raise RuntimeError(f"Failed to load standard model: {e}")
            
            if not model_loaded:
                raise FileNotFoundError("No model file found (neither fair nor standard).")
            
            if self.is_fairness_aware:
                if not hasattr(self.model, 'base_model'):
                    raise ValueError("Fairness-aware model missing base_model attribute.")
                if not hasattr(self.model.base_model, 'predict') or not hasattr(self.model.base_model, 'predict_proba'):
                    raise ValueError("Base model does not have required predict methods.")
            else:
                if not hasattr(self.model, 'predict') or not hasattr(self.model, 'predict_proba'):
                    raise ValueError("Model does not have required predict methods.")
            
            logger.info("Model validation successful.")
            
        except FileNotFoundError as e:
            logger.error(f"File not found error: {e}")
            raise RuntimeError(f"Model file not found: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    # Attempts to reconstruct fairness model from separate component files
    def _attempt_fairness_model_reconstruction(self):
        try:
            base_model_path = os.path.join(MODEL_DIR, 'base_model.pkl')
            fairness_components_path = os.path.join(MODEL_DIR, 'fairness_components.pkl')
            
            if os.path.exists(base_model_path) and os.path.exists(fairness_components_path):
                logger.info("Attempting to reconstruct fairness model from components...")
                
                with open(base_model_path, 'rb') as f:
                    base_model = pickle.load(f)
                
                with open(fairness_components_path, 'rb') as f:
                    fairness_components = pickle.load(f)
                
                self.model = FairnessAwareModel(base_model, fairness_components)
                self.is_fairness_aware = True
                self.available_sensitive_features = list(fairness_components.keys())
                
                logger.info("Successfully reconstructed fairness-aware model from components.")
            else:
                metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    if metadata.get('fairness_applied', False):
                        logger.warning("Metadata indicates fairness model but components not found. Loading as standard model.")
                
                raise FileNotFoundError("Fairness model components not found")
                
        except Exception as e:
            logger.warning(f"Failed to reconstruct fairness model: {e}")
            raise e

    # Validates that input data contains all required features
    def _validate_input_data(self, df: pd.DataFrame) -> None:
        required_features = [
            'Employment_Sector', 'Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
            'Salary_Frequency', 'Housing_Status', 'Years_at_Current_Address',
            'Number_of_Dependents', 'Comaker_Employment_Tenure_Months',
            'Comaker_Net_Salary_Per_Cutoff', 'Other_Income_Source',
            'Household_Head', 'Comaker_Relationship', 'Has_Community_Role',
            'Paluwagan_Participation', 'Disaster_Preparedness'
        ]
        
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        null_features = [f for f in required_features if df[f].isnull().any()]
        if null_features:
            raise ValueError(f"Null values found in features: {null_features}")

    # Preprocesses input data to match training data format
    def _preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        
        categorical_features = [
            'Employment_Sector', 'Salary_Frequency', 'Housing_Status', 
            'Household_Head', 'Comaker_Relationship', 'Has_Community_Role',
            'Paluwagan_Participation', 'Other_Income_Source', 'Disaster_Preparedness'
        ]
        
        for col in categorical_features:
            if col in df_processed.columns and df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].astype(str).str.strip()
                
                if col == 'Housing_Status':
                    df_processed[col] = df_processed[col].str.capitalize()
                elif col == 'Employment_Sector':
                    df_processed[col] = df_processed[col].str.capitalize()
                elif col in ['Household_Head', 'Has_Community_Role', 'Paluwagan_Participation']:
                    df_processed[col] = df_processed[col].str.capitalize()
        
        numerical_cols = [
            'Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
            'Years_at_Current_Address', 'Number_of_Dependents',
            'Comaker_Employment_Tenure_Months', 'Comaker_Net_Salary_Per_Cutoff'
        ]
        
        for col in numerical_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].clip(lower=0)
        
        if 'Number_of_Dependents' in df_processed.columns:
            df_processed['Number_of_Dependents'] = np.minimum(df_processed['Number_of_Dependents'], 5)
        
        df_processed['Proactive_Financial_Health_Score'] = 0
        
        disaster_prep_savings_insurance = df_processed['Disaster_Preparedness'].isin(['Savings', 'Insurance'])
        disaster_prep_community = df_processed['Disaster_Preparedness'] == 'Community Plan'
        df_processed.loc[disaster_prep_savings_insurance, 'Proactive_Financial_Health_Score'] += 2
        df_processed.loc[disaster_prep_community, 'Proactive_Financial_Health_Score'] += 1
        
        df_processed.loc[df_processed['Employment_Tenure_Months'] >= 12, 'Proactive_Financial_Health_Score'] += 1
        
        df_processed.loc[df_processed['Years_at_Current_Address'] >= 3, 'Proactive_Financial_Health_Score'] += 1
        
        df_processed.loc[df_processed['Housing_Status'] == 'Owned', 'Proactive_Financial_Health_Score'] += 1
        
        return df_processed

    # Makes a prediction based on input data with support for fairness-aware models
    def predict(self, input_data: Any, 
                apply_fairness: bool = True, 
                sensitive_feature_name: Optional[str] = None) -> Dict[str, Any]:
        try:
            if self.model is None:
                raise RuntimeError("PredictionService is not properly initialized. Model must be loaded.")
            
            if input_data is None:
                raise ValueError("Input data cannot be None.")
            
            try:
                if hasattr(input_data, 'model_dump'):
                    data_dict = input_data.model_dump()
                elif isinstance(input_data, dict):
                    data_dict = input_data
                else:
                    data_dict = dict(input_data)
                
                df = pd.DataFrame([data_dict])
                
                for col in df.columns:
                    if isinstance(df[col].iloc[0], Enum):
                        df[col] = df[col].map(lambda x: x.value)
                        
            except Exception as e:
                raise ValueError(f"Failed to convert input data to DataFrame: {e}")
            
            try:
                self._validate_input_data(df)
                df_processed = self._preprocess_input(df)
            except Exception as e:
                raise ValueError(f"Input validation/preprocessing failed: {e}")
            
            try:
                if self.is_fairness_aware:
                    if apply_fairness:
                        if sensitive_feature_name is None:
                            sensitive_feature_name = self.default_sensitive_feature
                        
                        if sensitive_feature_name is None and self.available_sensitive_features:
                            sensitive_feature_name = self.available_sensitive_features[0]
                            logger.info(f"No sensitive feature specified, using: {sensitive_feature_name}")
                        
                        if sensitive_feature_name and sensitive_feature_name in self.available_sensitive_features:
                            try:
                                binary_prediction = self.model.predict(
                                    df_processed, 
                                    apply_fairness=True, 
                                    sensitive_feature_name=sensitive_feature_name
                                )[0]
                                logger.info(f"Applied fairness correction for: {sensitive_feature_name}")
                            except Exception as fairness_error:
                                logger.warning(f"Fairness correction failed: {fairness_error}. Using base model.")
                                binary_prediction = self.model.predict(df_processed, apply_fairness=False)[0]
                        else:
                            binary_prediction = self.model.predict(df_processed, apply_fairness=False)[0]
                            logger.warning(f"Sensitive feature '{sensitive_feature_name}' not available. Using base model.")
                    else:
                        binary_prediction = self.model.predict(df_processed, apply_fairness=False)[0]
                    
                    prediction_proba = self.model.predict_proba(df_processed)
                else:
                    prediction_proba = self.model.predict_proba(df_processed)
                    binary_prediction = self.model.predict(df_processed)[0]
                
                if prediction_proba.shape[1] < 2:
                    raise RuntimeError("Model does not provide probability for positive class.")
                
                probability_of_default = float(prediction_proba[0, 1])
                
                if not self.is_fairness_aware or not apply_fairness:
                    binary_prediction = int(probability_of_default >= self.default_threshold)
                
            except Exception as e:
                raise RuntimeError(f"Failed to make prediction: {e}")
            
            if not (0 <= probability_of_default <= 1):
                logger.warning(f"Probability of default {probability_of_default} is outside [0,1] range. Clipping.")
                probability_of_default = max(0, min(1, probability_of_default))
                if not self.is_fairness_aware or not apply_fairness:
                    binary_prediction = int(probability_of_default >= self.default_threshold)
            
            result = {
                "probability_of_default": probability_of_default,
                "default_prediction": int(binary_prediction),
                "threshold_used": self.default_threshold,
                "is_fairness_aware": self.is_fairness_aware,
                "fairness_applied": self.is_fairness_aware and apply_fairness
            }
            
            if self.is_fairness_aware and apply_fairness and sensitive_feature_name:
                result["sensitive_feature_used"] = sensitive_feature_name
            
            return result
            
        except ValueError as e:
            logger.error(f"Input validation error: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"Runtime error during prediction: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    # Sets the threshold for binary classification
    def set_threshold(self, threshold: float) -> None:
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1.")
        self.default_threshold = threshold
        logger.info(f"Default threshold set to {threshold}")

    # Sets the default sensitive feature for fairness-aware predictions
    def set_default_sensitive_feature(self, feature_name: str) -> None:
        if self.is_fairness_aware and feature_name in self.available_sensitive_features:
            self.default_sensitive_feature = feature_name
            logger.info(f"Default sensitive feature set to: {feature_name}")
        else:
            raise ValueError(f"Feature '{feature_name}' is not available. Available features: {self.available_sensitive_features}")

    # Checks if the service is ready to make predictions
    def _is_service_ready(self) -> bool:
        return self.model is not None

    # Transforms probability of default to credit score
    @staticmethod
    def transform_pod_to_credit_score(pod: float, min_score: int = 300, max_score: int = 850) -> int:
        try:
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
            
            credit_score = min_score + (max_score - min_score) * (1 - pod)
            
            credit_score = int(round(credit_score))
            
            credit_score = max(min_score, min(max_score, credit_score))
            
            return credit_score
            
        except (TypeError, ValueError) as e:
            logger.error(f"Error transforming probability to credit score: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in credit score transformation: {e}")
            raise RuntimeError(f"Credit score transformation failed: {e}")

    # Returns the current status of the prediction service
    def get_service_status(self) -> Dict[str, Any]:
        required_features = [
            'Employment_Sector', 'Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
            'Salary_Frequency', 'Housing_Status', 'Years_at_Current_Address',
            'Number_of_Dependents', 'Comaker_Employment_Tenure_Months',
            'Comaker_Net_Salary_Per_Cutoff', 'Other_Income_Source',
            'Household_Head', 'Comaker_Relationship', 'Has_Community_Role',
            'Paluwagan_Participation', 'Disaster_Preparedness'
        ]
        
        categorical_features = [
            'Employment_Sector', 'Salary_Frequency', 'Housing_Status', 
            'Household_Head', 'Comaker_Relationship', 'Has_Community_Role',
            'Paluwagan_Participation', 'Other_Income_Source', 'Disaster_Preparedness'
        ]
        
        numerical_features = [
            'Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
            'Years_at_Current_Address', 'Number_of_Dependents',
            'Comaker_Employment_Tenure_Months', 'Comaker_Net_Salary_Per_Cutoff'
        ]
        
        status = {
            "is_ready": self._is_service_ready(),
            "model_loaded": self.model is not None,
            "is_fairness_aware": self.is_fairness_aware,
            "current_threshold": self.default_threshold,
            "required_features": required_features,
            "categorical_features": categorical_features,
            "numerical_features": numerical_features,
            "expected_feature_values": {
                "Employment_Sector": ["Public", "Private"],
                "Salary_Frequency": ["Monthly", "Bimonthly", "Biweekly", "Weekly"],
                "Housing_Status": ["Owned", "Rented"],
                "Household_Head": ["Yes", "No"],
                "Comaker_Relationship": ["Friend", "Sibling", "Parent", "Spouse"],
                "Has_Community_Role": ['None', 'Member', 'Leader', 'Multiple Leader'],
                "Paluwagan_Participation": ['Never', 'Rarely', 'Sometimes', 'Frequently'],
                "Other_Income_Source": ["None", "Freelance", "Business", "OFW Remittance"],
                "Disaster_Preparedness": ["None", "Savings", "Insurance", "Community Plan"]
            }
        }
        
        if self.is_fairness_aware:
            status["available_sensitive_features"] = self.available_sensitive_features
            status["default_sensitive_feature"] = self.default_sensitive_feature
        
        return status

    # Returns feature importance from the trained model if available
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        try:
            if not self._is_service_ready():
                return None
            
            model_to_check = self.model.base_model if self.is_fairness_aware else self.model
            
            if hasattr(model_to_check, 'named_steps') and 'classifier' in model_to_check.named_steps:
                classifier = model_to_check.named_steps['classifier']
                if hasattr(classifier, 'coef_'):
                    preprocessor = model_to_check.named_steps.get('preprocessor')
                    if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
                        try:
                            feature_names = preprocessor.get_feature_names_out()
                            coefficients = classifier.coef_[0]
                            
                            feature_importance = {}
                            for i, feature in enumerate(feature_names):
                                if i < len(coefficients):
                                    feature_importance[str(feature)] = float(coefficients[i])
                            
                            sorted_features = dict(sorted(feature_importance.items(), 
                                                        key=lambda x: abs(x[1]), reverse=True))
                            
                            return sorted_features
                        except Exception as e:
                            logger.warning(f"Could not extract feature names: {e}")
                            return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return None


# Initializes the prediction service with proper error handling
def initialize_prediction_service(threshold: float = 0.5, default_sensitive_feature: Optional[str] = None) -> Optional[PredictionService]:
    try:
        logger.info("Initializing PredictionService...")
        service = PredictionService(
            default_threshold=threshold,
            default_sensitive_feature=default_sensitive_feature
        )
        logger.info("PredictionService initialized successfully.")
        return service
    except Exception as e:
        logger.error(f"Failed to initialize PredictionService: {e}")
        return None


prediction_service = initialize_prediction_service()


# Performs complete credit assessment workflow from input to credit score
def get_credit_assessment(input_data, 
                         threshold: float = 0.5,
                         apply_fairness: bool = True,
                         sensitive_feature_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if prediction_service is None:
        logger.error("PredictionService is not available.")
        return None
    
    try:
        if prediction_service.default_threshold != threshold:
            prediction_service.set_threshold(threshold)
        
        prediction_result = prediction_service.predict(
            input_data, 
            apply_fairness=apply_fairness,
            sensitive_feature_name=sensitive_feature_name
        )
        pod = prediction_result["probability_of_default"]
        binary_prediction = prediction_result["default_prediction"]
        
        credit_score = PredictionService.transform_pod_to_credit_score(pod)
        
        if credit_score >= 750:
            risk_level = "Low Risk"
        elif credit_score >= 650:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
        
        recommendation = "Decline" if binary_prediction == 1 else "Approve"
        
        result = {
            "probability_of_default": pod,
            "default_prediction": binary_prediction,
            "credit_score": credit_score,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "threshold_used": threshold,
            "is_fairness_aware": prediction_result.get("is_fairness_aware", False),
            "fairness_applied": prediction_result.get("fairness_applied", False)
        }
        
        if "sensitive_feature_used" in prediction_result:
            result["sensitive_feature_used"] = prediction_result["sensitive_feature_used"]
        
        return result
        
    except Exception as e:
        logger.error(f"Credit assessment failed: {e}")
        return None


# Validates that input data contains all required fields with correct types
def validate_loan_application_data(data: dict) -> List[str]:
    errors = []
    
    required_fields = {
        'Employment_Sector': str,
        'Employment_Tenure_Months': (int, float),
        'Net_Salary_Per_Cutoff': (int, float),
        'Salary_Frequency': str,
        'Housing_Status': str,
        'Years_at_Current_Address': (int, float),
        'Number_of_Dependents': (int, float),
        'Comaker_Employment_Tenure_Months': (int, float),
        'Comaker_Net_Salary_Per_Cutoff': (int, float),
        'Other_Income_Source': str,
        'Household_Head': str,
        'Comaker_Relationship': str,
        'Has_Community_Role': str,
        'Paluwagan_Participation': str,
        'Disaster_Preparedness': str
    }
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif data[field] is None:
            errors.append(f"Field {field} cannot be None")
        elif not isinstance(data[field], required_fields[field]):
            errors.append(f"Field {field} must be of type {required_fields[field]}")
    
    if 'Employment_Tenure_Months' in data and data['Employment_Tenure_Months'] < 0:
        errors.append("Employment_Tenure_Months must be non-negative")
    
    if 'Net_Salary_Per_Cutoff' in data and data['Net_Salary_Per_Cutoff'] <= 0:
        errors.append("Net_Salary_Per_Cutoff must be positive")
    
    if 'Years_at_Current_Address' in data and data['Years_at_Current_Address'] < 0:
        errors.append("Years_at_Current_Address must be non-negative")
    
    if 'Number_of_Dependents' in data and data['Number_of_Dependents'] < 0:
        errors.append("Number_of_Dependents must be non-negative")
    
    valid_values = {
        'Employment_Sector': ['Public', 'Private'],
        'Salary_Frequency': ['Monthly', 'Bimonthly', 'Biweekly', 'Weekly'],
        'Housing_Status': ['Owned', 'Rented'],
        'Household_Head': ['Yes', 'No'],
        'Comaker_Relationship': ['Friend', 'Sibling', 'Parent', 'Spouse'],
        'Has_Community_Role': ['None', 'Member', 'Leader', 'Multiple Leader'],
        'Paluwagan_Participation': ['Never', 'Rarely', 'Sometimes', 'Frequently'],
        'Other_Income_Source': ['None', 'Freelance', 'Business', 'OFW Remittance'],
        'Disaster_Preparedness': ['None', 'Savings', 'Insurance', 'Community Plan']
    }
    
    for field, valid_list in valid_values.items():
        if field in data and data[field] not in valid_list:
            errors.append(f"{field} must be one of {valid_list}, got '{data[field]}'")
    
    return errors


if __name__ == "__main__":
    if prediction_service:
        status = prediction_service.get_service_status()
        print(f"Service Status: {status}")
        
        if status["is_fairness_aware"]:
            print(f"\nFairness-aware model loaded!")
            print(f"Available sensitive features: {status['available_sensitive_features']}")