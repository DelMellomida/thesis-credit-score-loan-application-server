import joblib
import pickle
import os
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
from app.schemas.loan_schema import LoanApplicationRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = './models'
MODEL_PATH = os.path.join(MODEL_DIR, 'new_client_model.pkl')

class PredictionService:
    def __init__(self, default_threshold: float = 0.5):
        self.model = None
        self.default_threshold = default_threshold
        self._load_model()

    def _load_model(self):
        """Load the trained model pipeline with comprehensive error handling."""
        try:
            logger.info("Loading model...")
            
            # Check if model directory exists
            if not os.path.exists(MODEL_DIR):
                raise FileNotFoundError(f"Model directory '{MODEL_DIR}' does not exist.")
            
            # Check if model file exists
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file '{MODEL_PATH}' does not exist.")
            
            # Load the model pipeline
            try:
                with open(MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Model loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {e}")
            
            # Validate model structure
            if not hasattr(self.model, 'predict') or not hasattr(self.model, 'predict_proba'):
                raise ValueError("Loaded model does not have required predict methods.")
            
            logger.info("Model validation successful.")
            
        except FileNotFoundError as e:
            logger.error(f"File not found error: {e}")
            raise RuntimeError(f"Model file not found: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def _validate_input_data(self, df: pd.DataFrame) -> None:
        """Validate input data contains all required features."""
        required_features = [
            # Financial features
            'Employment_Sector', 'Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
            'Salary_Frequency', 'Housing_Status', 'Years_at_Current_Address',
            'Number_of_Dependents', 'Comaker_Employment_Tenure_Months',
            'Comaker_Net_Salary_Per_Cutoff', 'Other_Income_Source',
            # Cultural features
            'Household_Head', 'Comaker_Relationship', 'Has_Community_Role',
            'Paluwagan_Participation', 'Disaster_Preparedness'
        ]
        
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Check for any null values in required features
        null_features = [f for f in required_features if df[f].isnull().any()]
        if null_features:
            raise ValueError(f"Null values found in features: {null_features}")

    def _preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data to match training data format."""
        df_processed = df.copy()
        
        # Categorical features that need standardization
        categorical_features = [
            'Employment_Sector', 'Salary_Frequency', 'Housing_Status', 
            'Household_Head', 'Comaker_Relationship', 'Has_Community_Role',
            'Paluwagan_Participation', 'Other_Income_Source', 'Disaster_Preparedness'
        ]
        
        # Standardize categorical values (same as in training)
        for col in categorical_features:
            if col in df_processed.columns and df_processed[col].dtype == 'object':
                # Strip whitespace
                df_processed[col] = df_processed[col].astype(str).str.strip()
                
                # Specific standardizations based on training preprocessing
                if col == 'Housing_Status':
                    df_processed[col] = df_processed[col].str.capitalize()
                elif col == 'Employment_Sector':
                    df_processed[col] = df_processed[col].str.capitalize()
                elif col in ['Household_Head', 'Has_Community_Role', 'Paluwagan_Participation']:
                    df_processed[col] = df_processed[col].str.capitalize()
        
        # Numerical features validation and preprocessing
        numerical_cols = [
            'Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
            'Years_at_Current_Address', 'Number_of_Dependents',
            'Comaker_Employment_Tenure_Months', 'Comaker_Net_Salary_Per_Cutoff'
        ]
        
        # Ensure non-negative values and reasonable ranges
        for col in numerical_cols:
            if col in df_processed.columns:
                # Ensure non-negative
                df_processed[col] = df_processed[col].clip(lower=0)
        
        # Apply capped penalty logic (same as training)
        if 'Number_of_Dependents' in df_processed.columns:
            df_processed['Number_of_Dependents'] = np.minimum(df_processed['Number_of_Dependents'], 5)
        
        return df_processed

    def predict(self, input_data: LoanApplicationRequest) -> Dict[str, Any]:
        """Make a prediction based on the input data with comprehensive error handling."""
        try:
            # Validate service is properly initialized
            if self.model is None:
                raise RuntimeError("PredictionService is not properly initialized. Model must be loaded.")
            
            # Validate input data
            if input_data is None:
                raise ValueError("Input data cannot be None.")
            
            # Convert to DataFrame
            try:
                df = pd.DataFrame([input_data.model_dump()])
            except Exception as e:
                raise ValueError(f"Failed to convert input data to DataFrame: {e}")
            
            # Validate and preprocess input data
            try:
                self._validate_input_data(df)
                df_processed = self._preprocess_input(df)
            except Exception as e:
                raise ValueError(f"Input validation/preprocessing failed: {e}")
            
            # Make prediction using the trained pipeline
            try:
                # Get probability predictions
                prediction_proba = self.model.predict_proba(df_processed)
                if prediction_proba.shape[1] < 2:
                    raise RuntimeError("Model does not provide probability for positive class.")
                
                # Probability of default (class 1)
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
        """Check if the service is ready to make predictions."""
        return self.model is not None

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
        
        return {
            "is_ready": self._is_service_ready(),
            "model_loaded": self.model is not None,
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

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the trained model if available."""
        try:
            if not self._is_service_ready():
                return None
            
            # Check if the model has a classifier with coefficients
            if hasattr(self.model, 'named_steps') and 'classifier' in self.model.named_steps:
                classifier = self.model.named_steps['classifier']
                if hasattr(classifier, 'coef_'):
                    # Get feature names from preprocessor
                    preprocessor = self.model.named_steps.get('preprocessor')
                    if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
                        try:
                            feature_names = preprocessor.get_feature_names_out()
                            coefficients = classifier.coef_[0]
                            
                            # Create feature importance dictionary
                            feature_importance = {}
                            for i, feature in enumerate(feature_names):
                                if i < len(coefficients):
                                    feature_importance[str(feature)] = float(coefficients[i])
                            
                            # Sort by absolute importance
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

# Complete workflow function for your model
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

# Utility function to validate input data format
def validate_loan_application_data(data: dict) -> List[str]:
    """Validate that the input data contains all required fields with correct types."""
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
    
    # Check for missing fields
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif data[field] is None:
            errors.append(f"Field {field} cannot be None")
        elif not isinstance(data[field], required_fields[field]):
            errors.append(f"Field {field} must be of type {required_fields[field]}")
    
    # Validate numerical ranges
    if 'Employment_Tenure_Months' in data and data['Employment_Tenure_Months'] < 0:
        errors.append("Employment_Tenure_Months must be non-negative")
    
    if 'Net_Salary_Per_Cutoff' in data and data['Net_Salary_Per_Cutoff'] <= 0:
        errors.append("Net_Salary_Per_Cutoff must be positive")
    
    if 'Years_at_Current_Address' in data and data['Years_at_Current_Address'] < 0:
        errors.append("Years_at_Current_Address must be non-negative")
    
    if 'Number_of_Dependents' in data and data['Number_of_Dependents'] < 0:
        errors.append("Number_of_Dependents must be non-negative")
    
    # Validate categorical values
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