import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class ClientType(Enum):
    """Client type enumeration for type-safe operations."""
    NEW = "new"
    RENEWING = "renewing"


class ScalingMethod(Enum):
    """Supported scaling methods for feature normalization."""
    LOG_MINMAX = "log_minmax"
    MINMAX = "minmax"
    ZSCORE = "zscore"
    ROBUST = "robust"


@dataclass
class FeatureConfig:
    """Configuration for individual feature with strict caps and constraints."""
    name: str
    max_contribution_pct: float         # Maximum % this feature can contribute to its component
    weight_in_component: float          # Weight within its component (0.0 to 1.0)
    scaling_method: ScalingMethod = ScalingMethod.LOG_MINMAX
    min_value: float = 0.0
    max_value: float = 1.0
    handle_outliers: bool = True
    outlier_percentiles: Tuple[float, float] = (0.05, 0.95)
    categorical_mapping: Optional[Dict[str, float]] = None
    missing_value_strategy: str = "median"  # median, mean, mode, zero
    
    def __post_init__(self):
        """Validate configuration on initialization."""
        if not 0 <= self.weight_in_component <= 1.0:
            raise ValueError(f"Feature {self.name}: weight_in_component must be between 0 and 1")
        if not 0 <= self.max_contribution_pct <= 1.0:
            raise ValueError(f"Feature {self.name}: max_contribution_pct must be between 0 and 1")


@dataclass  
class ComponentConfig:
    """Configuration for scoring components with mathematical feature isolation."""
    name: str
    max_contribution_pct: float         # Maximum % this component can contribute to final score
    features: Dict[str, FeatureConfig] = field(default_factory=dict)
    normalization_method: str = "weighted_capped"
    component_description: str = ""
    
    def add_feature(self, feature_config: FeatureConfig) -> None:
        """Add feature with validation and automatic weight normalization."""
        self.features[feature_config.name] = feature_config
        
        # Validate and normalize weights to sum to 1.0
        total_weight = sum(f.weight_in_component for f in self.features.values())
        if total_weight > 1.0:
            # Auto-normalize weights
            for feature_name, feature in self.features.items():
                feature.weight_in_component /= total_weight
            logger.info(f"Component {self.name}: Auto-normalized feature weights to sum to 1.0")
    
    def get_max_individual_contribution(self) -> float:
        """Get the maximum any single feature can contribute to this component."""
        if not self.features:
            return 0.0
        return max(f.max_contribution_pct for f in self.features.values())


@dataclass
class ClientTypeConfig:
    """Configuration for specific client type scoring with complete isolation."""
    client_type: ClientType
    components: Dict[str, ComponentConfig] = field(default_factory=dict)
    description: str = ""
    
    def add_component(self, component_config: ComponentConfig) -> None:
        """Add component with validation."""
        self.components[component_config.name] = component_config
        
        # Validate total component contributions
        total_contribution = sum(c.max_contribution_pct for c in self.components.values())
        if total_contribution > 1.0:
            logger.warning(f"Client type {self.client_type.value}: total contributions exceed 100% ({total_contribution:.1%})")
    
    def get_total_max_contribution(self) -> float:
        """Get total maximum contribution across all components."""
        return sum(c.max_contribution_pct for c in self.components.values())


class EnhancedCreditScoringConfig:
    """
    Master configuration class with complete feature isolation and data leakage prevention.
    
    CRITICAL DESIGN FEATURES:
    1. Client-type specific configurations
    2. Individual feature contribution caps
    3. Component-level contribution caps  
    4. Data-driven scaling parameters
    5. Severe constraints on leaky features
    """
    
    def __init__(self):
        self.client_configs: Dict[ClientType, ClientTypeConfig] = {}
        self.data_driven_ranges = self._get_data_driven_ranges()
        self.leakage_mitigation_features = self._get_leakage_features()
        
        # Initialize client-specific configurations
        self._setup_new_client_config()
        self._setup_renewing_client_config()
        
        # Validation
        self._validate_configurations()
        
        logger.info("Enhanced Credit Scoring Configuration initialized successfully")
    
    def _get_data_driven_ranges(self) -> Dict[str, Dict[str, float]]:
        """Data-driven scaling ranges from actual dataset analysis."""
        return {
            'net_salary': {
                'min_log': np.log1p(9000),      # ₱9,000 (5th percentile)
                'max_log': np.log1p(75000),     # ₱75,000 (95th percentile)
                'median': 25000
            },
            'comaker_salary': {
                'min_log': np.log1p(5000),
                'max_log': np.log1p(80000),
                'median': 20000
            },
            'employment_tenure': {
                'min_log': np.log1p(2),         # 2 months (5th percentile)
                'max_log': np.log1p(240),       # 240 months (95th percentile)
                'median': 36
            },
            'comaker_tenure': {
                'min_log': np.log1p(2),         # 2 months
                'max_log': np.log1p(180),       # 180 months
                'median': 24
            },
            'address_stability': {
                'min_log': np.log1p(0.1),
                'max_log': np.log1p(30),
                'median': 3
            },
            'household_income': {
                'min_log': np.log1p(15000),
                'max_log': np.log1p(150000),
                'median': 45000
            }
        }
    
    def _get_leakage_features(self) -> Dict[str, Dict[str, float]]:
        """
        Severely constrained mappings for data leakage prevention.
        
        CRITICAL: These features showed perfect/near-perfect prediction:
        - Community Role "Yes" = 0.0% default (perfect predictor)
        - Paluwagan "Yes" = 9.1% vs "No" = 75.1% (66% difference)
        """
        return {
            'has_community_role': {
                'Yes': 0.1,    # SEVERELY LIMITED from perfect predictor
                'No': 0.0       # Baseline
            },
            'paluwagan_participation': {
                'Yes': 0.1,    # SEVERELY LIMITED from 66% advantage
                'No': 0.0       # Baseline
            },
            'housing_status': {
                'Owned': 0.08,
                'Rented': 0.0
            },
            'household_head': {
                'Yes': 0.10,
                'No': 0.05
            },
            'employment_sector': {
                'Public': 0.12,
                'Private': 0.08
            },
            'disaster_preparedness': {
                'Community Plan': 0.1,
                'Insurance': 0.07,
                'Savings': 0.05,
                'None': 0.0
            },
            'other_income_source': {
                'OFW Remittance': 0.1,
                'Business': 0.07,
                'Freelance': 0.05,
                'None': 0.0
            },
            'comaker_relationship': {
                'Spouse': 0.18,
                'Parent': 0.12,
                'Sibling': 0.10,
                'Friend': 0.05
            },
            'salary_frequency': {
                'Weekly': 0.20,
                'Biweekly': 0.15,
                'Monthly': 0.10,
                'Bimonthly': 0.05
            }
        }
    
    def _setup_new_client_config(self) -> None:
        """Setup configuration for new clients (80% Financial, 20% Cultural)."""
        new_client_config = ClientTypeConfig(
            client_type=ClientType.NEW,
            description="New clients with no credit history - focus on financial capacity"
        )
        
        # Financial Stability Component (80% for new clients)
        financial_component = ComponentConfig(
            name="financial_stability",
            max_contribution_pct=0.80,
            component_description="Objective financial capacity and stability measures"
        )
        
        # Add financial features with caps and re-balanced weights
        financial_features = [
            FeatureConfig(
                name="income_adequacy",
                max_contribution_pct=0.25,
                weight_in_component=0.30,
                scaling_method=ScalingMethod.LOG_MINMAX
            ),
            FeatureConfig(
                name="household_income_capacity", 
                max_contribution_pct=0.25,
                weight_in_component=0.25,
                scaling_method=ScalingMethod.LOG_MINMAX
            ),
            FeatureConfig(
                name="employment_stability",
                max_contribution_pct=0.20,
                weight_in_component=0.20,
                scaling_method=ScalingMethod.LOG_MINMAX
            ),
            FeatureConfig(
                name="comaker_stability",
                max_contribution_pct=0.15,
                weight_in_component=0.15,
                scaling_method=ScalingMethod.LOG_MINMAX
            ),
            FeatureConfig(
                name="address_stability",
                max_contribution_pct=0.10,
                weight_in_component=0.07,
                scaling_method=ScalingMethod.LOG_MINMAX
            ),
            FeatureConfig(
                name="sector_stability",
                max_contribution_pct=0.05,
                weight_in_component=0.03,
                categorical_mapping=self.leakage_mitigation_features['employment_sector']
            )
        ]
        
        for feature in financial_features:
            financial_component.add_feature(feature)
        
        new_client_config.add_component(financial_component)
        
        # Cultural Context Component (20% for new clients)
        cultural_component = ComponentConfig(
            name="cultural_context",
            max_contribution_pct=0.10,
            component_description="Cultural and behavioral indicators with severe leakage constraints"
        )
        
        # Add cultural features with SEVERE caps and re-balanced weights
        cultural_features = [
            FeatureConfig(
                name="financial_discipline",     # Paluwagan 
                max_contribution_pct=0.03,
                weight_in_component=0.20,
                categorical_mapping=self.leakage_mitigation_features['paluwagan_participation']
            ),
            FeatureConfig(
                name="community_integration",   # Community role
                max_contribution_pct=0.02,
                weight_in_component=0.15,
                categorical_mapping=self.leakage_mitigation_features['has_community_role']
            ),
            FeatureConfig(
                name="family_stability",        # Housing + household head
                max_contribution_pct=0.08,
                weight_in_component=0.20,
            ),
            FeatureConfig(
                name="income_diversification",  # Other income
                max_contribution_pct=0.08,
                weight_in_component=0.15,
                categorical_mapping=self.leakage_mitigation_features['other_income_source']
            ),
            FeatureConfig(
                name="relationship_strength",   # Comaker relationship
                max_contribution_pct=0.06,
                weight_in_component=0.15,
                categorical_mapping=self.leakage_mitigation_features['comaker_relationship']
            ),
            FeatureConfig(
                name="resilience_planning",     # Disaster preparedness
                max_contribution_pct=0.07,
                weight_in_component=0.15,
                categorical_mapping=self.leakage_mitigation_features['disaster_preparedness']
            )
        ]
        
        for feature in cultural_features:
            cultural_component.add_feature(feature)
        
        new_client_config.add_component(cultural_component)
        self.client_configs[ClientType.NEW] = new_client_config
    
    def _setup_renewing_client_config(self) -> None:
        """Setup configuration for renewing clients (60% Credit, 37% Financial, 3% Cultural)."""
        renewing_client_config = ClientTypeConfig(
            client_type=ClientType.RENEWING,
            description="Renewing clients with credit history - focus on payment behavior"
        )
        
        # Credit Behavior Component (60% for renewing clients)
        credit_component = ComponentConfig(
            name="credit_behavior",
            max_contribution_pct=0.60,
            component_description="Objective payment history and credit behavior"
        )
        
        # Add credit behavior features
        credit_features = [
            FeatureConfig(
                name="payment_history",
                max_contribution_pct=0.35,
                weight_in_component=0.50,
                scaling_method=ScalingMethod.MINMAX
            ),
            FeatureConfig(
                name="grace_period_usage",
                max_contribution_pct=0.20,
                weight_in_component=0.30,
                scaling_method=ScalingMethod.MINMAX
            ),
            FeatureConfig(
                name="special_considerations",
                max_contribution_pct=0.10,
                weight_in_component=0.10,
            ),
            FeatureConfig(
                name="client_loyalty",
                max_contribution_pct=0.10,
                weight_in_component=0.10,
            )
        ]
        
        for feature in credit_features:
            credit_component.add_feature(feature)
        
        renewing_client_config.add_component(credit_component)
        
        # Financial Stability Component (37% for renewing clients)
        financial_component = ComponentConfig(
            name="financial_stability", 
            max_contribution_pct=0.37,
            component_description="Current financial capacity and stability"
        )
        
        # Add financial features (same as new clients but different weights)
        financial_features = [
            FeatureConfig(
                name="income_adequacy",
                max_contribution_pct=0.15,
                weight_in_component=0.30,
                scaling_method=ScalingMethod.LOG_MINMAX
            ),
            FeatureConfig(
                name="household_income_capacity",
                max_contribution_pct=0.12,
                weight_in_component=0.25,
                scaling_method=ScalingMethod.LOG_MINMAX
            ),
            FeatureConfig(
                name="employment_stability", 
                max_contribution_pct=0.10,
                weight_in_component=0.20,
                scaling_method=ScalingMethod.LOG_MINMAX
            ),
            FeatureConfig(
                name="comaker_stability",
                max_contribution_pct=0.08,
                weight_in_component=0.15,
                scaling_method=ScalingMethod.LOG_MINMAX
            ),
            FeatureConfig(
                name="address_stability",
                max_contribution_pct=0.05,
                weight_in_component=0.07,
                scaling_method=ScalingMethod.LOG_MINMAX
            ),
            FeatureConfig(
                name="sector_stability",
                max_contribution_pct=0.03,
                weight_in_component=0.03,
                categorical_mapping=self.leakage_mitigation_features['employment_sector']
            )
        ]
        
        for feature in financial_features:
            financial_component.add_feature(feature)
        
        renewing_client_config.add_component(financial_component)
        
        # Cultural Context Component (3% for renewing clients - SEVERELY LIMITED)
        cultural_component = ComponentConfig(
            name="cultural_context",
            max_contribution_pct=0.03,
            component_description="SEVERELY LIMITED cultural factors to prevent data leakage"
        )
        
        # Add cultural features with EXTREME caps for renewing clients
        cultural_features = [
            FeatureConfig(
                name="financial_discipline",
                max_contribution_pct=0.008,
                weight_in_component=0.20,
                categorical_mapping=self.leakage_mitigation_features['paluwagan_participation']
            ),
            FeatureConfig(
                name="community_integration",
                max_contribution_pct=0.005,
                weight_in_component=0.15,
                categorical_mapping=self.leakage_mitigation_features['has_community_role']
            ),
            FeatureConfig(
                name="family_stability",
                max_contribution_pct=0.012,
                weight_in_component=0.20,
            ),
            FeatureConfig(
                name="income_diversification",
                max_contribution_pct=0.008,
                weight_in_component=0.15,
                categorical_mapping=self.leakage_mitigation_features['other_income_source']
            ),
            FeatureConfig(
                name="relationship_strength",
                max_contribution_pct=0.007,
                weight_in_component=0.15,
                categorical_mapping=self.leakage_mitigation_features['comaker_relationship']
            ),
            FeatureConfig(
                name="resilience_planning",
                max_contribution_pct=0.006,
                weight_in_component=0.15,
                categorical_mapping=self.leakage_mitigation_features['disaster_preparedness']
            )
        ]
        
        for feature in cultural_features:
            cultural_component.add_feature(feature)
        
        renewing_client_config.add_component(cultural_component)
        self.client_configs[ClientType.RENEWING] = renewing_client_config
    
    def _validate_configurations(self) -> None:
        """Validate all configurations for consistency and caps."""
        for client_type, config in self.client_configs.items():
            total_contribution = config.get_total_max_contribution()
            if total_contribution > 1.0:
                raise ValueError(f"Client type {client_type.value}: total max contributions exceed 100%")
            
            logger.info(f"✓ {client_type.value} client config validated (max total: {total_contribution:.1%})")
    
    def get_client_config(self, client_type: ClientType) -> ClientTypeConfig:
        """Get configuration for specific client type."""
        return self.client_configs[client_type]
    
    def get_feature_caps_summary(self) -> Dict[str, Any]:
        """Get summary of all feature caps for transparency."""
        summary = {}
        
        for client_type, config in self.client_configs.items():
            client_summary = {}
            for comp_name, component in config.components.items():
                comp_summary = {
                    'component_max_pct': component.max_contribution_pct,
                    'features': {}
                }
                for feat_name, feature in component.features.items():
                    comp_summary['features'][feat_name] = {
                        'max_contribution_pct': feature.max_contribution_pct,
                        'weight_in_component': feature.weight_in_component,
                        'max_total_impact_pct': feature.max_contribution_pct * component.max_contribution_pct
                    }
                client_summary[comp_name] = comp_summary
            summary[client_type.value] = client_summary
        
        return summary
    
    def validate_loan_application_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate dataframe against LoanApplicationRequest schema.
        
        Ensures compatibility with the API input schema.
        """
        required_columns = [
            'Employment_Sector', 'Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
            'Salary_Frequency', 'Housing_Status', 'Years_at_Current_Address',
            'Household_Head', 'Number_of_Dependents', 'Comaker_Relationship',
            'Comaker_Employment_Tenure_Months', 'Comaker_Net_Salary_Per_Cutoff',
            'Has_Community_Role', 'Paluwagan_Participation', 'Other_Income_Source',
            'Disaster_Preparedness', 'Is_Renewing_Client', 'Grace_Period_Usage_Rate',
            'Late_Payment_Count', 'Had_Special_Consideration'
        ]
        
        schema_rules = {
            'Employment_Sector': {'type': 'categorical', 'allowed': ['Public', 'Private']},
            'Employment_Tenure_Months': {'type': 'numeric', 'min': 1, 'dtype': 'int'},
            'Net_Salary_Per_Cutoff': {'type': 'numeric', 'min': 0.01, 'dtype': 'float'},
            'Salary_Frequency': {'type': 'categorical', 'allowed': ['Monthly', 'Bimonthly', 'Biweekly', 'Weekly']},
            'Housing_Status': {'type': 'categorical', 'allowed': ['Owned', 'Rented']},
            'Years_at_Current_Address': {'type': 'numeric', 'min': 0, 'dtype': 'float'},
            'Household_Head': {'type': 'categorical', 'allowed': ['Yes', 'No']},
            'Number_of_Dependents': {'type': 'numeric', 'min': 0, 'dtype': 'int'},
            'Comaker_Relationship': {'type': 'categorical', 'allowed': ['Friend', 'Parent', 'Sibling', 'Spouse']},
            'Comaker_Employment_Tenure_Months': {'type': 'numeric', 'min': 1, 'dtype': 'int'},
            'Comaker_Net_Salary_Per_Cutoff': {'type': 'numeric', 'min': 0.01, 'dtype': 'float'},
            'Has_Community_Role': {'type': 'categorical', 'allowed': ['Yes', 'No']},
            'Paluwagan_Participation': {'type': 'categorical', 'allowed': ['Yes', 'No']},
            'Other_Income_Source': {'type': 'categorical', 'allowed': ['None', 'Freelance', 'Business', 'OFW Remittance']},
            'Disaster_Preparedness': {'type': 'categorical', 'allowed': ['None', 'Savings', 'Insurance', 'Community Plan']},
            'Is_Renewing_Client': {'type': 'binary', 'allowed': [0, 1]},
            'Grace_Period_Usage_Rate': {'type': 'numeric', 'min': 0.0, 'max': 1.0, 'dtype': 'float'},
            'Late_Payment_Count': {'type': 'numeric', 'min': 0, 'dtype': 'int'},
            'Had_Special_Consideration': {'type': 'binary', 'allowed': [0, 1]}
        }
        
        issues_found = []
        df_fixed = df.copy()
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues_found.extend([f"Missing required column: {col}" for col in missing_columns])
            return df_fixed, issues_found
        
        # Validate each column
        for col, rules in schema_rules.items():
            if col not in df.columns:
                continue
                
            if rules['type'] == 'categorical':
                invalid_values = df[col].dropna().unique()
                invalid_values = [v for v in invalid_values if v not in rules['allowed']]
                if invalid_values:
                    issues_found.append(f"{col} has invalid values: {invalid_values}")
                    # Fix by mapping to first allowed value
                    df_fixed[col] = df_fixed[col].fillna(rules['allowed'][0])
                    mask = ~df_fixed[col].isin(rules['allowed'])
                    df_fixed.loc[mask, col] = rules['allowed'][0]
                    
            elif rules['type'] in ['numeric', 'binary']:
                # Check minimum values
                if 'min' in rules:
                    invalid_count = (df[col] < rules['min']).sum()
                    if invalid_count > 0:
                        issues_found.append(f"{col} has {invalid_count} values below minimum {rules['min']}")
                        df_fixed[col] = df_fixed[col].clip(lower=rules['min'])
                
                # Check maximum values        
                if 'max' in rules:
                    invalid_count = (df[col] > rules['max']).sum()
                    if invalid_count > 0:
                        issues_found.append(f"{col} has {invalid_count} values above maximum {rules['max']}")
                        df_fixed[col] = df_fixed[col].clip(upper=rules['max'])
        
        return df_fixed, issues_found


class FeatureProcessor:
    """Advanced feature processor with mathematical isolation and robust scaling."""
    
    def __init__(self, config: EnhancedCreditScoringConfig):
        self.config = config
        self.scaling_cache = {}
        
    def process_numerical_feature(self, 
                                values: pd.Series, 
                                feature_config: FeatureConfig,
                                range_key: str) -> pd.Series:
        """Process numerical features with robust scaling and outlier handling."""
        # Handle missing values
        if feature_config.missing_value_strategy == "median":
            fill_value = values.median()
        elif feature_config.missing_value_strategy == "mean":
            fill_value = values.mean()
        elif feature_config.missing_value_strategy == "zero":
            fill_value = 0
        else:
            fill_value = values.median()
        
        processed_values = values.fillna(fill_value)
        
        # Handle outliers if specified
        if feature_config.handle_outliers:
            lower_pct, upper_pct = feature_config.outlier_percentiles
            lower_bound = processed_values.quantile(lower_pct)
            upper_bound = processed_values.quantile(upper_pct)
            processed_values = processed_values.clip(lower_bound, upper_bound)
        
        # Apply scaling method
        if feature_config.scaling_method == ScalingMethod.LOG_MINMAX:
            return self._log_minmax_scale(processed_values, range_key)
        elif feature_config.scaling_method == ScalingMethod.MINMAX:
            return self._minmax_scale(processed_values)
        elif feature_config.scaling_method == ScalingMethod.ZSCORE:
            return self._zscore_scale(processed_values)
        else:
            return self._robust_scale(processed_values)
    
    def _log_minmax_scale(self, values: pd.Series, range_key: str) -> pd.Series:
        """Log transform followed by min-max scaling using data-driven ranges."""
        if range_key not in self.config.data_driven_ranges:
            # Fallback to percentile-based scaling
            log_values = np.log1p(values)
            min_val = log_values.quantile(0.05)
            max_val = log_values.quantile(0.95)
        else:
            log_values = np.log1p(values)
            range_params = self.config.data_driven_ranges[range_key]
            min_val = range_params['min_log']
            max_val = range_params['max_log']
        
        if max_val == min_val:
            return pd.Series([0.5] * len(values), index=values.index)
        
        scaled = (log_values - min_val) / (max_val - min_val)
        return pd.Series(np.clip(scaled, 0, 1), index=values.index)
    
    def _minmax_scale(self, values: pd.Series) -> pd.Series:
        """Standard min-max scaling."""
        min_val = values.min()
        max_val = values.max()
        
        if max_val == min_val:
            return pd.Series([0.5] * len(values), index=values.index)
        
        scaled = (values - min_val) / (max_val - min_val)
        return pd.Series(np.clip(scaled, 0, 1), index=values.index)
    
    def _zscore_scale(self, values: pd.Series) -> pd.Series:
        """Z-score normalization with sigmoid transformation."""
        mean_val = values.mean()
        std_val = values.std()
        
        if std_val == 0:
            return pd.Series([0.5] * len(values), index=values.index)
        
        z_scores = (values - mean_val) / std_val
        sigmoid_values = 1 / (1 + np.exp(-z_scores))
        return pd.Series(sigmoid_values, index=values.index)
    
    def _robust_scale(self, values: pd.Series) -> pd.Series:
        """Robust scaling using median and IQR."""
        median_val = values.median()
        q75 = values.quantile(0.75)
        q25 = values.quantile(0.25)
        iqr = q75 - q25
        
        if iqr == 0:
            return pd.Series([0.5] * len(values), index=values.index)
        
        scaled = (values - median_val) / iqr
        sigmoid_values = 1 / (1 + np.exp(-scaled))
        return pd.Series(sigmoid_values, index=values.index)
    
    def process_categorical_feature(self, 
                                  values: pd.Series,
                                  feature_config: FeatureConfig) -> pd.Series:
        """Process categorical features with predefined mappings and missing value handling."""
        if feature_config.categorical_mapping is None:
            raise ValueError(f"Categorical feature {feature_config.name} requires categorical_mapping")
        
        # Handle missing values by mapping to lowest score
        min_score = min(feature_config.categorical_mapping.values())
        processed_values = values.fillna("__MISSING__")
        
        # Map values
        mapped_values = processed_values.map(feature_config.categorical_mapping)
        
        # Handle unmapped values (including missing)
        mapped_values = mapped_values.fillna(min_score)
        
        return mapped_values


class ComponentScorer(ABC):
    """Abstract base class for component scoring with mathematical isolation."""
    
    def __init__(self, component_config: ComponentConfig, processor: FeatureProcessor):
        self.config = component_config
        self.processor = processor
    
    @abstractmethod
    def calculate_component_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate component score with feature isolation."""
        pass
    
    def _apply_feature_caps(self, feature_scores: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Apply individual feature caps to prevent any feature from dominating."""
        capped_scores = {}
        
        for feature_name, scores in feature_scores.items():
            feature_config = self.config.features[feature_name]
            max_cap = feature_config.max_contribution_pct
            
            # Cap the feature contribution
            capped_scores[feature_name] = pd.Series(
                np.clip(scores, 0, max_cap), 
                index=scores.index
            )
        
        return capped_scores
    
    def _normalize_component_score(self, combined_score: pd.Series) -> pd.Series:
        """Normalize component score and apply component-level cap."""
        # Normalize to 0-1 range first
        if combined_score.max() > 0:
            normalized = combined_score / combined_score.max()
        else:
            normalized = combined_score
        
        # Apply component-level cap
        component_cap = self.config.max_contribution_pct
        final_score = pd.Series(
            np.clip(normalized * component_cap, 0, component_cap),
            index=combined_score.index
        )
        
        return final_score
    
class FinancialStabilityScorer(ComponentScorer):
    """
    Financial stability scorer for both client types.
    
    Uses objective financial data with robust scaling and feature isolation.
    """
    
    def calculate_component_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate financial stability score with feature isolation."""
        feature_scores = {}
        
        # Income adequacy (primary income per household member)
        if 'Net_Salary_Per_Cutoff' in df.columns:
            primary_income = df['Net_Salary_Per_Cutoff'].fillna(0)
            comaker_income = df['Comaker_Net_Salary_Per_Cutoff'].fillna(0)
            dependents = df['Number_of_Dependents'].fillna(0)
            has_comaker = (comaker_income > 0).astype(int)
            
            household_size = 1 + dependents + has_comaker
            income_per_member = (primary_income + comaker_income) / household_size
            
            feature_scores['income_adequacy'] = self.processor.process_numerical_feature(
                income_per_member,
                self.config.features['income_adequacy'],
                'net_salary'
            )
        
        # Household income capacity (total household income)
        if 'Net_Salary_Per_Cutoff' in df.columns and 'Comaker_Net_Salary_Per_Cutoff' in df.columns:
            total_household_income = df['Net_Salary_Per_Cutoff'].fillna(0) + \
                                   df['Comaker_Net_Salary_Per_Cutoff'].fillna(0)
            
            feature_scores['household_income_capacity'] = self.processor.process_numerical_feature(
                total_household_income,
                self.config.features['household_income_capacity'],
                'household_income'
            )
        
        # Employment stability
        if 'Employment_Tenure_Months' in df.columns:
            employment_tenure = df['Employment_Tenure_Months'].fillna(0)
            
            feature_scores['employment_stability'] = self.processor.process_numerical_feature(
                employment_tenure,
                self.config.features['employment_stability'],
                'employment_tenure'
            )
        
        # Comaker stability
        if 'Comaker_Employment_Tenure_Months' in df.columns:
            comaker_tenure = df['Comaker_Employment_Tenure_Months'].fillna(0)
            
            feature_scores['comaker_stability'] = self.processor.process_numerical_feature(
                comaker_tenure,
                self.config.features['comaker_stability'],
                'comaker_tenure'
            )

        # Address stability
        if 'Years_at_Current_Address' in df.columns:
            address_years = df['Years_at_Current_Address'].fillna(0)
            
            feature_scores['address_stability'] = self.processor.process_numerical_feature(
                address_years,
                self.config.features['address_stability'],
                'address_stability'
            )
        
        # Sector stability (if included)
        if 'sector_stability' in self.config.features and 'Employment_Sector' in df.columns:
            feature_scores['sector_stability'] = self.processor.process_categorical_feature(
                df['Employment_Sector'],
                self.config.features['sector_stability']
            )
        
        # Apply feature caps
        capped_scores = self._apply_feature_caps(feature_scores)
        
        # Combine features with weights
        combined_score = pd.Series([0.0] * len(df), index=df.index)
        for feature_name, scores in capped_scores.items():
            if feature_name in self.config.features:
                weight = self.config.features[feature_name].weight_in_component
                combined_score += scores * weight
        
        # Apply component-level normalization and cap
        return self._normalize_component_score(combined_score)


class CulturalContextScorer(ComponentScorer):
    """
    Cultural context scorer with severe data leakage prevention.
    
    Uses heavily constrained cultural features to prevent bias and leakage.
    """
    
    def calculate_component_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate cultural context score with severe leakage constraints."""
        feature_scores = {}
        
        # Financial discipline (Paluwagan participation - SEVERELY CONSTRAINED)
        if 'Paluwagan_Participation' in df.columns:
            feature_scores['financial_discipline'] = self.processor.process_categorical_feature(
                df['Paluwagan_Participation'],
                self.config.features['financial_discipline']
            )
        
        # Community integration (Community role - SEVERELY CONSTRAINED)  
        if 'Has_Community_Role' in df.columns:
            feature_scores['community_integration'] = self.processor.process_categorical_feature(
                df['Has_Community_Role'],
                self.config.features['community_integration']
            )
        
        # Family stability (Housing status + Household head)
        family_stability_score = pd.Series([0.0] * len(df), index=df.index)
        
        if 'Housing_Status' in df.columns:
            housing_scores = self.processor.process_categorical_feature(
                df['Housing_Status'],
                FeatureConfig(
                    name="housing_temp",
                    max_contribution_pct=0.5,
                    weight_in_component=0.5,
                    categorical_mapping=self.processor.config.leakage_mitigation_features['housing_status']
                )
            )
            family_stability_score += housing_scores * 0.5
        
        if 'Household_Head' in df.columns:
            head_scores = self.processor.process_categorical_feature(
                df['Household_Head'],
                FeatureConfig(
                    name="head_temp",
                    max_contribution_pct=0.5,
                    weight_in_component=0.5,
                    categorical_mapping=self.processor.config.leakage_mitigation_features['household_head']
                )
            )
            family_stability_score += head_scores * 0.5
        
        feature_scores['family_stability'] = family_stability_score
        
        # Income diversification (Other income source)
        if 'Other_Income_Source' in df.columns:
            feature_scores['income_diversification'] = self.processor.process_categorical_feature(
                df['Other_Income_Source'],
                self.config.features['income_diversification']
            )
        
        # Relationship strength (Comaker relationship)
        if 'Comaker_Relationship' in df.columns:
            feature_scores['relationship_strength'] = self.processor.process_categorical_feature(
                df['Comaker_Relationship'],
                self.config.features['relationship_strength']
            )
            
        # Resilience Planning (Disaster Preparedness)
        if 'Disaster_Preparedness' in df.columns:
            feature_scores['resilience_planning'] = self.processor.process_categorical_feature(
                df['Disaster_Preparedness'],
                self.config.features['resilience_planning']
            )

        # Apply feature caps
        capped_scores = self._apply_feature_caps(feature_scores)
        
        # Combine features with weights
        combined_score = pd.Series([0.0] * len(df), index=df.index)
        for feature_name, scores in capped_scores.items():
            if feature_name in self.config.features:
                weight = self.config.features[feature_name].weight_in_component
                combined_score += scores * weight
        
        # Additional penalty for dependents (capped)
        if 'Number_of_Dependents' in df.columns:
            dependents = df['Number_of_Dependents'].fillna(0)
            dependents_penalty = np.clip(dependents * 0.01, 0, 0.05)  # Max 5% penalty
            combined_score -= dependents_penalty
        
        # Apply component-level normalization and cap
        return self._normalize_component_score(combined_score)

class CreditBehaviorScorer(ComponentScorer):
    """Credit behavior scorer for renewing clients only."""
    
    def calculate_component_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate credit behavior score for renewing clients."""
        feature_scores = {}
        
        # Payment history score (inverted - lower late payments = higher score)
        if 'Late_Payment_Count' in df.columns:
            late_payments = df['Late_Payment_Count'].fillna(0)
            max_late = max(late_payments.max(), 1)
            payment_score = 1.0 - (late_payments / max_late)
            
            feature_scores['payment_history'] = self.processor.process_numerical_feature(
                payment_score, 
                self.config.features['payment_history'],
                'payment_history'
            )
        
        # Grace period usage score (inverted - lower usage = higher score)
        if 'Grace_Period_Usage_Rate' in df.columns:
            grace_usage = df['Grace_Period_Usage_Rate'].fillna(0)
            grace_score = 1.0 - grace_usage
            
            feature_scores['grace_period_usage'] = self.processor.process_numerical_feature(
                grace_score,
                self.config.features['grace_period_usage'],
                'grace_usage'
            )
        
        # Special considerations penalty (inverted)
        if 'Had_Special_Consideration' in df.columns:
            special_considerations = df['Had_Special_Consideration'].fillna(0)
            special_score = 1.0 - special_considerations
            
            feature_scores['special_considerations'] = special_score * \
                self.config.features['special_considerations'].max_contribution_pct
        
        # Client loyalty bonus
        feature_scores['client_loyalty'] = pd.Series(
            [self.config.features['client_loyalty'].max_contribution_pct] * len(df),
            index=df.index
        )
        
        # Apply feature caps
        capped_scores = self._apply_feature_caps(feature_scores)
        
        # Combine features with weights
        combined_score = pd.Series([0.0] * len(df), index=df.index)
        for feature_name, scores in capped_scores.items():
            weight = self.config.features[feature_name].weight_in_component
            combined_score += scores * weight
        
        return self._normalize_component_score(combined_score)


def get_available_features() -> Dict[str, List[str]]:
    """Get list of all available features in the enhanced system."""
    return {
        'input_features': [
            'Employment_Sector', 'Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
            'Salary_Frequency', 'Housing_Status', 'Years_at_Current_Address',
            'Household_Head', 'Number_of_Dependents', 'Comaker_Relationship',
            'Comaker_Employment_Tenure_Months', 'Comaker_Net_Salary_Per_Cutoff',
            'Has_Community_Role', 'Paluwagan_Participation', 'Other_Income_Source',
            'Disaster_Preparedness', 'Is_Renewing_Client', 'Grace_Period_Usage_Rate',
            'Late_Payment_Count', 'Had_Special_Consideration'
        ],
        'output_features': [
            'Credit_Behavior_Score', 'Financial_Stability_Score', 'Cultural_Context_Score',
            'Credit_Risk_Score', 'Client_Type'
        ],
        'component_features': {
            'credit_behavior': ['payment_history', 'grace_period_usage', 'special_considerations', 'client_loyalty'],
            'financial_stability': ['income_adequacy', 'household_income_capacity', 'employment_stability', 'comaker_stability', 'address_stability', 'sector_stability'],
            'cultural_context': ['financial_discipline', 'community_integration', 'family_stability', 'income_diversification', 'relationship_strength', 'resilience_planning']
        }
    }

def validate_loan_application_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate dataframe against LoanApplicationRequest schema.
        
        Ensures compatibility with the API input schema.
        """
        required_columns = [
            'Employment_Sector', 'Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
            'Salary_Frequency', 'Housing_Status', 'Years_at_Current_Address',
            'Household_Head', 'Number_of_Dependents', 'Comaker_Relationship',
            'Comaker_Employment_Tenure_Months', 'Comaker_Net_Salary_Per_Cutoff',
            'Has_Community_Role', 'Paluwagan_Participation', 'Other_Income_Source',
            'Disaster_Preparedness', 'Is_Renewing_Client', 'Grace_Period_Usage_Rate',
            'Late_Payment_Count', 'Had_Special_Consideration'
        ]
        
        schema_rules = {
            'Employment_Sector': {'type': 'categorical', 'allowed': ['Public', 'Private']},
            'Employment_Tenure_Months': {'type': 'numeric', 'min': 1, 'dtype': 'int'},
            'Net_Salary_Per_Cutoff': {'type': 'numeric', 'min': 0.01, 'dtype': 'float'},
            'Salary_Frequency': {'type': 'categorical', 'allowed': ['Monthly', 'Bimonthly', 'Biweekly', 'Weekly']},
            'Housing_Status': {'type': 'categorical', 'allowed': ['Owned', 'Rented']},
            'Years_at_Current_Address': {'type': 'numeric', 'min': 0, 'dtype': 'float'},
            'Household_Head': {'type': 'categorical', 'allowed': ['Yes', 'No']},
            'Number_of_Dependents': {'type': 'numeric', 'min': 0, 'dtype': 'int'},
            'Comaker_Relationship': {'type': 'categorical', 'allowed': ['Friend', 'Parent', 'Sibling', 'Spouse']},
            'Comaker_Employment_Tenure_Months': {'type': 'numeric', 'min': 1, 'dtype': 'int'},
            'Comaker_Net_Salary_Per_Cutoff': {'type': 'numeric', 'min': 0.01, 'dtype': 'float'},
            'Has_Community_Role': {'type': 'categorical', 'allowed': ['Yes', 'No']},
            'Paluwagan_Participation': {'type': 'categorical', 'allowed': ['Yes', 'No']},
            'Other_Income_Source': {'type': 'categorical', 'allowed': ['None', 'Freelance', 'Business', 'OFW Remittance']},
            'Disaster_Preparedness': {'type': 'categorical', 'allowed': ['None', 'Savings', 'Insurance', 'Community Plan']},
            'Is_Renewing_Client': {'type': 'binary', 'allowed': [0, 1]},
            'Grace_Period_Usage_Rate': {'type': 'numeric', 'min': 0.0, 'max': 1.0, 'dtype': 'float'},
            'Late_Payment_Count': {'type': 'numeric', 'min': 0, 'dtype': 'int'},
            'Had_Special_Consideration': {'type': 'binary', 'allowed': [0, 1]}
        }
        
        issues_found = []
        df_fixed = df.copy()
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues_found.extend([f"Missing required column: {col}" for col in missing_columns])
            return df_fixed, issues_found
        
        # Validate each column
        for col, rules in schema_rules.items():
            if col not in df.columns:
                continue
                
            if rules['type'] == 'categorical':
                invalid_values = df[col].dropna().unique()
                invalid_values = [v for v in invalid_values if v not in rules['allowed']]
                if invalid_values:
                    issues_found.append(f"{col} has invalid values: {invalid_values}")
                    # Fix by mapping to first allowed value
                    df_fixed[col] = df_fixed[col].fillna(rules['allowed'][0])
                    mask = ~df_fixed[col].isin(rules['allowed'])
                    df_fixed.loc[mask, col] = rules['allowed'][0]
                    
            elif rules['type'] in ['numeric', 'binary']:
                # Check minimum values
                if 'min' in rules:
                    invalid_count = (df[col] < rules['min']).sum()
                    if invalid_count > 0:
                        issues_found.append(f"{col} has {invalid_count} values below minimum {rules['min']}")
                        df_fixed[col] = df_fixed[col].clip(lower=rules['min'])
                
                # Check maximum values        
                if 'max' in rules:
                    invalid_count = (df[col] > rules['max']).sum()
                    if invalid_count > 0:
                        issues_found.append(f"{col} has {invalid_count} values above maximum {rules['max']}")
                        df_fixed[col] = df_fixed[col].clip(upper=rules['max'])
        
        return df_fixed, issues_found

def create_loan_application_from_dict(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Create DataFrame from LoanApplicationRequest-compatible dictionary.
    
    Useful for converting API requests to DataFrame format.
    """
    # Ensure all required fields are present with defaults
    defaults = {
        'Employment_Sector': 'Private',
        'Employment_Tenure_Months': 12,
        'Net_Salary_Per_Cutoff': 15000.0,
        'Salary_Frequency': 'Monthly',
        'Housing_Status': 'Rented',
        'Years_at_Current_Address': 1.0,
        'Household_Head': 'No',
        'Number_of_Dependents': 0,
        'Comaker_Relationship': 'Friend',
        'Comaker_Employment_Tenure_Months': 12,
        'Comaker_Net_Salary_Per_Cutoff': 10000.0,
        'Has_Community_Role': 'No',
        'Paluwagan_Participation': 'No',
        'Other_Income_Source': 'None',
        'Disaster_Preparedness': 'None',
        'Is_Renewing_Client': 0,
        'Grace_Period_Usage_Rate': 0.0,
        'Late_Payment_Count': 0,
        'Had_Special_Consideration': 0
    }
    
    # Merge with defaults
    complete_data = {**defaults, **data}
    
    # Create DataFrame
    df = pd.DataFrame([complete_data])
    
    # Validate and fix any issues
    df_fixed, issues = validate_loan_application_schema(df)
    
    if issues:
        logger.warning(f"Schema issues found and fixed: {issues}")
    
    return df_fixed


class EnhancedCreditScoringTransformer:
    """
    Main transformer class with complete feature isolation and client-type specific scoring.
    
    ARCHITECTURE FEATURES:
    1. Client-type specific scoring (New vs Renewing)
    2. Mathematical feature isolation with caps
    3. Component-level contribution limits
    4. Severe data leakage prevention
    5. Interpretable scoring with explanations
    """
    
    def __init__(self, config: Optional[EnhancedCreditScoringConfig] = None):
        self.config = config or EnhancedCreditScoringConfig()
        self.processor = FeatureProcessor(self.config)
        
        # Initialize component scorers
        self.scorers = {
            'credit_behavior': CreditBehaviorScorer,
            'financial_stability': FinancialStabilityScorer,
            'cultural_context': CulturalContextScorer
        }
        
        logger.info("Enhanced Credit Scoring Transformer initialized successfully")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe with client-type specific scoring and feature isolation."""
        if 'Is_Renewing_Client' not in df.columns:
            raise ValueError("Column 'Is_Renewing_Client' is required for client type detection")
        
        df_transformed = df.copy()
        
        # Separate client types
        new_clients_mask = df['Is_Renewing_Client'] == 0
        renewing_clients_mask = df['Is_Renewing_Client'] == 1
        
        new_clients_df = df[new_clients_mask]
        renewing_clients_df = df[renewing_clients_mask]
        
        # Initialize score columns
        for component in ['Credit_Behavior_Score', 'Financial_Stability_Score', 'Cultural_Context_Score']:
            df_transformed[component] = 0.0
        
        # Process new clients
        if len(new_clients_df) > 0:
            new_scores = self._score_client_type(new_clients_df, ClientType.NEW)
            for component, scores in new_scores.items():
                df_transformed.loc[new_clients_mask, component] = scores
        
        # Process renewing clients
        if len(renewing_clients_df) > 0:
            renewing_scores = self._score_client_type(renewing_clients_df, ClientType.RENEWING)
            for component, scores in renewing_scores.items():
                df_transformed.loc[renewing_clients_mask, component] = scores
        
        # Calculate final credit risk scores
        df_transformed['Credit_Risk_Score'] = self._calculate_final_scores(df_transformed)
        
        # Add client type labels for transparency
        df_transformed['Client_Type'] = df['Is_Renewing_Client'].map({
            0: 'New',
            1: 'Renewing'
        })
        
        return df_transformed
    
    def _score_client_type(self, df: pd.DataFrame, client_type: ClientType) -> Dict[str, pd.Series]:
        """Score specific client type using appropriate configuration."""
        client_config = self.config.get_client_config(client_type)
        scores = {}
        
        for component_name, component_config in client_config.components.items():
            scorer_class = self.scorers[component_name]
            scorer = scorer_class(component_config, self.processor)
            
            component_score = scorer.calculate_component_score(df)
            scores[f"{component_name.title().replace('_', '_')}_Score"] = component_score
        
        return scores
    
    def _calculate_final_scores(self, df: pd.DataFrame) -> pd.Series:
        """Calculate final credit risk scores respecting client-type specific weights."""
        final_scores = pd.Series([0.0] * len(df), index=df.index)
        
        # Process each client type separately
        new_clients_mask = df['Is_Renewing_Client'] == 0
        renewing_clients_mask = df['Is_Renewing_Client'] == 1
        
        # New clients: Financial (80%) + Cultural (20%)
        if new_clients_mask.sum() > 0:
            new_config = self.config.get_client_config(ClientType.NEW)
            new_scores = (
                df.loc[new_clients_mask, 'Financial_Stability_Score'] * 
                new_config.components['financial_stability'].max_contribution_pct +
                df.loc[new_clients_mask, 'Cultural_Context_Score'] * 
                new_config.components['cultural_context'].max_contribution_pct
            )
            final_scores.loc[new_clients_mask] = new_scores
        
        # Renewing clients: Credit (60%) + Financial (37%) + Cultural (3%)
        if renewing_clients_mask.sum() > 0:
            renewing_config = self.config.get_client_config(ClientType.RENEWING)
            renewing_scores = (
                df.loc[renewing_clients_mask, 'Credit_Behavior_Score'] * 
                renewing_config.components['credit_behavior'].max_contribution_pct +
                df.loc[renewing_clients_mask, 'Financial_Stability_Score'] * 
                renewing_config.components['financial_stability'].max_contribution_pct +
                df.loc[renewing_clients_mask, 'Cultural_Context_Score'] * 
                renewing_config.components['cultural_context'].max_contribution_pct
            )
            final_scores.loc[renewing_clients_mask] = renewing_scores
        
        # Ensure scores are in valid range
        return pd.Series(np.clip(final_scores, 0, 1), index=df.index)
    
    def get_feature_importance(self, client_type: Optional[ClientType] = None) -> Dict[str, Any]:
        """Get feature importance analysis showing caps and actual contributions."""
        if client_type:
            client_configs = {client_type: self.config.get_client_config(client_type)}
        else:
            client_configs = self.config.client_configs
        
        importance_analysis = {}
        
        for ct, config in client_configs.items():
            client_analysis = {
                'total_max_contribution': config.get_total_max_contribution(),
                'components': {}
            }
            
            for comp_name, component in config.components.items():
                comp_analysis = {
                    'component_max_pct': component.max_contribution_pct,
                    'features': {}
                }
                
                for feat_name, feature in component.features.items():
                    max_total_impact = feature.max_contribution_pct * component.max_contribution_pct
                    comp_analysis['features'][feat_name] = {
                        'max_contribution_to_component': feature.max_contribution_pct,
                        'weight_in_component': feature.weight_in_component,
                        'max_total_impact_on_final_score': max_total_impact,
                        'effective_cap_pct': max_total_impact * 100
                    }
                
                client_analysis['components'][comp_name] = comp_analysis
            
            importance_analysis[ct.value] = client_analysis
        
        return importance_analysis
    
    def get_score_explanation(self, applicant_data: pd.Series) -> Dict[str, Any]:
        """Get detailed explanation of score calculation with feature contributions."""
        # Determine client type
        is_renewing = bool(applicant_data.get('Is_Renewing_Client', 0))
        client_type = ClientType.RENEWING if is_renewing else ClientType.NEW
        
        # Convert to DataFrame for processing
        df_single = pd.DataFrame([applicant_data])
        transformed = self.transform(df_single)
        
        # Get scores
        credit_behavior_score = transformed['Credit_Behavior_Score'].iloc[0]
        financial_stability_score = transformed['Financial_Stability_Score'].iloc[0]
        cultural_context_score = transformed['Cultural_Context_Score'].iloc[0]
        final_score = transformed['Credit_Risk_Score'].iloc[0]
        
        # Get configuration
        client_config = self.config.get_client_config(client_type)
        
        explanation = {
            'final_credit_risk_score': round(final_score, 4),
            'client_type': client_type.value,
            'score_interpretation': self._interpret_score(final_score),
            
            'component_contributions': {
                'financial_stability': {
                    'score': round(financial_stability_score, 4),
                    'max_contribution_pct': client_config.components['financial_stability'].max_contribution_pct,
                    'actual_contribution': round(financial_stability_score * client_config.components['financial_stability'].max_contribution_pct, 4),
                    'features_breakdown': self._get_financial_features_breakdown(applicant_data, client_type)
                },
                'cultural_context': {
                    'score': round(cultural_context_score, 4),
                    'max_contribution_pct': client_config.components['cultural_context'].max_contribution_pct,
                    'actual_contribution': round(cultural_context_score * client_config.components['cultural_context'].max_contribution_pct, 4),
                    'features_breakdown': self._get_cultural_features_breakdown(applicant_data, client_type)
                }
            },
            
            'feature_isolation_summary': {
                'total_possible_max_contribution': client_config.get_total_max_contribution(),
                'actual_total_contribution': round(final_score, 4),
                'leakage_prevention': {
                    'community_role_impact': 'Limited to 0.5% max for renewing, 2% max for new clients',
                    'paluwagan_impact': 'Limited to 0.8% max for renewing, 3% max for new clients',
                    'individual_feature_caps': 'All features mathematically capped',
                    'component_level_caps': 'Components cannot exceed specified percentages'
                }
            },
            
            'client_type_specific_weights': {
                client_type.value: {
                    comp_name: comp_config.max_contribution_pct 
                    for comp_name, comp_config in client_config.components.items()
                }
            }
        }
        
        # Add credit behavior for renewing clients
        if client_type == ClientType.RENEWING:
            explanation['component_contributions']['credit_behavior'] = {
                'score': round(credit_behavior_score, 4),
                'max_contribution_pct': client_config.components['credit_behavior'].max_contribution_pct,
                'actual_contribution': round(credit_behavior_score * client_config.components['credit_behavior'].max_contribution_pct, 4),
                'features_breakdown': self._get_credit_features_breakdown(applicant_data)
            }
        
        return explanation
    
    def _interpret_score(self, score: float) -> str:
        """Interpret credit risk score."""
        if score >= 0.8:
            return "Excellent credit risk profile - High approval likelihood"
        elif score >= 0.7:
            return "Good credit risk profile - Favorable terms likely" 
        elif score >= 0.6:
            return "Acceptable credit risk profile - Standard terms"
        elif score >= 0.5:
            return "Marginal credit risk profile - Review required"
        elif score >= 0.4:
            return "Poor credit risk profile - Higher risk/rates"
        else:
            return "Very poor credit risk profile - Decline recommended"
    
    def _get_financial_features_breakdown(self, applicant_data: pd.Series, client_type: ClientType) -> Dict[str, Any]:
        """Get breakdown of financial features contribution."""
        primary_salary = applicant_data.get('Net_Salary_Per_Cutoff', 0)
        comaker_salary = applicant_data.get('Comaker_Net_Salary_Per_Cutoff', 0)
        employment_tenure = applicant_data.get('Employment_Tenure_Months', 0)
        comaker_tenure = applicant_data.get('Comaker_Employment_Tenure_Months', 0)
        address_years = applicant_data.get('Years_at_Current_Address', 0)
        dependents = applicant_data.get('Number_of_Dependents', 0)
        
        household_size = 1 + dependents + (1 if comaker_salary > 0 else 0)
        income_per_member = (primary_salary + comaker_salary) / household_size
        total_household_income = primary_salary + comaker_salary
        
        client_config = self.config.get_client_config(client_type)
        financial_config = client_config.components['financial_stability']
        
        return {
            'income_adequacy': {
                'value': f"₱{income_per_member:,.0f} per household member",
                'max_impact_pct': financial_config.features['income_adequacy'].max_contribution_pct * 100,
                'description': 'Primary income divided by household size'
            },
            'household_income_capacity': {
                'value': f"₱{total_household_income:,.0f} total household income",
                'max_impact_pct': financial_config.features['household_income_capacity'].max_contribution_pct * 100,
                'description': 'Combined primary and comaker income'
            },
            'employment_stability': {
                'value': f"{employment_tenure} months tenure",
                'max_impact_pct': financial_config.features['employment_stability'].max_contribution_pct * 100,
                'description': 'Length of current employment'
            },
            'comaker_stability': {
                'value': f"{comaker_tenure} months comaker tenure",
                'max_impact_pct': financial_config.features['comaker_stability'].max_contribution_pct * 100,
                'description': 'Strength of comaker employment'
            },
            'address_stability': {
                'value': f"{address_years} years at current address",
                'max_impact_pct': financial_config.features['address_stability'].max_contribution_pct * 100,
                'description': 'Residential stability indicator'
            }
        }
    
    def _get_cultural_features_breakdown(self, applicant_data: pd.Series, client_type: ClientType) -> Dict[str, Any]:
        """Get breakdown of cultural features contribution with leakage prevention details."""
        community_role = applicant_data.get('Has_Community_Role', 'No')
        paluwagan = applicant_data.get('Paluwagan_Participation', 'No')
        housing_status = applicant_data.get('Housing_Status', 'Rented')
        household_head = applicant_data.get('Household_Head', 'No')
        other_income = applicant_data.get('Other_Income_Source', 'None')
        comaker_relationship = applicant_data.get('Comaker_Relationship', 'Friend')
        disaster_preparedness = applicant_data.get('Disaster_Preparedness', 'None')
        
        client_config = self.config.get_client_config(client_type)
        cultural_config = client_config.components['cultural_context']
        
        return {
            'financial_discipline': {
                'value': f"Paluwagan: {paluwagan}",
                'max_impact_pct': cultural_config.features['financial_discipline'].max_contribution_pct * 100,
                'leakage_prevention': 'SEVERELY LIMITED - was 66% default rate difference',
                'description': 'Participation in traditional savings groups'
            },
            'community_integration': {
                'value': f"Community Role: {community_role}",
                'max_impact_pct': cultural_config.features['community_integration'].max_contribution_pct * 100,
                'leakage_prevention': 'SEVERELY LIMITED - was perfect predictor (0% default)',
                'description': 'Leadership role in community'
            },
            'family_stability': {
                'value': f"Housing: {housing_status}, Head: {household_head}",
                'max_impact_pct': cultural_config.features['family_stability'].max_contribution_pct * 100,
                'description': 'Housing ownership and household leadership'
            },
            'income_diversification': {
                'value': f"Other Income: {other_income}",
                'max_impact_pct': cultural_config.features['income_diversification'].max_contribution_pct * 100,
                'description': 'Additional income sources beyond employment'
            },
            'relationship_strength': {
                'value': f"Comaker: {comaker_relationship}",
                'max_impact_pct': cultural_config.features['relationship_strength'].max_contribution_pct * 100,
                'description': 'Relationship type with loan comaker'
            },
            'resilience_planning': {
                'value': f"Disaster Preparedness: {disaster_preparedness}",
                'max_impact_pct': cultural_config.features['resilience_planning'].max_contribution_pct * 100,
                'description': 'Plans for dealing with financial shocks'
            }
        }
    
    def _get_credit_features_breakdown(self, applicant_data: pd.Series) -> Dict[str, Any]:
        """Get breakdown of credit behavior features (renewing clients only)."""
        late_payments = applicant_data.get('Late_Payment_Count', 0)
        grace_usage = applicant_data.get('Grace_Period_Usage_Rate', 0)
        special_consideration = applicant_data.get('Had_Special_Consideration', 0)
        
        return {
            'payment_history': {
                'value': f"{late_payments} late payments",
                'max_impact_pct': 21,  # 35% of 60% component
                'description': 'Historical payment performance'
            },
            'grace_period_usage': {
                'value': f"{grace_usage:.1%} grace period usage",
                'max_impact_pct': 12,  # 20% of 60% component
                'description': 'Frequency of payment extensions'
            },
            'special_considerations': {
                'value': 'Yes' if special_consideration else 'No',
                'max_impact_pct': 6,   # 10% of 60% component
                'description': 'Required special payment arrangements'
            },
            'client_loyalty': {
                'value': 'Renewing client',
                'max_impact_pct': 6,   # 10% of 60% component
                'description': 'Loyalty bonus for returning customers'
            }
        }
