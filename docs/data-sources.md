# Dataset Structure and Data Requirements

## Primary Data Sources

### 1. Main Dataset: Filipino Microfinance Loan Records
**Source Options:**
- Partner with Philippine MFIs (CARD Bank, ASA Philippines, TSPI)
- BSP Financial Inclusion Survey data (publicly available)
- Kaggle microfinance datasets adapted with Filipino context
- Synthetic data generation based on Philippine economic patterns

### 2. Supplementary Cultural Context Data
- BSP Financial Inclusion Survey (2017-2023)
- FIES (Family Income and Expenditure Survey) - PSA
- Regional economic indicators - PSA/NEDA
- OFW remittance data - BSP
- Disaster/typhoon occurrence data - PAGASA

## Dataset Structure and Columns

### Core Credit Scoring Variables (Standard)

#### Personal Demographics
```
- customer_id (unique identifier)
- age (numerical)
- gender (Male/Female/Other)
- marital_status (Single/Married/Divorced/Widowed)
- education_level (Elementary/HS/College/Graduate)
- number_of_dependents (numerical)
- employment_type (Employed/Self-Employed/Unemployed/OFW)
- years_employed (numerical)
```

#### Financial Information
```
- monthly_income (numerical, PHP)
- annual_income (numerical, PHP)
- existing_debt (numerical, PHP)
- debt_to_income_ratio (numerical)
- bank_account_type (Savings/Current/None)
- credit_history_length (months)
- number_of_existing_loans (numerical)
- previous_defaults (0/1)
```

#### Loan Details
```
- loan_amount (numerical, PHP)
- loan_purpose (Business/Education/Medical/Agriculture/Housing)
- loan_term (months)
- interest_rate (percentage)
- collateral_type (Property/Vehicle/None/Social_Guarantee)
- loan_approval_date (date)
- loan_status (Current/Paid/Default/Late)
```

### Cultural Context Variables (Your Innovation)

#### Filipino-Specific Cultural Factors
```
- remittance_dependency (0-1 scale: family income from OFW)
- extended_family_size (number of family members supported)
- community_social_capital (0-1 scale: bayanihan participation)
- informal_credit_participation (0/1: paluwagan, rotating credit)
- regional_location (Luzon/Visayas/Mindanao/NCR)
- province_code (specific province)
- urban_rural_classification (Urban/Rural)
- disaster_exposure_index (0-1 scale: typhoon frequency in area)
```

#### Economic Context Variables
```
- seasonal_income_variation (0-1 scale: income stability)
- agricultural_dependency (0/1: farming/fishing occupation)
- digital_financial_inclusion (0-1 scale: mobile banking usage)
- local_economic_development_index (regional prosperity score)
- distance_to_bank_branch (kilometers)
- mobile_money_usage (0/1: GCash, PayMaya usage)
```

#### Social Capital Indicators
```
- community_tenure (years living in current location)
- local_business_network (0/1: part of local business associations)
- church_organization_membership (0/1)
- cooperative_membership (0/1)
- community_leadership_role (0/1)
```

### Target Variables

#### Primary Target
```
- loan_default (0/1: binary classification)
- days_past_due (numerical: for severity analysis)
- repayment_behavior (Excellent/Good/Fair/Poor)
```

#### Recommendation System Targets
```
- loan_product_taken (categorical: actual product chosen)
- loan_satisfaction_score (1-5 scale)
- repeat_customer (0/1)
- loan_completion_rate (0-1 scale)
```

## Sample Dataset Structure

```python
# Example dataset columns (pandas DataFrame)
filipino_microfinance_data = {
    # Standard Demographics
    'customer_id': [1001, 1002, 1003, ...],
    'age': [32, 45, 28, ...],
    'gender': ['Female', 'Male', 'Female', ...],
    'education_level': ['HS Graduate', 'College Graduate', 'Elementary', ...],
    'employment_type': ['Self-Employed', 'OFW', 'Employed', ...],
    
    # Financial Data
    'monthly_income': [15000, 25000, 12000, ...],  # PHP
    'debt_to_income_ratio': [0.3, 0.45, 0.2, ...],
    'existing_loans': [1, 0, 2, ...],
    
    # Cultural Context (Your Innovation)
    'remittance_dependency': [0.0, 0.7, 0.2, ...],  # 0-1 scale
    'extended_family_size': [8, 12, 5, ...],
    'community_social_capital': [0.8, 0.6, 0.9, ...],  # bayanihan score
    'informal_credit_participation': [1, 0, 1, ...],  # paluwagan
    'regional_location': ['Luzon', 'Mindanao', 'Visayas', ...],
    'disaster_exposure_index': [0.8, 0.3, 0.6, ...],  # typhoon exposure
    
    # Loan Information
    'loan_amount': [50000, 100000, 25000, ...],  # PHP
    'loan_purpose': ['Business', 'Agriculture', 'Education', ...],
    'loan_term': [12, 24, 6, ...],  # months
    
    # Target Variables
    'loan_default': [0, 0, 1, ...],  # binary
    'repayment_behavior': ['Good', 'Excellent', 'Poor', ...],
    
    # Recommendation Targets
    'loan_product_taken': ['Micro-Business', 'Agricultural', 'Emergency', ...],
    'loan_satisfaction_score': [4, 5, 2, ...]
}
```

## Data Size Expectations

### Realistic Dataset Size for Bachelor's Thesis
- **Minimum**: 5,000-10,000 loan records
- **Optimal**: 15,000-25,000 loan records
- **Features**: 25-35 columns (15 standard + 10-20 cultural)
- **Time span**: 2-3 years of historical data

### Data Split
- **Training**: 70% (credit scoring model development)
- **Validation**: 15% (hyperparameter tuning)
- **Test**: 15% (final evaluation and recommendation testing)

## Data Collection Strategy

### Option 1: Partnership with Philippine MFIs
- Contact CARD Bank, ASA Philippines, or TSPI Development Corporation
- Request anonymized historical loan data
- Supplement with cultural context surveys

### Option 2: Public Data Integration
- Use BSP Financial Inclusion Survey as base
- Combine with synthetic loan data generation
- Add cultural context from FIES and regional statistics

### Option 3: Kaggle + Cultural Enhancement
- Start with existing microfinance datasets
- Adapt/simulate Filipino cultural context variables
- Use Philippine economic data for regional adjustments

## Key Data Features for Your Innovation

### Most Important Standard Features
Age, loan amount, and interest rate are consistently top predictors in microfinance default models.

### Your Cultural Innovation Features (Novel)
1. **Remittance Dependency Index** - uniquely Filipino
2. **Community Social Capital Score** - bayanihan culture
3. **Disaster Resilience Capacity** - Philippines-specific
4. **Informal Credit History** - paluwagan participation
5. **Extended Family Network Size** - Filipino family structure

## Cultural Context Data Sources Detail

### BSP Financial Inclusion Survey Variables
```
- Access to formal financial services
- Usage of digital financial services
- Barriers to financial inclusion
- Preference for informal financial services
- Regional financial behavior patterns
```

### FIES (Family Income and Expenditure Survey) Variables
```
- Household income sources
- Expenditure patterns
- Remittance receipts
- Family size and composition
- Regional economic indicators
```

### OFW Remittance Data (BSP)
```
- Regional remittance flow patterns
- Seasonal remittance variations
- Remittance dependency by province
- Economic impact of remittances
```

### Disaster/Climate Data (PAGASA)
```
- Typhoon frequency by region
- Seasonal disaster patterns
- Economic impact of natural disasters
- Community resilience indicators
```

## Data Quality Considerations

### Missing Data Handling
- Cultural context variables may have higher missing rates
- Implement culturally-appropriate imputation strategies
- Consider regional/demographic patterns for missing value estimation

### Bias Mitigation
- Ensure representative sampling across regions
- Balance urban/rural representation
- Include diverse income levels and employment types
- Validate cultural variable accuracy with domain experts

### Privacy and Ethics
- Anonymize all personal identifiers
- Ensure cultural representation is respectful
- Comply with Philippine Data Privacy Act
- Obtain proper consent for cultural data usage

This dataset structure provides the foundation for both your credit scoring innovation and recommendation system, with clear cultural differentiation that makes your thesis unique and valuable for the Philippine microfinance industry.