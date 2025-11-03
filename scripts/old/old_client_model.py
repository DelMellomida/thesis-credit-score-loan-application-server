import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Import SHAP for interpretability analysis
try:
    import shap
    HAS_SHAP = True
    print("SHAP library loaded successfully")
except ImportError:
    HAS_SHAP = False
    print("SHAP library not available. Will skip SHAP analysis.")

# Configuration
RAW_DATA_PATH = '../data/raw/synthetic_training_data12.csv'
MODEL_DIR = '../models'

print("="*80)
print("ROBUST MODEL TRAINING WITH VIF-BASED FEATURE SELECTION")
print("="*80)

# Load and prepare data
print("\n=== LOADING AND PREPROCESSING DATA ===")
df = pd.read_csv(RAW_DATA_PATH)
print(f"Original dataset shape: {df.shape}")

# Check for duplicates
duplicates = df.duplicated()
print(f"Number of duplicate rows: {duplicates.sum()}")
if duplicates.sum() > 0:
    df = df.drop_duplicates()
    print(f"Dataset shape after removing duplicates: {df.shape}")

# Filter for new clients
new_clients_df = df[df['Is_Renewing_Client'] == 0].copy()
print(f"New clients dataset shape: {new_clients_df.shape}")

# Check target variable distribution
print(f"\nTarget variable (Default) distribution:")
print(new_clients_df['Default'].value_counts())
print(f"Default rate: {new_clients_df['Default'].mean():.2%}")

# Define initial feature sets
financial_features = [
    'Employment_Sector', 'Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
    'Salary_Frequency', 'Housing_Status', 'Years_at_Current_Address',
    'Number_of_Dependents', 'Comaker_Employment_Tenure_Months',
    'Comaker_Net_Salary_Per_Cutoff', 'Other_Income_Source'
]

cultural_features = [
    'Household_Head', 'Comaker_Relationship', 'Has_Community_Role',
    'Paluwagan_Participation', 'Disaster_Preparedness'
]

all_features = financial_features + cultural_features
X = new_clients_df[all_features].copy()
y = new_clients_df['Default']

# Data cleaning and validation
print("\n=== DATA CLEANING ===")

# Numerical columns for outlier handling
numerical_cols = ['Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
                 'Years_at_Current_Address', 'Number_of_Dependents',
                 'Comaker_Employment_Tenure_Months', 'Comaker_Net_Salary_Per_Cutoff']

# Remove outliers using IQR method
print("Outlier detection and capping for numerical features:")
for col in numerical_cols:
    if col in X.columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
        if outliers > 0:
            print(f"  {col}: {outliers} outliers detected and capped")
            X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)

# Ensure non-negative values for numerical features
for col in numerical_cols:
    if col in X.columns and (X[col] < 0).any():
        print(f"  WARNING: Negative values found in {col}, setting to 0")
        X[col] = X[col].clip(lower=0)

# Standardize categorical values
categorical_features = ['Employment_Sector', 'Salary_Frequency', 'Housing_Status', 
                       'Household_Head', 'Comaker_Relationship', 'Has_Community_Role',
                       'Paluwagan_Participation', 'Other_Income_Source', 'Disaster_Preparedness']

print("\nStandardizing categorical values...")
for col in categorical_features:
    if col in X.columns and X[col].dtype == 'object':
        X[col] = X[col].str.strip()
        if col in ['Housing_Status', 'Employment_Sector']:
            X[col] = X[col].str.capitalize()
        elif col in ['Household_Head', 'Has_Community_Role', 'Paluwagan_Participation']:
            X[col] = X[col].str.capitalize()

# Handle missing values
print("\n=== MISSING VALUE IMPUTATION ===")
for col in categorical_features:
    if col in X.columns and X[col].isnull().any():
        missing_count = X[col].isnull().sum()
        if col == 'Comaker_Relationship':
            fill_value = 'Friend'
        elif col == 'Other_Income_Source':
            fill_value = 'None'
        elif col == 'Disaster_Preparedness':
            fill_value = 'None'
        elif col == 'Salary_Frequency':
            fill_value = 'Monthly'
        else:
            mode_value = X[col].mode()
            fill_value = mode_value[0] if not mode_value.empty else 'Unknown'
        
        X[col].fillna(fill_value, inplace=True)
        print(f"  {col}: Filled {missing_count} missing values with '{fill_value}'")

# Fill numerical missing values with median
for col in numerical_cols:
    if col in X.columns and X[col].isnull().any():
        missing_count = X[col].isnull().sum()
        median_value = X[col].median()
        X[col].fillna(median_value, inplace=True)
        print(f"  {col}: Filled {missing_count} missing values with median {median_value:.2f}")

# Apply capped penalty logic
X['Number_of_Dependents'] = np.minimum(X['Number_of_Dependents'], 5)
print(f"\nApplied capping to Number_of_Dependents (max=5)")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Train default rate: {y_train.mean():.2%}")
print(f"Test default rate: {y_test.mean():.2%}")

print("\n" + "="*80)
print("PHASE 1: ITERATIVE VIF-BASED FEATURE SELECTION")
print("="*80)

def create_preprocessor_and_vif_check(features_dict, numerical_features):
    """Create preprocessor and calculate VIF for current feature set"""
    
    # Define ordinal and nominal features based on current feature set
    ordinal_features = {}
    nominal_features = []
    
    for feature in features_dict['all_features']:
        if feature == 'Salary_Frequency':
            ordinal_features['Salary_Frequency'] = ['Monthly', 'Bimonthly', 'Biweekly', 'Weekly']
        elif feature == 'Comaker_Relationship':
            ordinal_features['Comaker_Relationship'] = ['Friend', 'Sibling', 'Parent', 'Spouse']
        elif feature == 'Other_Income_Source':
            ordinal_features['Other_Income_Source'] = ['None', 'Freelance', 'Business', 'OFW Remittance']
        elif feature == 'Disaster_Preparedness':
            ordinal_features['Disaster_Preparedness'] = ['None', 'Savings', 'Insurance', 'Community Plan']
        elif feature == 'Paluwagan_Participation':
            ordinal_features['Paluwagan_Participation'] = ['Never', 'Rarely', 'Sometimes', 'Frequently']
        elif feature == 'Has_Community_Role':
            ordinal_features['Has_Community_Role'] = ['None', 'Member', 'Leader', 'Multiple Leader']
        elif feature in categorical_features and feature not in ordinal_features:
            nominal_features.append(feature)
    
    # Create ordinal transformers
    ordinal_transformers = []
    for feature, categories in ordinal_features.items():
        if feature in features_dict['all_features']:
            ordinal_transformers.append(
                (f'ordinal_{feature}',
                 OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=-1),
                 [feature])
            )
    
    # Filter numerical features to only include those in current feature set
    current_numerical = [f for f in numerical_features if f in features_dict['all_features']]
    current_nominal = [f for f in nominal_features if f in features_dict['all_features']]
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), current_numerical),
            ('nom', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), current_nominal)
        ] + ordinal_transformers,
        remainder='passthrough'
    )
    
    return preprocessor, ordinal_features, nominal_features

def calculate_vif(X_processed, feature_names):
    """Calculate VIF for all features"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_names
    vif_data["VIF"] = [variance_inflation_factor(X_processed, i) for i in range(X_processed.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)

# Initialize current feature sets
current_features = {
    'financial_features': financial_features.copy(),
    'cultural_features': cultural_features.copy(),
    'all_features': all_features.copy()
}

print("\n1. Starting iterative VIF-based feature removal...")
iteration = 0
max_iterations = 20  # Safety limit

while iteration < max_iterations:
    iteration += 1
    print(f"\n--- VIF Iteration {iteration} ---")
    print(f"Current feature count: {len(current_features['all_features'])}")
    
    # Create preprocessor for current feature set
    try:
        X_current = X_train[current_features['all_features']].copy()
        preprocessor, ordinal_features, nominal_features = create_preprocessor_and_vif_check(
            current_features, numerical_cols
        )
        
        # Fit preprocessor and transform data
        X_processed = preprocessor.fit_transform(X_current)
        feature_names = preprocessor.get_feature_names_out()
        
        # Calculate VIF
        vif_data = calculate_vif(X_processed, feature_names)
        
        # Find maximum VIF
        max_vif = vif_data.iloc[0]['VIF']
        max_vif_feature = vif_data.iloc[0]['Feature']
        
        print(f"Highest VIF: {max_vif:.2f} for feature '{max_vif_feature}'")
        
        if max_vif <= 5.0:
            print("✓ All features have VIF <= 5.0. VIF cleaning complete!")
            break
        
        # Identify original feature to remove
        # Map processed feature name back to original feature name
        feature_to_remove = None
        for orig_feature in current_features['all_features']:
            if orig_feature in max_vif_feature or max_vif_feature.startswith(f'num__{orig_feature}') or max_vif_feature.startswith(f'nom__{orig_feature}') or max_vif_feature.startswith(f'ordinal_{orig_feature}'):
                feature_to_remove = orig_feature
                break
        
        if feature_to_remove is None:
            # Fallback: try to extract feature name from processed name
            for orig_feature in current_features['all_features']:
                if orig_feature.lower() in max_vif_feature.lower():
                    feature_to_remove = orig_feature
                    break
        
        if feature_to_remove:
            print(f"⚠️  Removing feature: {feature_to_remove}")
            
            # Remove from all relevant lists
            if feature_to_remove in current_features['financial_features']:
                current_features['financial_features'].remove(feature_to_remove)
            if feature_to_remove in current_features['cultural_features']:
                current_features['cultural_features'].remove(feature_to_remove)
            current_features['all_features'].remove(feature_to_remove)
        else:
            print(f"⚠️  Could not identify original feature for '{max_vif_feature}'. Breaking loop.")
            break
            
    except Exception as e:
        print(f"Error in VIF calculation: {str(e)}")
        break

print(f"\n=== VIF CLEANING RESULTS ===")
print(f"Iterations completed: {iteration}")
print(f"Final feature count: {len(current_features['all_features'])}")
print(f"Removed features: {set(all_features) - set(current_features['all_features'])}")
print(f"Remaining features: {current_features['all_features']}")

# Display final VIF scores
if iteration <= max_iterations:
    print(f"\nFinal VIF scores:")
    print(vif_data.head(10).to_string(index=False))

print("\n" + "="*80)
print("PHASE 2: MODEL TRAINING AND COMPARISON")
print("="*80)

# Update X_train and X_test with final feature set
X_train_final = X_train[current_features['all_features']].copy()
X_test_final = X_test[current_features['all_features']].copy()

# Create final preprocessor
final_preprocessor, final_ordinal_features, final_nominal_features = create_preprocessor_and_vif_check(
    current_features, numerical_cols
)

print("\n2. Training Forced-Sensitive Model (C=10.0)...")

# Create forced-sensitive pipeline
forced_sensitive_pipeline = Pipeline([
    ('preprocessor', final_preprocessor),
    ('classifier', LogisticRegression(
        penalty='l2',
        C=10.0,
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
        solver='lbfgs'
    ))
])

# Train forced-sensitive model
forced_sensitive_pipeline.fit(X_train_final, y_train)

# Evaluate forced-sensitive model
y_pred_sensitive = forced_sensitive_pipeline.predict(X_test_final)
y_pred_proba_sensitive = forced_sensitive_pipeline.predict_proba(X_test_final)[:, 1]

print("\nForced-Sensitive Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_sensitive):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_sensitive, zero_division=0):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_sensitive, zero_division=0):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred_sensitive, zero_division=0):.3f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_sensitive):.3f}")

print("\n3. Training Final Balanced Model with GridSearchCV...")

# Create balanced pipeline for grid search
balanced_pipeline = Pipeline([
    ('preprocessor', final_preprocessor),
    ('classifier', LogisticRegression(
        penalty='l2',
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
        solver='lbfgs'
    ))
])

# Define parameter grid
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10]
}

# Perform grid search
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    balanced_pipeline,
    param_grid,
    cv=cv_strategy,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_final, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV ROC-AUC score: {grid_search.best_score_:.3f}")

# Get best model
best_balanced_model = grid_search.best_estimator_

print("\n" + "="*80)
print("PHASE 3: FINAL MODEL EVALUATION AND SERIALIZATION")
print("="*80)

# Evaluate best balanced model
y_pred_balanced = best_balanced_model.predict(X_test_final)
y_pred_proba_balanced = best_balanced_model.predict_proba(X_test_final)[:, 1]

print("\n4. Final Balanced Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_balanced):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_balanced, zero_division=0):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_balanced, zero_division=0):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred_balanced, zero_division=0):.3f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_balanced):.3f}")

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_balanced)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Final Balanced Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_balanced)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba_balanced):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Final Balanced Model')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n5. SHAP Analysis...")

if HAS_SHAP:
    try:
        # Get processed data for SHAP
        X_train_processed = forced_sensitive_pipeline.named_steps['preprocessor'].transform(X_train_final)
        X_test_processed = best_balanced_model.named_steps['preprocessor'].transform(X_test_final)
        processed_feature_names = best_balanced_model.named_steps['preprocessor'].get_feature_names_out()
        
        print("Calculating SHAP values for both models...")
        
        # SHAP for forced-sensitive model
        explainer_sensitive = shap.LinearExplainer(
            forced_sensitive_pipeline.named_steps['classifier'],
            X_train_processed,
            feature_perturbation="interventional"
        )
        shap_values_sensitive = explainer_sensitive.shap_values(X_test_processed)
        
        # SHAP for balanced model
        explainer_balanced = shap.LinearExplainer(
            best_balanced_model.named_steps['classifier'],
            X_train_processed,
            feature_perturbation="interventional"
        )
        shap_values_balanced = explainer_balanced.shap_values(X_test_processed)
        
        # Create stacked SHAP comparison plots for better readability
        print("Generating stacked SHAP comparison plots...")

        # Create figure with vertical stacking for better feature name readability
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 28))

        # Top subplot - Sensitive model
        plt.sca(ax1)
        shap.summary_plot(shap_values_sensitive, X_test_processed, 
                        feature_names=processed_feature_names, 
                        show=False, 
                        max_display=len(processed_feature_names))
        ax1.set_title('Forced-Sensitive Model (C=10.0)', fontsize=18, fontweight='bold', pad=20)
        ax1.tick_params(axis='y', labelsize=12)
        ax1.tick_params(axis='x', labelsize=12)

        # Bottom subplot - Balanced model
        plt.sca(ax2)
        shap.summary_plot(shap_values_balanced, X_test_processed, 
                        feature_names=processed_feature_names, 
                        show=False, 
                        max_display=len(processed_feature_names))
        ax2.set_title(f"Final Balanced Model (C={best_balanced_model.named_steps['classifier'].C})", 
                    fontsize=18, fontweight='bold', pad=20)
        ax2.tick_params(axis='y', labelsize=12)
        ax2.tick_params(axis='x', labelsize=12)

        # Adjust spacing between plots
        plt.tight_layout(pad=4.0)

        # Optional: Save the plot
        # plt.savefig('shap_comparison_stacked.png', dpi=300, bbox_inches='tight', facecolor='white')

        plt.show()

        # Alternative: Create individual plots if stacked version is still too cramped
        print("\nAlternative: Creating individual SHAP plots...")

        # Individual plot for Sensitive model
        plt.figure(figsize=(16, 12))
        shap.summary_plot(shap_values_sensitive, X_test_processed, 
                        feature_names=processed_feature_names, 
                        show=False, 
                        max_display=len(processed_feature_names))
        plt.title('Forced-Sensitive Model (C=10.0)', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()

        # Individual plot for Balanced model
        plt.figure(figsize=(16, 12))
        shap.summary_plot(shap_values_balanced, X_test_processed, 
                        feature_names=processed_feature_names, 
                        show=False, 
                        max_display=len(processed_feature_names))
        plt.title(f"Final Balanced Model (C={best_balanced_model.named_steps['classifier'].C})", 
                fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"SHAP analysis failed: {str(e)}")
        print("Continuing without SHAP visualization...")

print("\n6. Serializing Final Model...")

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Save the best model pipeline
model_path = f"{MODEL_DIR}/new_client_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(best_balanced_model, f)

print(f"✓ Final model saved successfully to: {model_path}")

# Verify saved model
print("\nVerifying saved model...")
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

sample_predictions = loaded_model.predict(X_test_final[:5])
print(f"Sample predictions from loaded model: {sample_predictions}")
print("✓ Model pipeline successfully serialized and verified!")

print("\n" + "="*80)
print("FINAL SUMMARY AND RECOMMENDATIONS")
print("="*80)

print(f"\n📊 FINAL RESULTS:")
print(f"• Features used: {len(current_features['all_features'])}")
print(f"• Features removed due to multicollinearity: {len(all_features) - len(current_features['all_features'])}")
print(f"• Optimal regularization parameter: C = {best_balanced_model.named_steps['classifier'].C}")
print(f"• Final model ROC AUC: {roc_auc_score(y_test, y_pred_proba_balanced):.3f}")

print(f"\n🔧 MODEL CONFIGURATION:")
print(f"• Algorithm: Logistic Regression with L2 regularization")
print(f"• Class balancing: Balanced class weights")
print(f"• Cross-validation: 3-fold stratified")
print(f"• VIF threshold: ≤ 5.0 (multicollinearity removed)")

print(f"\n📁 DELIVERABLES:")
print(f"• Model file: {model_path}")
print(f"• Pipeline includes: Preprocessor + Trained Classifier")
print(f"• Ready for production deployment")

# Performance comparison
print(f"\n📈 MODEL COMPARISON:")
print(f"{'Metric':<15} {'Sensitive':<12} {'Balanced':<12} {'Difference':<12}")
print("-" * 55)
print(f"{'ROC AUC':<15} {roc_auc_score(y_test, y_pred_proba_sensitive):<12.3f} {roc_auc_score(y_test, y_pred_proba_balanced):<12.3f} {roc_auc_score(y_test, y_pred_proba_balanced) - roc_auc_score(y_test, y_pred_proba_sensitive):<12.3f}")
print(f"{'Precision':<15} {precision_score(y_test, y_pred_sensitive, zero_division=0):<12.3f} {precision_score(y_test, y_pred_balanced, zero_division=0):<12.3f} {precision_score(y_test, y_pred_balanced, zero_division=0) - precision_score(y_test, y_pred_sensitive, zero_division=0):<12.3f}")
print(f"{'Recall':<15} {recall_score(y_test, y_pred_sensitive, zero_division=0):<12.3f} {recall_score(y_test, y_pred_balanced, zero_division=0):<12.3f} {recall_score(y_test, y_pred_balanced, zero_division=0) - recall_score(y_test, y_pred_sensitive, zero_division=0):<12.3f}")

print(f"\n✅ TRAINING COMPLETE!")
print("The final balanced model has been successfully trained, validated, and serialized.")