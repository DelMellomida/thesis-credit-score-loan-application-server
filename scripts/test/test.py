import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
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

# Load and prepare data
RAW_DATA_PATH = '../data/raw/synthetic_training_data4.csv'

print("=== LOADING AND PREPROCESSING DATA ===")
df = pd.read_csv(RAW_DATA_PATH)
print(f"Original dataset shape: {df.shape}")

# Filter for new clients
new_clients_df = df[df['Is_Renewing_Client'] == 0].copy()
print(f"New clients dataset shape: {new_clients_df.shape}")

# --- FIX 1: Corrected Feature Lists (Removed Paluwagan_Participation) ---
financial_features = [
    'Employment_Sector', 'Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
    'Housing_Status', 'Years_at_Current_Address',
    'Number_of_Dependents', 'Comaker_Employment_Tenure_Months',
    'Comaker_Net_Salary_Per_Cutoff', 'Other_Income_Source'
]

cultural_features = [
    'Household_Head',
    'Has_Community_Role',
    # 'Paluwagan_Participation' REMOVED
    'Disaster_Preparedness'
]

all_features = financial_features + cultural_features
X = new_clients_df[all_features].copy()
y = new_clients_df['Default']

# Data cleaning lists
numerical_cols = ['Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
                 'Years_at_Current_Address', 'Number_of_Dependents',
                 'Comaker_Employment_Tenure_Months', 'Comaker_Net_Salary_Per_Cutoff']

categorical_features = ['Employment_Sector', 'Housing_Status',
                       'Household_Head',
                       'Has_Community_Role',
                       # 'Paluwagan_Participation' REMOVED
                       'Other_Income_Source', 'Disaster_Preparedness']

# Basic cleaning and imputation
for col in categorical_features:
    if col in X.columns and X[col].dtype == 'object':
        X[col] = X[col].str.strip().str.capitalize()

for col in categorical_features:
    if col in X.columns and X[col].isnull().any():
        if col == 'Other_Income_Source':
            fill_value = 'None'
        elif col == 'Disaster_Preparedness':
            fill_value = 'None'
        else:
            mode_value = X[col].mode()
            fill_value = mode_value[0] if not mode_value.empty else 'Unknown'
        X[col].fillna(fill_value, inplace=True)

for col in numerical_cols:
    if col in X.columns and X[col].isnull().any():
        X[col].fillna(X[col].median(), inplace=True)

# Cap dependents
X['Number_of_Dependents'] = np.minimum(X['Number_of_Dependents'], 5)

ordinal_features = {
    'Other_Income_Source': ['None', 'Freelance', 'Business', 'OFW Remittance'],
    'Disaster_Preparedness': ['None', 'Savings', 'Insurance', 'Community Plan']
}

nominal_features = ['Employment_Sector', 'Housing_Status', 'Household_Head',
                   'Has_Community_Role'] # 'Paluwagan_Participation' REMOVED

# Create ordinal transformers
ordinal_transformers = []
for feature, categories in ordinal_features.items():
    if feature in X.columns:
        ordinal_transformers.append(
            (f'ordinal_{feature}',
             OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=-1),
             [feature])
        )

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('nom', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), nominal_features)
    ] + ordinal_transformers,
    remainder='passthrough'
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Train default rate: {y_train.mean():.2%}")
print(f"Test default rate: {y_test.mean():.2%}")

print("\n" + "="*80)
print("PHASE 1: FORCING SENSITIVITY - THE 'WHAT IF' ANALYSIS")
print("="*80)

# Create a temporary pipeline with weak regularization
print("\n1. Creating 'Forced-Sensitive' Model with Weak Regularization...")
forced_sensitive_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        penalty='l2',
        C=10.0,
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    ))
])

print("Training the forced-sensitive model...")
forced_sensitive_pipeline.fit(X_train, y_train)

# Evaluate the forced-sensitive model
y_pred_sensitive = forced_sensitive_pipeline.predict(X_test)
y_pred_proba_sensitive = forced_sensitive_pipeline.predict_proba(X_test)[:, 1]

print("\nForced-Sensitive Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_sensitive):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_sensitive, zero_division=0):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_sensitive, zero_division=0):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred_sensitive, zero_division=0):.3f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_sensitive):.3f}")

print("\n" + "="*80)
print("PHASE 2: ANALYZE THE SENSITIVE MODEL'S BEHAVIOR")
print("="*80)

# 1. Check for Multicollinearity using VIF
print("\n1. Checking for Multicollinearity (VIF Analysis)...")
X_train_processed = forced_sensitive_pipeline.named_steps['preprocessor'].transform(X_train)
processed_feature_names = forced_sensitive_pipeline.named_steps['preprocessor'].get_feature_names_out()
X_train_processed_df = pd.DataFrame(X_train_processed, columns=processed_feature_names)

vif_data = pd.DataFrame()
vif_data["Feature"] = X_train_processed_df.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_processed_df.values, i) for i in range(X_train_processed_df.shape[1])]
vif_data = vif_data.sort_values('VIF', ascending=False)

print("\nVariance Inflation Factor (VIF) Analysis:")
print(vif_data)

high_vif_features = vif_data[vif_data['VIF'] > 5]
if len(high_vif_features) > 0:
    print(f"\n⚠️  Features with VIF > 5 (potential multicollinearity):")
    for _, row in high_vif_features.iterrows():
        print(f"   {row['Feature']}: {row['VIF']:.2f}")
else:
    print("\n✓ No significant multicollinearity detected (all VIF < 5)")

# 2. Analyze Coefficients
print("\n2. Analyzing Coefficients from Forced-Sensitive Model...")
classifier = forced_sensitive_pipeline.named_steps['classifier']
coefficients = classifier.coef_[0]
coef_df = pd.DataFrame({
    'Feature': processed_feature_names,
    'Coefficient': coefficients
}).sort_values('Coefficient', key=abs, ascending=False)
print("\nTop 15 Most Important Features (by absolute coefficient):")
print(coef_df.head(15).to_string(index=False))

# 3. Checking Expected Modeling Logic
print("\n3. Checking Expected Modeling Logic...")
expected_logic = {
    'Employment_Tenure_Months': 'negative',
    'Net_Salary_Per_Cutoff': 'negative',
    'Number_of_Dependents': 'positive',
    'Years_at_Current_Address': 'negative'
}
print("\nKey Feature Logic Check:")
for feature_base, expected_sign in expected_logic.items():
    matching_features = [f for f in coef_df['Feature'] if feature_base in str(f)]
    for feature in matching_features:
        coef_value = coef_df[coef_df['Feature'] == feature]['Coefficient'].iloc[0]
        actual_sign = 'negative' if coef_value < 0 else 'positive'
        status = "✓" if actual_sign == expected_sign else "⚠️"
        print(f"  {status} {feature}: {coef_value:.4f} (expected {expected_sign}, got {actual_sign})")

# 4. SHAP Analysis
print("\n4. SHAP Analysis on Forced-Sensitive Model...")
# --- FIX 2: Final Corrected SHAP Block using LinearExplainer ---
if HAS_SHAP:
    try:
        print("Initializing SHAP LinearExplainer...")
        # LinearExplainer is the correct, most efficient tool for logistic regression
        explainer = shap.LinearExplainer(
            forced_sensitive_pipeline.named_steps['classifier'],
            X_train_processed,
            feature_perturbation="interventional"
        )
        
        print("Calculating SHAP values...")
        X_test_processed = forced_sensitive_pipeline.named_steps['preprocessor'].transform(X_test)
        shap_values_sensitive = explainer.shap_values(X_test_processed)

        print("Generating SHAP summary plot...")
        plt.figure()
        shap.summary_plot(shap_values_sensitive, X_test_processed, feature_names=processed_feature_names, show=False)
        plt.title('SHAP Summary Plot - Forced-Sensitive Model')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"SHAP analysis failed: {str(e)}")
else:
    print("SHAP not available - skipping SHAP analysis")


print("\n" + "="*80)
print("PHASE 3: FINDING THE OPTIMAL BALANCE")
print("="*80)

# Create the balanced pipeline
balanced_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Modified parameter grid
param_grid_balanced = {
    'classifier__penalty': ['l2'],
    'classifier__C': [0.1, 1, 10, 50, 100],
    'classifier__solver': ['lbfgs'],
    'classifier__class_weight': ['balanced']
}

print("3. Training Models with Controlled Regularization...")
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search_balanced = GridSearchCV(
    balanced_pipeline,
    param_grid_balanced,
    cv=cv_strategy,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_search_balanced.fit(X_train, y_train)
print(f"\nBest parameters: {grid_search_balanced.best_params_}")
print(f"Best CV ROC-AUC score: {grid_search_balanced.best_score_:.3f}")

best_balanced_model = grid_search_balanced.best_estimator_
y_pred_balanced = best_balanced_model.predict(X_test)
y_pred_proba_balanced = best_balanced_model.predict_proba(X_test)[:, 1]

print("\n4. Best Balanced Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_balanced):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_balanced, zero_division=0):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_balanced, zero_division=0):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred_balanced, zero_division=0):.3f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_balanced):.3f}")

# Final SHAP verification on balanced model
print("\n6. Final SHAP Verification on Balanced Model...")
# --- FIX 3: Final Corrected SHAP Block 2 using LinearExplainer ---
if HAS_SHAP:
    try:
        print("Initializing SHAP LinearExplainer for balanced model...")
        explainer_balanced = shap.LinearExplainer(
            best_balanced_model.named_steps['classifier'],
            X_train_processed,
            feature_perturbation="interventional"
        )
        
        print("Calculating SHAP values for balanced model...")
        shap_values_balanced = explainer_balanced.shap_values(X_test_processed)

        print("Generating side-by-side SHAP comparison plot...")
        plt.figure(figsize=(14, 7))
        
        plt.subplot(1, 2, 1)
        shap.summary_plot(shap_values_sensitive, X_test_processed, feature_names=processed_feature_names, show=False)
        plt.title(f"Forced-Sensitive Model (C=10.0)")
        
        plt.subplot(1, 2, 2)
        shap.summary_plot(shap_values_balanced, X_test_processed, feature_names=processed_feature_names, show=False)
        plt.title(f"Balanced Model (C={best_balanced_model.named_steps['classifier'].C})")
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Final SHAP analysis failed: {str(e)}")


# Performance comparison summary
print("\n" + "="*80)
print("FINAL SUMMARY AND RECOMMENDATIONS")
print("="*80)
print("\nModel Performance Comparison:")
print(f"{'Metric':<15} {'Sensitive':<12} {'Balanced':<12} {'Difference':<12}")
print("-" * 55)
print(f"{'ROC AUC':<15} {roc_auc_score(y_test, y_pred_proba_sensitive):<12.3f} {roc_auc_score(y_test, y_pred_proba_balanced):<12.3f} {roc_auc_score(y_test, y_pred_proba_balanced) - roc_auc_score(y_test, y_pred_proba_sensitive):<12.3f}")
print(f"{'Precision':<15} {precision_score(y_test, y_pred_sensitive, zero_division=0):<12.3f} {precision_score(y_test, y_pred_balanced, zero_division=0):<12.3f} {precision_score(y_test, y_pred_balanced, zero_division=0) - precision_score(y_test, y_pred_sensitive, zero_division=0):<12.3f}")
print(f"{'Recall':<15} {recall_score(y_test, y_pred_sensitive, zero_division=0):<12.3f} {recall_score(y_test, y_pred_balanced, zero_division=0):<12.3f} {recall_score(y_test, y_pred_balanced, zero_division=0) - recall_score(y_test, y_pred_sensitive, zero_division=0):<12.3f}")
print(f"\nOptimal Regularization Parameter: C = {grid_search_balanced.best_params_['classifier__C']}")