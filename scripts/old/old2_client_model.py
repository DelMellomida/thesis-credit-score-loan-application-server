import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

RAW_DATA_PATH = '../data/raw/synthetic_training_data.csv'  # Updated to use the uploaded file
MODEL_DIR = '../models'

df = pd.read_csv(RAW_DATA_PATH)
print(f"Original dataset shape: {df.shape}")

# Basic data validation
print("\n=== DATA VALIDATION ===")
print("\nColumn names in dataset:")
print(df.columns.tolist())

# Check data types
print("\nData types:")
print(df.dtypes)

# Check for duplicates
duplicates = df.duplicated()
print(f"\nNumber of duplicate rows: {duplicates.sum()}")
if duplicates.sum() > 0:
    df = df.drop_duplicates()
    print(f"Dataset shape after removing duplicates: {df.shape}")

print("\nChecking for missing values in the dataset:")
missing_summary = df.isnull().sum()
missing_percent = (missing_summary / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing_summary,
    'Missing_Percentage': missing_percent
})
print(missing_df[missing_df['Missing_Count'] > 0])

# Filter for new clients
new_clients_df = df[df['Is_Renewing_Client'] == 0].copy()
print(f"\nNew clients dataset shape: {new_clients_df.shape}")

# Check target variable distribution
print("\nTarget variable (Default) distribution:")
print(new_clients_df['Default'].value_counts())
print(f"Default rate: {new_clients_df['Default'].mean():.2%}")

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

# Remove outliers for numerical features using IQR method
numerical_cols = ['Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
                 'Years_at_Current_Address', 'Number_of_Dependents',
                 'Comaker_Employment_Tenure_Months', 'Comaker_Net_Salary_Per_Cutoff']

print("\nOutlier detection for numerical features:")
for col in numerical_cols:
    if col in X.columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
        if outliers > 0:
            print(f"{col}: {outliers} outliers detected")
            # Cap outliers instead of removing them
            X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)

# Validate numerical features
print("\nValidating numerical features:")
for col in numerical_cols:
    if col in X.columns and X[col].notna().any():
        print(f"{col}: min={X[col].min():.2f}, max={X[col].max():.2f}, mean={X[col].mean():.2f}")
        
        # Ensure non-negative values
        if (X[col] < 0).any():
            print(f"  WARNING: Negative values found in {col}, setting to 0")
            X[col] = X[col].clip(lower=0)

print("\nChecking for missing values in selected features:")
print(X.isnull().sum())

# Handle missing values before preprocessing
# For categorical features, fill with 'Unknown' or mode
categorical_features = ['Employment_Sector', 'Salary_Frequency', 'Housing_Status', 
                       'Household_Head', 'Comaker_Relationship', 'Has_Community_Role',
                       'Paluwagan_Participation', 'Other_Income_Source', 'Disaster_Preparedness']

print("\n=== MISSING VALUE IMPUTATION ===")

# First, standardize categorical values (handle case sensitivity)
print("\nStandardizing categorical values...")
for col in categorical_features:
    if col in X.columns and X[col].dtype == 'object':
        # Strip whitespace and standardize case
        X[col] = X[col].str.strip()
        
        # Specific standardizations based on expected values
        if col == 'Housing_Status':
            X[col] = X[col].str.capitalize()  # 'owned' -> 'Owned', 'rented' -> 'Rented'
        elif col in ['Employment_Sector']:
            X[col] = X[col].str.capitalize()  # 'public' -> 'Public', 'private' -> 'Private'
        elif col in ['Household_Head', 'Has_Community_Role', 'Paluwagan_Participation']:
            X[col] = X[col].str.capitalize()  # 'yes' -> 'Yes', 'no' -> 'No'

for col in categorical_features:
    if col in X.columns and X[col].isnull().any():
        missing_count = X[col].isnull().sum()
        # For ordinal features, use a placeholder that maintains order
        if col == 'Comaker_Relationship':
            fill_value = 'Friend'  # Lowest tier
        elif col == 'Other_Income_Source':
            fill_value = 'None'  # Lowest tier
        elif col == 'Disaster_Preparedness':
            fill_value = 'None'  # Lowest tier
        elif col == 'Salary_Frequency':
            fill_value = 'Monthly'  # Lowest tier
        else:
            # For nominal features, use mode
            mode_value = X[col].mode()
            fill_value = mode_value[0] if not mode_value.empty else 'Unknown'
        
        X[col].fillna(fill_value, inplace=True)
        print(f"{col}: Filled {missing_count} missing values with '{fill_value}'")

# For numerical features, use median
for col in numerical_cols:
    if X[col].isnull().any():
        missing_count = X[col].isnull().sum()
        median_value = X[col].median()
        X[col].fillna(median_value, inplace=True)
        print(f"{col}: Filled {missing_count} missing values with median {median_value:.2f}")

print("\nMissing values after imputation:")
print(X.isnull().sum().sum(), "total missing values")

# Apply capped penalty logic
X['Number_of_Dependents'] = np.minimum(X['Number_of_Dependents'], 5)
print(f"\nApplied capping to Number_of_Dependents (max=5)")

# Validate categorical features
print("\n=== CATEGORICAL FEATURE VALIDATION ===")
for col in categorical_features:
    print(f"\n{col} value counts:")
    print(X[col].value_counts())

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"Target distribution: {y.value_counts()}")
print(f"Default rate: {y.mean():.2%}")

ordinal_features = {
    'Salary_Frequency': ['Monthly', 'Bimonthly', 'Biweekly', 'Weekly'],
    'Comaker_Relationship': ['Friend', 'Sibling', 'Parent', 'Spouse'],
    'Other_Income_Source': ['None', 'Freelance', 'Business', 'OFW Remittance'],
    'Disaster_Preparedness': ['None', 'Savings', 'Insurance', 'Community Plan']
}

# Verify that all values in ordinal features are valid
print("\nVerifying ordinal feature values:")
for feature, valid_categories in ordinal_features.items():
    if feature in X.columns:
        unique_values = X[feature].dropna().unique()
        print(f"\n{feature} unique values: {unique_values}")
        invalid_values = [v for v in unique_values if v not in valid_categories]
        if invalid_values:
            print(f"  WARNING: Invalid values found: {invalid_values}")
            print(f"  Expected values: {valid_categories}")

# Check nominal features for their actual values
nominal_features = ['Employment_Sector', 'Housing_Status', 'Household_Head', 
                   'Has_Community_Role', 'Paluwagan_Participation']

print("\nVerifying nominal feature values:")
for feature in nominal_features:
    if feature in X.columns:
        unique_values = X[feature].dropna().unique()
        print(f"\n{feature} unique values: {unique_values}")

numerical_features = ['Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
                     'Years_at_Current_Address', 'Number_of_Dependents',
                     'Comaker_Employment_Tenure_Months', 'Comaker_Net_Salary_Per_Cutoff']

# Create transformers with handle_unknown parameter for safety
ordinal_transformers = []
for feature, categories in ordinal_features.items():
    ordinal_transformers.append(
        (f'ordinal_{feature}', 
         OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=-1), 
         [feature])
    )

# Use OneHotEncoder with handle_unknown='ignore' to avoid errors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('nom', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), nominal_features)
    ] + ordinal_transformers,
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Check if we have enough data for modeling
if len(y_train) < 20:
    print(f"\nWARNING: Very small training set ({len(y_train)} samples). Results may be unreliable.")
    
if y_train.nunique() < 2:
    raise ValueError("Training set has only one class. Cannot train a classifier.")

# Verify employment sector distribution
print("\nEmployment Sector distribution in train set:")
print(X_train['Employment_Sector'].value_counts())
print("\nEmployment Sector distribution in test set:")
print(X_test['Employment_Sector'].value_counts())

# Check if we have both sectors in test set
if len(X_test['Employment_Sector'].unique()) < 2:
    print("\nWARNING: Test set doesn't have both employment sectors. Fairness analysis may be limited.")

def calculate_fairness_metrics(y_true, y_pred, sensitive_feature):
    """Calculate fairness metrics with proper error handling"""
    try:
        public_mask = sensitive_feature == 'Public'
        private_mask = sensitive_feature == 'Private'
        
        # Check if both groups exist
        if not public_mask.any() or not private_mask.any():
            print("WARNING: One of the employment sectors has no samples")
            return {
                'demographic_parity_diff': 0,
                'tpr_diff': 0,
                'fpr_diff': 0,
                'public_positive_rate': 0,
                'private_positive_rate': 0
            }
        
        # Calculate positive rates (non-default predictions)
        public_positive_rate = (y_pred[public_mask] == 0).mean()
        private_positive_rate = (y_pred[private_mask] == 0).mean()
        demographic_parity_diff = abs(public_positive_rate - private_positive_rate)
        
        # Calculate confusion matrices with error handling
        from sklearn.metrics import confusion_matrix
        
        # Get unique classes in each group
        public_classes = np.unique(np.concatenate([y_true[public_mask], y_pred[public_mask]]))
        private_classes = np.unique(np.concatenate([y_true[private_mask], y_pred[private_mask]]))
        
        # Create confusion matrices with all possible labels
        all_labels = [0, 1]
        public_cm = confusion_matrix(y_true[public_mask], y_pred[public_mask], labels=all_labels)
        private_cm = confusion_matrix(y_true[private_mask], y_pred[private_mask], labels=all_labels)
        
        # Calculate TPR and FPR with safety checks
        # TPR = TP / (TP + FN)
        public_tp = public_cm[1, 1]
        public_fn = public_cm[1, 0]
        public_tpr = public_tp / (public_tp + public_fn) if (public_tp + public_fn) > 0 else 0
        
        private_tp = private_cm[1, 1]
        private_fn = private_cm[1, 0]
        private_tpr = private_tp / (private_tp + private_fn) if (private_tp + private_fn) > 0 else 0
        
        # FPR = FP / (FP + TN)
        public_fp = public_cm[0, 1]
        public_tn = public_cm[0, 0]
        public_fpr = public_fp / (public_fp + public_tn) if (public_fp + public_tn) > 0 else 0
        
        private_fp = private_cm[0, 1]
        private_tn = private_cm[0, 0]
        private_fpr = private_fp / (private_fp + private_tn) if (private_fp + private_tn) > 0 else 0
        
        tpr_diff = abs(public_tpr - private_tpr)
        fpr_diff = abs(public_fpr - private_fpr)
        
        print(f"\nDetailed Fairness Analysis:")
        print(f"Public sector - Samples: {public_mask.sum()}, Positive rate: {public_positive_rate:.3f}")
        print(f"Private sector - Samples: {private_mask.sum()}, Positive rate: {private_positive_rate:.3f}")
        print(f"Public confusion matrix:\n{public_cm}")
        print(f"Private confusion matrix:\n{private_cm}")
        
        return {
            'demographic_parity_diff': demographic_parity_diff,
            'tpr_diff': tpr_diff,
            'fpr_diff': fpr_diff,
            'public_positive_rate': public_positive_rate,
            'private_positive_rate': private_positive_rate
        }
    
    except Exception as e:
        print(f"Error in fairness calculation: {str(e)}")
        return {
            'demographic_parity_diff': 0,
            'tpr_diff': 0,
            'fpr_diff': 0,
            'public_positive_rate': 0,
            'private_positive_rate': 0
        }

pipeline_temp = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

X_train_processed = pipeline_temp.named_steps['preprocessor'].fit_transform(X_train)
y_pred_temp = pipeline_temp.fit(X_train, y_train).predict(X_test)

fairness_metrics = calculate_fairness_metrics(
    y_test, y_pred_temp, X_test['Employment_Sector']
)

print("\nFairness Analysis (Before Mitigation):")
print(f"Demographic Parity Difference: {fairness_metrics['demographic_parity_diff']:.3f}")
print(f"TPR Difference: {fairness_metrics['tpr_diff']:.3f}")
print(f"FPR Difference: {fairness_metrics['fpr_diff']:.3f}")

# Check class distribution before proceeding
print(f"\nChecking class distribution in training set:")
print(y_train.value_counts())
minority_class_count = y_train.value_counts().min()
print(f"Minority class has {minority_class_count} samples")

# Adjust SMOTE k_neighbors based on minority class size
# SMOTE needs k_neighbors < minority_class_count
if minority_class_count < 6:
    k_neighbors = min(minority_class_count - 1, 1)  # At least 1, but less than minority count
    print(f"\nAdjusting SMOTE k_neighbors to {k_neighbors} due to small minority class")
else:
    k_neighbors = 5  # Default value

smote = SMOTE(random_state=42, k_neighbors=k_neighbors)

# Adjust cross-validation folds based on minority class size
# Each fold needs at least k_neighbors + 1 samples of minority class
min_samples_per_fold = k_neighbors + 1
max_cv_folds = min(5, minority_class_count // min_samples_per_fold)
cv_folds = max(2, max_cv_folds)  # At least 2 folds, but no more than data allows

print(f"\nUsing {cv_folds}-fold cross-validation (adjusted based on minority class size)")

param_grid = {
    'classifier__penalty': ['l1', 'l2', 'elasticnet'],
    'classifier__C': [0.001, 0.01, 0.1, 1, 10],
    'classifier__solver': ['saga'],
    'classifier__l1_ratio': [0.5]
}

# For elasticnet, only use l1_ratio when penalty='elasticnet'
# Create separate parameter grids
param_grid_l1_l2 = {
    'classifier__penalty': ['l1', 'l2'],
    'classifier__C': [0.001, 0.01, 0.1, 1, 10],
    'classifier__solver': ['saga']
}

param_grid_elasticnet = {
    'classifier__penalty': ['elasticnet'],
    'classifier__C': [0.001, 0.01, 0.1, 1, 10],
    'classifier__solver': ['saga'],
    'classifier__l1_ratio': [0.1, 0.5, 0.9]
}

param_grid_list = [param_grid_l1_l2, param_grid_elasticnet]

pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Use StratifiedKFold to ensure each fold has both classes
from sklearn.model_selection import StratifiedKFold
cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid_list,
    cv=cv_strategy,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    error_score='raise'  # This will help us see if there are still issues
)

print("\nPerforming Grid Search...")
try:
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC score: {grid_search.best_score_:.3f}")
    
    best_model = grid_search.best_estimator_
    
except ValueError as e:
    print(f"\nError during grid search with SMOTE: {str(e)}")
    print("\nFalling back to class weight balancing instead of SMOTE...")
    
    # Alternative approach: Use class weights instead of SMOTE
    from sklearn.utils.class_weight import compute_class_weight
    
    # Compute class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    # Create pipeline without SMOTE
    pipeline_no_smote = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight=class_weight_dict))
    ])
    
    # Simplified parameter grid
    param_grid_simple = {
        'classifier__penalty': ['l2'],
        'classifier__C': [0.001, 0.01, 0.1, 1, 10],
        'classifier__solver': ['lbfgs']
    }
    
    grid_search = GridSearchCV(
        pipeline_no_smote,
        param_grid_simple,
        cv=3,  # Use fewer folds
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters (with class weights): {grid_search.best_params_}")
    print(f"Best CV ROC-AUC score: {grid_search.best_score_:.3f}")
    
    best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("\nModel Performance on Test Set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

classifier = best_model.named_steps['classifier']
preprocessor_fitted = best_model.named_steps['preprocessor']

feature_names = []
if hasattr(preprocessor_fitted, 'get_feature_names_out'):
    feature_names = preprocessor_fitted.get_feature_names_out()
else:
    feature_names = (numerical_features + 
                    [f"{feat}_encoded" for feat in nominal_features] +
                    list(ordinal_features.keys()))

coefficients = classifier.coef_[0]
coef_df = pd.DataFrame({
    'Feature': feature_names[:len(coefficients)],
    'Coefficient': coefficients
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nTop 10 Most Important Features (by absolute coefficient):")
print(coef_df.head(10))

key_features = {
    'Employment_Tenure_Months': None,
    'Paluwagan_Participation': None,
    'Salary_Frequency': None,
    'Comaker_Relationship': None
}

for idx, feature in enumerate(feature_names):
    if 'Employment_Tenure_Months' in str(feature):
        key_features['Employment_Tenure_Months'] = coefficients[idx]
    elif 'Paluwagan_Participation' in str(feature):
        key_features['Paluwagan_Participation'] = coefficients[idx]
    elif 'Salary_Frequency' in str(feature):
        key_features['Salary_Frequency'] = coefficients[idx]
    elif 'Comaker_Relationship' in str(feature):
        key_features['Comaker_Relationship'] = coefficients[idx]

print("\nKey Feature Coefficients Analysis:")
for feature, coef in key_features.items():
    if coef is not None:
        print(f"{feature}: {coef:.4f}")
        if feature == 'Employment_Tenure_Months' and coef < 0:
            print("  ✓ Negative coefficient indicates lower default risk with longer tenure")
        elif feature == 'Paluwagan_Participation' and coef < 0:
            print("  ✓ Negative coefficient indicates participation reduces default risk")

final_pipeline = Pipeline([
    ('preprocessor', preprocessor_fitted),
    ('classifier', classifier)
])

# Create models directory if it doesn't exist
import os
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = f"{MODEL_DIR}/new_client_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(final_pipeline, f)

print(f"\nModel saved successfully to: {model_path}")

print("\nVerifying saved model...")
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

sample_predictions = loaded_model.predict(X_test[:5])
print(f"Sample predictions from loaded model: {sample_predictions}")
print("Model pipeline successfully serialized and verified!")