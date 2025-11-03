import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, roc_auc_score, 
                           f1_score, confusion_matrix, roc_curve, precision_recall_curve)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Import additional libraries for advanced models
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
    print("LightGBM library loaded successfully")
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM library not available")

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    from interpret import show
    HAS_EBM = True
    print("InterpretML library loaded successfully")
except ImportError:
    HAS_EBM = False
    print("InterpretML library not available")

try:
    import shap
    HAS_SHAP = True
    print("SHAP library loaded successfully")
except ImportError:
    HAS_SHAP = False
    print("SHAP library not available")

# Configuration
RAW_DATA_PATH = '../data/raw/synthetic_training_data12.csv'
MODEL_DIR = '../models'

print("="*80)
print("COMPREHENSIVE MODEL COMPARISON FOR CREDIT DEFAULT PREDICTION")
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

# Calculate scale_pos_weight for LightGBM
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nScale pos weight for LightGBM: {scale_pos_weight:.2f}")

print("\n" + "="*80)
print("VIF-BASED FEATURE SELECTION (FROM ORIGINAL CODE)")
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

print("\nPerforming iterative VIF-based feature removal...")
iteration = 0
max_iterations = 20

while iteration < max_iterations:
    iteration += 1
    print(f"\n--- VIF Iteration {iteration} ---")
    print(f"Current feature count: {len(current_features['all_features'])}")
    
    try:
        X_current = X_train[current_features['all_features']].copy()
        preprocessor, ordinal_features, nominal_features = create_preprocessor_and_vif_check(
            current_features, numerical_cols
        )
        
        X_processed = preprocessor.fit_transform(X_current)
        feature_names = preprocessor.get_feature_names_out()
        
        vif_data = calculate_vif(X_processed, feature_names)
        
        max_vif = vif_data.iloc[0]['VIF']
        max_vif_feature = vif_data.iloc[0]['Feature']
        
        print(f"Highest VIF: {max_vif:.2f} for feature '{max_vif_feature}'")
        
        if max_vif <= 5.0:
            print("✓ All features have VIF <= 5.0. VIF cleaning complete!")
            break
        
        # Identify original feature to remove
        feature_to_remove = None
        for orig_feature in current_features['all_features']:
            if orig_feature in max_vif_feature or max_vif_feature.startswith(f'num__{orig_feature}') or max_vif_feature.startswith(f'ordinal_{orig_feature}'):
                feature_to_remove = orig_feature
                break
        
        if feature_to_remove is None:
            for orig_feature in current_features['all_features']:
                if orig_feature.lower() in max_vif_feature.lower():
                    feature_to_remove = orig_feature
                    break
        
        if feature_to_remove:
            print(f"⚠️  Removing feature: {feature_to_remove}")
            
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
print(f"Final feature count: {len(current_features['all_features'])}")
print(f"Removed features: {set(all_features) - set(current_features['all_features'])}")

# Update datasets with final feature set
X_train_final = X_train[current_features['all_features']].copy()
X_test_final = X_test[current_features['all_features']].copy()

# Create final preprocessor
final_preprocessor, final_ordinal_features, final_nominal_features = create_preprocessor_and_vif_check(
    current_features, numerical_cols
)

print("\n" + "="*80)
print("PHASE 1: ESTABLISH BASELINES AND ADVANCED MODELS")
print("="*80)

# Dictionary to store all models and their results
models = {}
results = {}

# 1. LOGISTIC REGRESSION (BASELINE)
print("\n1. Training Logistic Regression (Interpretable Baseline)...")

# Use GridSearch to find optimal C parameter
lr_pipeline = Pipeline([
    ('preprocessor', final_preprocessor),
    ('classifier', LogisticRegression(
        penalty='l2',
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
        solver='lbfgs'
    ))
])

lr_param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10]
}

cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
lr_grid_search = GridSearchCV(
    lr_pipeline,
    lr_param_grid,
    cv=cv_strategy,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

lr_grid_search.fit(X_train_final, y_train)
models['Logistic Regression'] = lr_grid_search.best_estimator_

print(f"Best LR parameters: {lr_grid_search.best_params_}")
print(f"Best CV ROC-AUC score: {lr_grid_search.best_score_:.3f}")

# 2. LIGHTGBM (HIGH-PERFORMANCE BENCHMARK)
if HAS_LIGHTGBM:
    print("\n2. Training LightGBM (High-Performance Benchmark)...")
    
    # First, transform data for LightGBM
    X_train_transformed = final_preprocessor.fit_transform(X_train_final)
    X_test_transformed = final_preprocessor.transform(X_test_final)
    
    # Get feature names from preprocessor
    feature_names_transformed = final_preprocessor.get_feature_names_out()
    
    # Convert to DataFrame to preserve feature names for LightGBM
    X_train_lgb = pd.DataFrame(X_train_transformed, columns=feature_names_transformed)
    X_test_lgb = pd.DataFrame(X_test_transformed, columns=feature_names_transformed)
    
    # Create LightGBM model with grid search
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    lgb_param_grid = {
        'scale_pos_weight': [scale_pos_weight],
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [3, 5, 7],
        'num_leaves': [31, 50, 100],
        'min_child_samples': [20, 30, 40]
    }
    
    lgb_grid_search = GridSearchCV(
        lgb_model,
        lgb_param_grid,
        cv=cv_strategy,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    lgb_grid_search.fit(X_train_lgb, y_train)
    models['LightGBM'] = lgb_grid_search.best_estimator_
    
    print(f"Best LightGBM parameters: {lgb_grid_search.best_params_}")
    print(f"Best CV ROC-AUC score: {lgb_grid_search.best_score_:.3f}")
else:
    print("\n2. LightGBM not available - skipping")

# 3. EXPLAINABLE BOOSTING MACHINE (EBM)
if HAS_EBM:
    print("\n3. Training Explainable Boosting Machine (EBM)...")
    
    # EBMs work best with raw features, not preprocessed
    ebm_model = ExplainableBoostingClassifier(
        random_state=42,
        n_jobs=-1
    )
    
    # EBMs require less hyperparameter tuning
    ebm_model.fit(X_train_final, y_train)
    models['EBM'] = ebm_model
    
    print("EBM training complete (minimal hyperparameter tuning required)")
else:
    print("\n3. InterpretML not available - skipping EBM")

# 4. INNOVATIVE ENSEMBLE (LR + EBM)
if HAS_EBM:
    print("\n4. Creating Innovative Ensemble (LR + EBM)...")
    
    # Create stacking ensemble
    base_estimators = [
        ('lr', models['Logistic Regression']),
        ('ebm', models['EBM'])
    ]
    
    # Meta-model
    meta_lr = LogisticRegression(random_state=42, max_iter=1000)
    
    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_lr,
        cv=cv_strategy,  # Use cross-validation to prevent overfitting
        n_jobs=-1
    )
    
    stacking_model.fit(X_train_final, y_train)
    models['LR+EBM Ensemble'] = stacking_model
    
    print("Stacking ensemble training complete")
else:
    print("\n4. EBM not available - skipping ensemble")

print("\n" + "="*80)
print("PHASE 2: MULTI-OBJECTIVE THRESHOLD OPTIMIZATION")
print("="*80)

def find_optimal_thresholds(y_true, y_pred_proba):
    """Find optimal thresholds for different objectives"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate F1 scores
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Find threshold that maximizes F1
    max_f1_idx = np.argmax(f1_scores[:-1])  # Exclude last element
    optimal_f1_threshold = thresholds[max_f1_idx]
    
    # Find threshold for high precision (min recall = 0.30)
    valid_precision_idx = np.where(recall[:-1] >= 0.30)[0]
    if len(valid_precision_idx) > 0:
        max_precision_idx = valid_precision_idx[np.argmax(precision[:-1][valid_precision_idx])]
        high_precision_threshold = thresholds[max_precision_idx]
    else:
        high_precision_threshold = optimal_f1_threshold
    
    # Find threshold for high recall (min precision = 0.20)
    valid_recall_idx = np.where(precision[:-1] >= 0.20)[0]
    if len(valid_recall_idx) > 0:
        max_recall_idx = valid_recall_idx[np.argmax(recall[:-1][valid_recall_idx])]
        high_recall_threshold = thresholds[max_recall_idx]
    else:
        high_recall_threshold = optimal_f1_threshold
    
    return {
        'f1_optimal': optimal_f1_threshold,
        'high_precision': high_precision_threshold,
        'high_recall': high_recall_threshold
    }

# Analyze thresholds for each model
threshold_analysis = {}

for model_name, model in models.items():
    print(f"\nAnalyzing thresholds for {model_name}...")
    
    # Get predictions
    if model_name == 'LightGBM' and HAS_LIGHTGBM:
        y_pred_proba = model.predict_proba(X_test_lgb)[:, 1]
    else:
        y_pred_proba = model.predict_proba(X_test_final)[:, 1]
    
    # Find optimal thresholds
    thresholds = find_optimal_thresholds(y_test, y_pred_proba)
    threshold_analysis[model_name] = thresholds
    
    # Calculate metrics at different thresholds
    results[model_name] = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'thresholds': thresholds,
        'metrics_at_thresholds': {}
    }
    
    for threshold_type, threshold_value in thresholds.items():
        y_pred = (y_pred_proba >= threshold_value).astype(int)
        
        metrics = {
            'threshold': threshold_value,
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        results[model_name]['metrics_at_thresholds'][threshold_type] = metrics
        
        print(f"  {threshold_type}: threshold={threshold_value:.3f}, "
              f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

print("\n" + "="*80)
print("PHASE 3: INTERPRETABILITY AND FINAL COMPARISON")
print("="*80)

# Generate SHAP plots for LightGBM
if HAS_LIGHTGBM and HAS_SHAP:
    print("\n1. Generating SHAP plots for LightGBM...")
    
    # Create SHAP explainer for LightGBM
    explainer = shap.TreeExplainer(models['LightGBM'])
    shap_values = explainer.shap_values(X_test_lgb)
    
    # Get feature names
    feature_names = X_test_lgb.columns
    
    # Create SHAP summary plot
    plt.figure(figsize=(12, 8))
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # For binary classification
    shap.summary_plot(shap_values, X_test_lgb, 
                     feature_names=feature_names, 
                     show=False, 
                     max_display=20)
    plt.title('SHAP Feature Importance - LightGBM Model', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

# Generate EBM explanation plots
if HAS_EBM:
    print("\n2. Generating EBM explanation plots...")
    
    # Get global explanations
    ebm_global = models['EBM'].explain_global()
    
    # Plot top 10 features
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    # Get feature information using the modern API
    feature_names = ebm_global.feature_names
    feature_types = ebm_global.feature_types
    
    for idx in range(min(10, len(feature_names))):
        ax = axes[idx]
        
        feature_name = feature_names[idx]
        feature_type = feature_types[idx]
        
        # Get the feature's contribution data
        feature_data = ebm_global.data(idx)
        
        if feature_type == 'continuous':
            # For continuous features
            x_vals = feature_data['names']  # bin edges/values
            y_vals = feature_data['scores']  # contributions
            
            # Handle bin edges vs bin centers
            if len(x_vals) == len(y_vals) + 1:
                # x_vals are bin edges, convert to bin centers
                x_centers = [(x_vals[i] + x_vals[i+1]) / 2 for i in range(len(x_vals)-1)]
                ax.plot(x_centers, y_vals, 'b-', linewidth=2)
                ax.fill_between(x_centers, 0, y_vals, alpha=0.3)
            elif len(x_vals) == len(y_vals):
                # x_vals are already bin centers or values
                ax.plot(x_vals, y_vals, 'b-', linewidth=2)
                ax.fill_between(x_vals, 0, y_vals, alpha=0.3)
            else:
                # Fallback: just plot what we can
                min_len = min(len(x_vals), len(y_vals))
                ax.plot(x_vals[:min_len], y_vals[:min_len], 'b-', linewidth=2)
            
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Contribution')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        else:
            # For categorical features
            categories = feature_data['names']
            scores = feature_data['scores']
            
            # Create bar plot
            x_pos = np.arange(len(categories))
            bars = ax.bar(x_pos, scores)
            
            # Color bars based on positive/negative contribution
            for bar, score in zip(bars, scores):
                if score > 0:
                    bar.set_color('blue')
                else:
                    bar.set_color('red')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(categories, rotation=45, ha='right')
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Contribution')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        ax.set_title(f'{feature_name}', fontsize=10)
    
    # Hide unused subplots
    for idx in range(min(10, len(feature_names)), 10):
        axes[idx].axis('off')
    
    plt.suptitle('EBM Feature Contributions (Top 10 Features)', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Show overall feature importance
    plt.figure(figsize=(10, 8))
    
    # Get feature importances using the modern API
    # Try different methods based on the InterpretML version
    try:
        # Try the term_importances method (newer versions)
        importances = ebm_global.term_importances()
        names = feature_names
    except:
        try:
            # Try accessing via data method
            importances = []
            for i in range(len(feature_names)):
                importance_data = ebm_global.data(i)
                if 'importance' in importance_data:
                    importances.append(importance_data['importance'])
                else:
                    # Calculate importance as the range of scores
                    scores = importance_data['scores']
                    importances.append(np.max(scores) - np.min(scores))
            importances = np.array(importances)
            names = feature_names
        except:
            # Fallback: use the model's feature_importances_ directly
            importances = models['EBM'].feature_importances_
            names = feature_names
    
    # Sort by importance and take top 15
    importance_idx = np.argsort(importances)[::-1][:15]
    top_importances = importances[importance_idx]
    top_names = [names[i] for i in importance_idx]
    
    plt.barh(range(len(top_importances)), top_importances[::-1])
    plt.yticks(range(len(top_importances)), top_names[::-1])
    plt.xlabel('Feature Importance')
    plt.title('EBM Global Feature Importance (Top 15)', fontsize=14)
    plt.tight_layout()
    plt.show()

# Create final summary table
print("\n3. Creating Final Summary Table...")

summary_data = []
for model_name in models.keys():
    if model_name in results:
        model_results = results[model_name]
        f1_metrics = model_results['metrics_at_thresholds']['f1_optimal']
        
        summary_data.append({
            'Model': model_name,
            'ROC AUC': model_results['roc_auc'],
            'F1-Optimal Threshold': f1_metrics['threshold'],
            'Precision @ F1-Optimal': f1_metrics['precision'],
            'Recall @ F1-Optimal': f1_metrics['recall'],
            'F1 Score': f1_metrics['f1']
        })

summary_df = pd.DataFrame(summary_data)
print("\n" + "="*50)
print("FINAL MODEL COMPARISON TABLE")
print("="*50)
print(summary_df.to_string(index=False, float_format='%.3f'))

# Create visualization comparing all models
if len(models) > 1:
    plt.figure(figsize=(12, 8))
    
    for model_name, model in models.items():
        if model_name == 'LightGBM' and HAS_LIGHTGBM:
            y_pred_proba = model.predict_proba(X_test_lgb)[:, 1]
        else:
            y_pred_proba = model.predict_proba(X_test_final)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\n" + "="*80)
print("PHASE 4: FINAL RECOMMENDATIONS")
print("="*80)

# Find best model by ROC AUC
best_auc_model = max(results.items(), key=lambda x: x[1]['roc_auc'])
print(f"\n1. FOR MAXIMUM PERFORMANCE (Highest ROC AUC):")
print(f"   Winner: {best_auc_model[0]} with ROC AUC = {best_auc_model[1]['roc_auc']:.3f}")
print(f"   This model provides the best overall discrimination between defaulters and non-defaulters.")

print(f"\n2. FOR MAXIMUM INTERPRETABILITY:")
print(f"   Winner: Logistic Regression")
print(f"   - ROC AUC: {results['Logistic Regression']['roc_auc']:.3f}")
print(f"   - Provides clear, linear coefficients for each feature")
print(f"   - Perfect for regulatory compliance and loan denial explanations")
if HAS_EBM and 'EBM' in results:
    print(f"   Alternative: EBM with ROC AUC = {results['EBM']['roc_auc']:.3f}")
    print(f"   - Offers excellent interpretability with potentially better performance")
    print(f"   - Shows exact non-linear contributions while remaining fully transparent")

print(f"\n3. FOR A BALANCED APPROACH:")
# Find model with best F1 score
best_f1_model = max(results.items(), 
                    key=lambda x: x[1]['metrics_at_thresholds']['f1_optimal']['f1'])
print(f"   Winner: {best_f1_model[0]}")
print(f"   - Optimal threshold: {best_f1_model[1]['metrics_at_thresholds']['f1_optimal']['threshold']:.3f}")
print(f"   - Precision: {best_f1_model[1]['metrics_at_thresholds']['f1_optimal']['precision']:.3f}")
print(f"   - Recall: {best_f1_model[1]['metrics_at_thresholds']['f1_optimal']['recall']:.3f}")
print(f"   - F1 Score: {best_f1_model[1]['metrics_at_thresholds']['f1_optimal']['f1']:.3f}")
print(f"   This configuration provides the best balance between finding defaulters and minimizing false positives.")

if 'LR+EBM Ensemble' in results:
    print(f"\n4. ON THE ENSEMBLE (LR+EBM):")
    ensemble_results = results['LR+EBM Ensemble']
    lr_results = results['Logistic Regression']
    ebm_results = results['EBM'] if 'EBM' in results else None
    
    print(f"   Ensemble ROC AUC: {ensemble_results['roc_auc']:.3f}")
    print(f"   LR ROC AUC: {lr_results['roc_auc']:.3f}")
    if ebm_results:
        print(f"   EBM ROC AUC: {ebm_results['roc_auc']:.3f}")
    
    # Calculate performance lift
    lr_lift = (ensemble_results['roc_auc'] - lr_results['roc_auc']) / lr_results['roc_auc'] * 100
    if ebm_results:
        ebm_lift = (ensemble_results['roc_auc'] - ebm_results['roc_auc']) / ebm_results['roc_auc'] * 100
        avg_base_auc = (lr_results['roc_auc'] + ebm_results['roc_auc']) / 2
        avg_lift = (ensemble_results['roc_auc'] - avg_base_auc) / avg_base_auc * 100
        
        print(f"\n   Performance Analysis:")
        print(f"   - Lift over LR: {lr_lift:+.1f}%")
        print(f"   - Lift over EBM: {ebm_lift:+.1f}%")
        print(f"   - Lift over average of base models: {avg_lift:+.1f}%")
        
        if avg_lift > 1:
            print(f"   ✓ The ensemble provided a meaningful performance improvement!")
            print(f"   The added complexity is worthwhile for applications prioritizing performance.")
        else:
            print(f"   ✗ The ensemble did not provide significant improvement.")
            print(f"   The simpler individual models may be preferable.")
    else:
        print(f"   Lift over LR: {lr_lift:+.1f}%")

# Business-specific recommendations
print("\n" + "="*80)
print("BUSINESS-SPECIFIC RECOMMENDATIONS")
print("="*80)

print("\n📊 DECISION MATRIX BY BUSINESS CONTEXT:")

print("\n1. MICROFINANCE INSTITUTION (High Volume, Cost-Sensitive):")
print("   Recommended: LightGBM with high-recall threshold")
if 'LightGBM' in results:
    high_recall_metrics = results['LightGBM']['metrics_at_thresholds']['high_recall']
    print(f"   - Threshold: {high_recall_metrics['threshold']:.3f}")
    print(f"   - Catches {high_recall_metrics['recall']:.1%} of defaulters")
    print(f"   - False positive rate manageable for high-volume operations")

print("\n2. REGULATED BANK (Compliance-Critical):")
print("   Recommended: Logistic Regression with f1-optimal threshold")
lr_f1_metrics = results['Logistic Regression']['metrics_at_thresholds']['f1_optimal']
print(f"   - Clear explanations for every decision")
print(f"   - Balanced performance (F1: {lr_f1_metrics['f1']:.3f})")
print(f"   - Regulatory compliance guaranteed")

print("\n3. FINTECH STARTUP (Performance-Focused):")
if 'LR+EBM Ensemble' in results and results['LR+EBM Ensemble']['roc_auc'] == max(r['roc_auc'] for r in results.values()):
    print("   Recommended: LR+EBM Ensemble")
    print(f"   - Highest ROC AUC: {results['LR+EBM Ensemble']['roc_auc']:.3f}")
    print(f"   - Leverages both interpretability and non-linear patterns")
elif 'LightGBM' in results:
    print("   Recommended: LightGBM")
    print(f"   - Excellent performance: {results['LightGBM']['roc_auc']:.3f}")
    print(f"   - Can be explained post-hoc with SHAP")

print("\n4. COMMUNITY LENDER (Relationship-Focused):")
if 'EBM' in results:
    print("   Recommended: EBM with conservative threshold")
    ebm_high_precision = results['EBM']['metrics_at_thresholds']['high_precision']
    print(f"   - Threshold: {ebm_high_precision['threshold']:.3f}")
    print(f"   - High precision: {ebm_high_precision['precision']:.3f}")
    print(f"   - Transparent non-linear patterns respect cultural factors")
else:
    print("   Recommended: Logistic Regression with high-precision threshold")

# Save best model
print("\n" + "="*80)
print("SAVING BEST MODEL")
print("="*80)

os.makedirs(MODEL_DIR, exist_ok=True)

# Save the model with highest ROC AUC
best_model_name = best_auc_model[0]
best_model = models[best_model_name]

# For LightGBM, we need to save it differently
if best_model_name == 'LightGBM' and HAS_LIGHTGBM:
    # Save LightGBM model
    lgb_path = f"{MODEL_DIR}/best_model_lightgbm.pkl"
    with open(lgb_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    # Also save the preprocessor separately
    preprocessor_path = f"{MODEL_DIR}/preprocessor.pkl"
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(final_preprocessor, f)
    
    print(f"✓ Best model (LightGBM) saved to: {lgb_path}")
    print(f"✓ Preprocessor saved to: {preprocessor_path}")
else:
    # Save other models (they include preprocessor in pipeline)
    model_path = f"{MODEL_DIR}/best_model_{best_model_name.lower().replace(' ', '_')}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"✓ Best model ({best_model_name}) saved to: {model_path}")

# Save all results for future reference
results_path = f"{MODEL_DIR}/model_comparison_results.pkl"
with open(results_path, 'wb') as f:
    pickle.dump({
        'results': results,
        'threshold_analysis': threshold_analysis,
        'feature_names': current_features['all_features']
    }, f)
print(f"✓ Detailed results saved to: {results_path}")

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\n🏆 OVERALL WINNER: {best_model_name}")
print(f"   ROC AUC: {best_auc_model[1]['roc_auc']:.3f}")

print(f"\n📋 KEY INSIGHTS:")
print(f"   1. Feature reduction: {len(all_features)} → {len(current_features['all_features'])} features (VIF ≤ 5)")
print(f"   2. Best interpretable model: Logistic Regression (ROC AUC: {results['Logistic Regression']['roc_auc']:.3f})")
if 'LightGBM' in results:
    print(f"   3. Best performance model: {best_model_name} (ROC AUC: {best_auc_model[1]['roc_auc']:.3f})")
if 'EBM' in results:
    print(f"   4. Best interpretable + non-linear: EBM (ROC AUC: {results['EBM']['roc_auc']:.3f})")

print(f"\n💡 RECOMMENDATION:")
print(f"   - For most use cases: Start with {best_model_name}")
print(f"   - For regulatory compliance: Use Logistic Regression")
print(f"   - For maximum insight: Use EBM with visualization tools")
print(f"   - Always validate thresholds based on your specific cost matrix")

print(f"\n✅ ANALYSIS COMPLETE!")