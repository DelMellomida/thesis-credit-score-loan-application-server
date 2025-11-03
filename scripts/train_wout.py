import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os
import sys

# --- Configuration ---
# Define file paths to ensure the script can find the data and save the models
DATA_DIR = '../data'
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'synthetic_trainingwout_data.csv')
MODEL_DIR = '../models'
MODEL_PATH = os.path.join(MODEL_DIR, 'credit_model.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'encoder.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
POLY_PATH = os.path.join(MODEL_DIR, 'polynomial_features.pkl')
SELECTOR_PATH = os.path.join(MODEL_DIR, 'feature_selector.pkl')


def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [DATA_DIR, os.path.join(DATA_DIR, 'raw'), MODEL_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory ensured: {directory}")


def validate_data(df):
    """
    Validate the loaded data for basic quality checks and apply data quality rules:
    1. Remove samples with more than 50% missing values
    2. Remove features with more than 95% missing values
    3. Fill remaining missing values with mean (continuous) or mode (categorical)
    """
    print(f"Initial data shape: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    
    # Check if target column exists
    if 'Default' not in df.columns:
        raise ValueError("Target column 'Default' not found in the dataset")
    
    # Handle missing values in target column first
    target_missing = df['Default'].isnull().sum()
    if target_missing > 0:
        print(f"Warning: Found {target_missing} missing values in target column 'Default'")
        print("Removing rows with missing target values...")
        df = df.dropna(subset=['Default'])
        print(f"Data shape after removing missing targets: {df.shape}")
    
    # Check target distribution
    print(f"Target distribution:\n{df['Default'].value_counts()}")
    
    # Rule 1: Remove samples with more than 50% missing values
    print("\n--- Rule 1: Removing samples with >50% missing values ---")
    missing_per_row = df.isnull().sum(axis=1)
    total_features = len(df.columns)
    threshold_50_percent = total_features * 0.5
    
    rows_to_remove = missing_per_row > threshold_50_percent
    rows_removed_count = rows_to_remove.sum()
    
    if rows_removed_count > 0:
        print(f"Removing {rows_removed_count} samples with more than 50% missing values")
        df = df[~rows_to_remove]
        print(f"Data shape after removing samples: {df.shape}")
    else:
        print("No samples have more than 50% missing values")
    
    # Rule 2: Remove features with more than 95% missing values
    print("\n--- Rule 2: Removing features with >95% missing values ---")
    missing_per_feature = df.isnull().sum() / len(df)
    features_to_remove = missing_per_feature[missing_per_feature > 0.95].index.tolist()
    
    # Don't remove the target column even if it has high missing values
    features_to_remove = [f for f in features_to_remove if f != 'Default']
    
    if features_to_remove:
        print(f"Removing {len(features_to_remove)} features with >95% missing values:")
        for feature in features_to_remove:
            print(f"  - {feature}: {missing_per_feature[feature]:.2%} missing")
        df = df.drop(columns=features_to_remove)
        print(f"Data shape after removing features: {df.shape}")
    else:
        print("No features have more than 95% missing values")
    
    # Rule 3: Fill remaining missing values with appropriate statistics
    print("\n--- Rule 3: Filling remaining missing values ---")
    missing_counts = df.isnull().sum()
    features_with_missing = missing_counts[missing_counts > 0].index.tolist()
    
    if features_with_missing:
        print(f"Filling missing values for {len(features_with_missing)} features:")
        for col in features_with_missing:
            if col != 'Default':  # Skip target column
                missing_count = missing_counts[col]
                missing_percent = (missing_count / len(df)) * 100
                
                if df[col].dtype in ['object', 'category']:
                    # For categorical columns, fill with mode (most frequent value)
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col].fillna(mode_val[0], inplace=True)
                        print(f"  - {col}: {missing_count} ({missing_percent:.1f}%) filled with mode '{mode_val[0]}'")
                    else:
                        df[col].fillna('Unknown', inplace=True)
                        print(f"  - {col}: {missing_count} ({missing_percent:.1f}%) filled with 'Unknown'")
                else:
                    # For numerical columns, fill with mean
                    mean_val = df[col].mean()
                    df[col].fillna(mean_val, inplace=True)
                    print(f"  - {col}: {missing_count} ({missing_percent:.1f}%) filled with mean {mean_val:.4f}")
    
    # Final validation
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"\nWarning: {remaining_missing} missing values remain after preprocessing")
        print("Missing values by column:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    else:
        print("\nAll missing values have been handled successfully.")
    
    print(f"\nFinal data shape: {df.shape}")
    return df


def validate_features(df, categorical_features, numerical_features):
    """
    Validate that specified features exist in the dataset.
    Note: Some features may have been removed during data quality checks.
    """
    available_features = set(df.columns) - {'Default'}  # Exclude target column
    
    missing_categorical = [f for f in categorical_features if f not in available_features]
    missing_numerical = [f for f in numerical_features if f not in available_features]
    
    if missing_categorical:
        print(f"Warning: Missing categorical features (may have been removed): {missing_categorical}")
        categorical_features = [f for f in categorical_features if f in available_features]
    
    if missing_numerical:
        print(f"Warning: Missing numerical features (may have been removed): {missing_numerical}")
        numerical_features = [f for f in numerical_features if f in available_features]
    
    print(f"Using {len(categorical_features)} categorical and {len(numerical_features)} numerical features")
    print(f"Available features: {sorted(list(available_features))}")
    return categorical_features, numerical_features


def train_model():
    """
    Enhanced model training pipeline with:
    1. Manual preprocessing with OneHotEncoder and StandardScaler
    2. Polynomial feature generation with interactions
    3. Feature selection using L2 regularized logistic regression
    4. Hyperparameter tuning with GridSearchCV
    5. Final model training with optimized parameters
    
    Returns:
    - model_components dict: Contains all trained components for testing
    - None: If training fails
    """
    print("--- Starting Enhanced Model Training (No Cultural Context) ---")

    # 1. Create Directories
    create_directories()

    # 2. Load Data
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"Successfully loaded data from {RAW_DATA_PATH}")
    except FileNotFoundError:
        print(f"Error: The file {RAW_DATA_PATH} was not found.")
        print("Please ensure you have created the synthetic dataset and placed it in the correct directory.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

    # 3. Validate Data
    try:
        df = validate_data(df)
    except Exception as e:
        print(f"Data validation failed: {str(e)}")
        return None

    # 4. Define Features (Updated to match actual dataset columns)
    # Define features (X) and target (y)
    X = df.drop('Default', axis=1)
    y = df['Default']

    # Define categorical and numerical features based on actual dataset
    categorical_features = [
        'Salary_Frequency',    # Assuming this is categorical (e.g., 'Weekly', 'Bi-weekly', 'Monthly')
        'Household_Head'       # Assuming this is categorical (e.g., 'Yes', 'No')
    ]
    
    numerical_features = [
        'Employment_Tenure_Months',
        'Net_Salary_Per_Cutoff',
        'Years_at_Current_Address',
        'Number_of_Dependents',
        'Comaker_Employment_Tenure_Months',
        'Comaker_Net_Salary_Per_Cutoff',
        'Is_Renewing_Client',
        'Grace_Period_Usage_Rate',
        'Late_Payment_Count',
        'Had_Special_Consideration'
    ]

    # Validate that features exist in the dataset
    categorical_features, numerical_features = validate_features(df, categorical_features, numerical_features)

    if not categorical_features and not numerical_features:
        print("Error: No valid features found in the dataset")
        return None

    print(f"\nFeature Summary:")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")

    # 5. Split Data
    try:
        # Final check for any remaining NaN values before splitting
        if X.isnull().any().any():
            print("Warning: NaN values found in features. Performing additional cleanup...")
            X = X.fillna(0)  # Fill any remaining NaN with 0 as fallback
        
        if y.isnull().any():
            print("Error: NaN values still present in target after preprocessing")
            return None
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"Data split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows) sets.")
    except Exception as e:
        print(f"Error splitting data: {str(e)}")
        return None

    # 6. Manual Preprocessing
    print("\n--- Manual Preprocessing ---")
    
    # One-Hot Encoding for categorical features
    if categorical_features:
        print("Applying One-Hot Encoding to categorical features...")
        ohe = OneHotEncoder(handle_unknown='ignore', drop='first')
        ohe.fit(X_train[categorical_features])
        X_train_categorical = ohe.transform(X_train[categorical_features]).toarray()
        X_test_categorical = ohe.transform(X_test[categorical_features]).toarray()
        print(f"Categorical features shape after OHE: {X_train_categorical.shape}")
        
        # Print feature names for debugging
        if hasattr(ohe, 'get_feature_names_out'):
            cat_feature_names = ohe.get_feature_names_out(categorical_features)
            print(f"Categorical feature names: {list(cat_feature_names)}")
    else:
        X_train_categorical = np.array([]).reshape(X_train.shape[0], 0)
        X_test_categorical = np.array([]).reshape(X_test.shape[0], 0)
        ohe = None
        print("No categorical features to encode")

    # Standard Scaling for numerical features
    if numerical_features:
        print("Applying Standard Scaling to numerical features...")
        scaler = StandardScaler()
        X_train_continuous = scaler.fit_transform(X_train[numerical_features])
        X_test_continuous = scaler.transform(X_test[numerical_features])
        print(f"Numerical features shape after scaling: {X_train_continuous.shape}")
        print(f"Numerical feature names: {numerical_features}")
    else:
        X_train_continuous = np.array([]).reshape(X_train.shape[0], 0)
        X_test_continuous = np.array([]).reshape(X_test.shape[0], 0)
        scaler = None
        print("No numerical features to scale")

    # Combine features
    X_train_transformed = np.hstack([X_train_continuous, X_train_categorical])
    X_test_transformed = np.hstack([X_test_continuous, X_test_categorical])
    print(f"Combined features shape: {X_train_transformed.shape}")

    # 7. Polynomial Features
    print("\n--- Generating Polynomial Features ---")
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_transformed)
    X_test_poly = poly.transform(X_test_transformed)
    print(f"Polynomial features shape: {X_train_poly.shape}")

    # 8. Feature Selection
    print("\n--- Feature Selection ---")
    selector = SelectFromModel(LogisticRegression(penalty='l2', solver='liblinear', max_iter=2000))
    selector.fit(X_train_poly, y_train)
    X_train_sel = selector.transform(X_train_poly)
    X_test_sel = selector.transform(X_test_poly)
    print(f"Selected features shape: {X_train_sel.shape}")
    print(f"Features selected: {X_train_sel.shape[1]} out of {X_train_poly.shape[1]}")

    # 9. Hyperparameter Tuning
    print("\n--- Hyperparameter Tuning ---")
    param_grid = [
        {'penalty': ['l1'], 'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear'], 'class_weight': [None, 'balanced']},
        {'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear'], 'class_weight': [None, 'balanced']},
        {'penalty': ['elasticnet'], 'C': [0.01, 0.1, 1], 'solver': ['saga'], 'class_weight': [None, 'balanced'], 'l1_ratio': [0.25, 0.5, 0.75]}
    ]

    grid_search = GridSearchCV(
        LogisticRegression(max_iter=3000, random_state=42),
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    print("Running GridSearchCV (this may take a while)...")
    grid_search.fit(X_train_sel, y_train)

    best_params = grid_search.best_params_
    print("Best Parameters from GridSearchCV:", best_params)
    print("Best Cross-validation Score:", grid_search.best_score_)

    # 10. Train Final Model
    print("\n--- Training Final Model ---")
    classifier = LogisticRegression(
        random_state=42,
        C=best_params['C'],
        solver=best_params['solver'],
        penalty=best_params['penalty'],
        class_weight=best_params['class_weight'],
        max_iter=5000,
        l1_ratio=best_params['l1_ratio'] if 'l1_ratio' in best_params else None
    )
    classifier.fit(X_train_sel, y_train)

    # 11. Evaluate the Model
    print("\n--- Model Evaluation ---")
    try:
        y_pred = classifier.predict(X_test_sel)
        y_pred_proba = classifier.predict_proba(X_test_sel)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance
        if hasattr(classifier, 'coef_'):
            n_features = len(classifier.coef_[0])
            print(f"\nFinal model uses {n_features} features")
            
            # Get feature importance (absolute values of coefficients)
            feature_importance = np.abs(classifier.coef_[0])
            top_features_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
            print(f"Top 10 feature importance values: {feature_importance[top_features_idx]}")
            
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return False

    # 12. Training Summary (No Model Saving - Testing Only)
    print("\n--- Training Summary ---")
    print("Model training completed successfully!")
    print(f"Final model configuration:")
    print(f"  - Features used: {len(categorical_features) + len(numerical_features)}")
    print(f"  - Categorical: {categorical_features}")
    print(f"  - Numerical: {numerical_features}")
    print(f"  - Best hyperparameters: {best_params}")
    print(f"  - Cross-validation score: {grid_search.best_score_:.4f}")
    print(f"  - Final test metrics: Accuracy={accuracy:.4f}, ROC-AUC={auc:.4f}")
    print("\nNote: Model components not saved (testing mode)")
    
    # Optional: Return model components for further testing
    model_components = {
        'classifier': classifier,
        'encoder': ohe,
        'scaler': scaler,
        'poly': poly,
        'selector': selector,
        'feature_info': {
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'best_params': best_params,
            'test_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': auc
            }
        }
    }
    
    print("Model components available in memory for testing")

    print("\n--- Enhanced Training Script Finished Successfully ---")
    print("Model trained without cultural context features (testing mode - no persistence)")
    return model_components


if __name__ == "__main__":
    model_components = train_model()
    if model_components is None:
        sys.exit(1)
    else:
        print(f"\nTraining completed! Model components available in memory.")
        print(f"Access via: model_components['classifier'], model_components['scaler'], etc.")