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
import json
import warnings
warnings.filterwarnings('ignore')

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