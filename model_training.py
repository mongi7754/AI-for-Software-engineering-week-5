"""
Model Training Module for Patient Readmission Prediction
Implements and compares multiple models with hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Handles model training, hyperparameter tuning, and model selection
    Focuses on both performance and interpretability for healthcare applications
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
    def split_data(self, X, y, test_size=0.2, val_size=0.2):
        """
        Split data into training, validation, and test sets
        Following best practices for model evaluation
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: separate validation set from temporary set
        val_size_adj = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adj, random_state=self.random_state, stratify=y_temp
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def initialize_models(self):
        """
        Initialize multiple models for comparison
        Includes both interpretable and high-performance models
        """
        models = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'xgboost': {
                'model': XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0]
                }
            }
        }
        
        return models
    
    def train_models(self, X_train, y_train, X_val, y_val, cv=5):
        """
        Train multiple models with hyperparameter tuning
        Uses cross-validation to select best parameters
        """
        models = self.initialize_models()
        results = {}
        
        for name, model_info in models.items():
            print(f"\n{'='*50}")
            print(f"Training {name}...")
            print(f"{'='*50}")
            
            model = model_info['model']
            params = model_info['params']
            
            # Perform grid search with cross-validation
            grid_search = GridSearchCV(
                model, params, cv=cv, scoring='roc_auc', 
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Store results
            self.models[name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            # Validate on validation set
            val_score = grid_search.best_estimator_.score(X_val, y_val)
            self.models[name]['val_score'] = val_score
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score (AUC): {grid_search.best_score_:.4f}")
            print(f"Validation score (Accuracy): {val_score:.4f}")
            
            # Update best model
            if grid_search.best_score_ > self.best_score:
                self.best_score = grid_search.best_score_
                self.best_model = grid_search.best_estimator_
                self.best_model_name = name
        
        print(f"\nBest model: {self.best_model_name} with AUC: {self.best_score:.4f}")
        
        return self.models
    
    def evaluate_model(self, model, X_test, y_test, feature_names=None):
        """
        Comprehensive model evaluation
        Includes multiple metrics and confusion matrix
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Print results
        print(f"\n{'='*50}")
        print("MODEL EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc_roc:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"True Negatives:  {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives:  {cm[1,1]}")
        
        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_') and feature_names is not None:
            print(f"\nTop 10 Most Important Features:")
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(feature_importance.head(10))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': cm
        }
    
    def save_model(self, model, filepath):
        """Save trained model for deployment"""
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load saved model"""
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model

# Example usage
if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    # Demo the training pipeline
    preprocessor = DataPreprocessor()
    df = preprocessor.generate_synthetic_data(1000)
    df_clean = preprocessor.handle_missing_values(df)
    df_engineered = preprocessor.feature_engineering(df_clean)
    X, y, preprocessor_obj = preprocessor.prepare_features(df_engineered)
    
    # Split data
    trainer = ModelTrainer()
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y)
    
    # Train models
    models = trainer.train_models(X_train, y_train, X_val, y_val)
    
    # Evaluate best model
    best_model = trainer.best_model
    evaluation_results = trainer.evaluate_model(
        best_model, X_test, y_test, preprocessor_obj.feature_names
    )
    
    # Save model
    trainer.save_model(best_model, '../models/readmission_model.pkl')
