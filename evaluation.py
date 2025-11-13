"""
Model Evaluation and Monitoring Module
Handles comprehensive model assessment and concept drift detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, precision_recall_curve, auc, 
                           classification_report, confusion_matrix)
import json
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation and monitoring
    Includes performance metrics, fairness assessment, and drift detection
    """
    
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.performance_history = []
        
    def comprehensive_evaluation(self, X_test, y_test, feature_names=None):
        """
        Perform comprehensive model evaluation with multiple metrics and visualizations
        """
        # Generate predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Create visualizations
        self._plot_roc_curve(y_test, y_pred_proba, metrics['auc_roc'])
        self._plot_precision_recall_curve(y_test, y_pred_proba)
        self._plot_confusion_matrix(y_test, y_pred)
        
        # Feature importance if available
        if hasattr(self.model, 'feature_importances_') and feature_names is not None:
            self._plot_feature_importance(feature_names)
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive evaluation metrics"""
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                   f1_score, roc_auc_score, average_precision_score)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'average_precision': average_precision_score(y_true, y_pred_proba)
        }
        
        # Calculate specificity and NPV from confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        return metrics
    
    def _plot_roc_curve(self, y_true, y_pred_proba, auc_score):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    
    def _plot_precision_recall_curve(self, y_true, y_pred_proba):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.show()
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Readmitted', 'Readmitted'],
                   yticklabels=['Not Readmitted', 'Readmitted'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
    
    def _plot_feature_importance(self, feature_names, top_n=15):
        """Plot feature importance for tree-based models"""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=True).tail(top_n)
            
            plt.figure(figsize=(10, 8))
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Most Important Features')
            plt.tight_layout()
            plt.show()

class ConceptDriftDetector:
    """
    Monitor and detect concept drift in deployed models
    """
    
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.performance_history = []
        self.data_drift_detected = False
        self.concept_drift_detected = False
    
    def monitor_performance(self, current_metrics, threshold=0.05):
        """
        Monitor model performance for significant drops
        """
        self.performance_history.append(current_metrics)
        
        if len(self.performance_history) > self.window_size:
            # Calculate moving average
            recent_performance = self.performance_history[-self.window_size:]
            historical_performance = self.performance_history[-2*self.window_size:-self.window_size]
            
            if len(historical_performance) == self.window_size:
                recent_auc = np.mean([m['auc_roc'] for m in recent_performance])
                historical_auc = np.mean([m['auc_roc'] for m in historical_performance])
                
                performance_drop = historical_auc - recent_auc
                
                if performance_drop > threshold:
                    print(f"⚠️  Concept drift detected! Performance drop: {performance_drop:.3f}")
                    self.concept_drift_detected = True
                    return True
        
        return False
    
    def detect_data_drift(self, reference_data, current_data, numerical_columns):
        """
        Detect data drift using statistical tests
        """
        from scipy.stats import ks_2samp
        
        drift_detected = False
        
        for col in numerical_columns:
            if col in reference_data.columns and col in current_data.columns:
                stat, p_value = ks_2samp(reference_data[col].dropna(), 
                                       current_data[col].dropna())
                
                if p_value < 0.05:  # Significant difference
                    print(f"Data drift detected in {col} (p-value: {p_value:.4f})")
                    drift_detected = True
        
        self.data_drift_detected = drift_detected
        return drift_detected

# Example usage
if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    from model_training import ModelTrainer
    
    # Generate and prepare data
    preprocessor = DataPreprocessor()
    df = preprocessor.generate_synthetic_data(1000)
    df_clean = preprocessor.handle_missing_values(df)
    df_engineered = preprocessor.feature_engineering(df_clean)
    X, y, preprocessor_obj = preprocessor.prepare_features(df_engineered)
    
    # Split and train model
    trainer = ModelTrainer()
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y)
    models = trainer.train_models(X_train, y_train, X_val, y_val)
    
    # Comprehensive evaluation
    evaluator = ModelEvaluator(trainer.best_model, preprocessor_obj)
    metrics = evaluator.comprehensive_evaluation(X_test, y_test, preprocessor_obj.feature_names)
    
    print("\nFinal Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
