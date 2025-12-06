"""
Model Training and Evaluation Module
Implements XGBoost classifier with SMOTE for imbalanced data
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import yaml


class FraudDetectionModel:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model = None
        self.smote = None
    
    def apply_smote(self, X_train, y_train):
        """Apply SMOTE to balance training data"""
        print("Applying SMOTE for class balancing...")
        print(f"Before SMOTE - Class distribution:\n{pd.Series(y_train).value_counts()}")
        
        smote_config = self.config['smote']
        self.smote = SMOTE(
            sampling_strategy=smote_config['sampling_strategy'],
            k_neighbors=smote_config['k_neighbors'],
            random_state=smote_config['random_state']
        )
        
        X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)
        
        print(f"After SMOTE - Class distribution:\n{pd.Series(y_resampled).value_counts()}")
        print(f"Training samples increased from {len(y_train)} to {len(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def train(self, X_train, y_train, use_smote=True):
        """Train XGBoost classifier"""
        print("="*50)
        print("Training XGBoost Model")
        print("="*50)
        
        # Apply SMOTE if specified
        if use_smote:
            X_train, y_train = self.apply_smote(X_train, y_train)
        
        # Get XGBoost parameters from config
        xgb_params = self.config['xgboost']
        
        # Initialize model
        self.model = xgb.XGBClassifier(**xgb_params)
        
        # Train model
        print("\nTraining model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False
        )
        
        print("Training complete!")
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test, threshold=0.5):
        """Comprehensive model evaluation"""
        print("="*50)
        print("Model Evaluation")
        print("="*50)
        
        # Predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Classification metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        print("\nKey Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        return metrics, cm, y_proba
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix - Fraud Detection')
        plt.tight_layout()
        plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Confusion matrix saved to outputs/confusion_matrix.png")
    
    def plot_roc_curve(self, y_test, y_proba):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Fraud Detection')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/roc_curve.png', dpi=300, bbox_inches='tight')
        print("ROC curve saved to outputs/roc_curve.png")
    
    def plot_precision_recall_curve(self, y_test, y_proba):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Fraud Detection')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/precision_recall_curve.png', dpi=300, bbox_inches='tight')
        print("Precision-Recall curve saved to outputs/precision_recall_curve.png")
    
    def plot_feature_importance(self, X_train, top_n=20):
        """Plot feature importance"""
        importance = self.model.feature_importances_
        feature_names = X_train.columns
        
        # Create dataframe and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
        print("Feature importance plot saved to outputs/feature_importance.png")
    
    def save_model(self, filepath=None):
        """Save trained model to disk"""
        if filepath is None:
            filepath = self.config['model']['save_path']
        
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """Load trained model from disk"""
        if filepath is None:
            filepath = self.config['model']['save_path']
        
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self


if __name__ == "__main__":
    # Load processed data
    print("Loading processed data...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    # Train model
    model = FraudDetectionModel()
    model.train(X_train, y_train, use_smote=True)
    
    # Evaluate
    metrics, cm, y_proba = model.evaluate(X_test, y_test)
    
    # Create visualizations
    import os
    os.makedirs('outputs', exist_ok=True)
    
    model.plot_confusion_matrix(cm)
    model.plot_roc_curve(y_test, y_proba)
    model.plot_precision_recall_curve(y_test, y_proba)
    model.plot_feature_importance(X_train)
    
    # Save model
    model.save_model()
