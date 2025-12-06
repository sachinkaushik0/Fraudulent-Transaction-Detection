"""
Model Explainability Module using SHAP
Provides interpretability for fraud detection predictions
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import joblib


class FraudExplainer:
    def __init__(self, model, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model = model
        self.explainer = None
        self.shap_values = None
    
    def create_explainer(self, X_train):
        """Create SHAP explainer for the model"""
        print("Creating SHAP explainer...")
        print("This may take a few minutes for large datasets...")
        
        # Sample data for faster computation
        sample_size = self.config['shap']['sample_size']
        if len(X_train) > sample_size:
            X_sample = X_train.sample(n=sample_size, random_state=42)
        else:
            X_sample = X_train
        
        # Create TreeExplainer for XGBoost
        self.explainer = shap.TreeExplainer(self.model)
        print("SHAP explainer created successfully!")
        
        return self.explainer
    
    def compute_shap_values(self, X):
        """Compute SHAP values for given data"""
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        print(f"Computing SHAP values for {len(X)} samples...")
        self.shap_values = self.explainer.shap_values(X)
        print("SHAP values computed!")
        
        return self.shap_values
    
    def plot_summary(self, X, shap_values=None):
        """Generate SHAP summary plot"""
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X, 
            max_display=self.config['shap']['max_display'],
            show=False
        )
        plt.tight_layout()
        plt.savefig('outputs/shap_summary.png', dpi=300, bbox_inches='tight')
        print("SHAP summary plot saved to outputs/shap_summary.png")
        plt.close()
    
    def plot_feature_importance(self, X, shap_values=None):
        """Generate SHAP feature importance bar plot"""
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X,
            plot_type="bar",
            max_display=self.config['shap']['max_display'],
            show=False
        )
        plt.tight_layout()
        plt.savefig('outputs/shap_importance.png', dpi=300, bbox_inches='tight')
        print("SHAP feature importance saved to outputs/shap_importance.png")
        plt.close()
    
    def explain_prediction(self, X, index=0):
        """Explain a single prediction using force plot"""
        if self.shap_values is None:
            shap_values = self.compute_shap_values(X.iloc[[index]])
        else:
            shap_values = self.shap_values[index:index+1]
        
        # Create force plot
        shap.force_plot(
            self.explainer.expected_value,
            shap_values,
            X.iloc[index],
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        plt.savefig(f'outputs/shap_force_plot_{index}.png', dpi=300, bbox_inches='tight')
        print(f"Force plot for prediction {index} saved to outputs/shap_force_plot_{index}.png")
        plt.close()
    
    def plot_dependence(self, X, feature, shap_values=None):
        """Plot SHAP dependence for a specific feature"""
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature, shap_values, X,
            show=False
        )
        plt.tight_layout()
        plt.savefig(f'outputs/shap_dependence_{feature}.png', dpi=300, bbox_inches='tight')
        print(f"Dependence plot for {feature} saved to outputs/shap_dependence_{feature}.png")
        plt.close()
    
    def generate_report(self, X, y_true, y_pred, y_proba):
        """Generate comprehensive explainability report"""
        print("="*50)
        print("Generating Explainability Report")
        print("="*50)
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(X)
        
        # Generate plots
        self.plot_summary(X, shap_values)
        self.plot_feature_importance(X, shap_values)
        
        # Explain specific predictions
        # Find a fraud case and a legitimate case
        fraud_indices = np.where(y_true == 1)[0]
        legit_indices = np.where(y_true == 0)[0]
        
        if len(fraud_indices) > 0:
            fraud_idx = fraud_indices[0]
            self.explain_prediction(X, fraud_idx)
            print(f"\nExplained fraud prediction (index {fraud_idx})")
        
        if len(legit_indices) > 0:
            legit_idx = legit_indices[0]
            self.explain_prediction(X, legit_idx)
            print(f"Explained legitimate prediction (index {legit_idx})")
        
        # Top features analysis
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_features = pd.DataFrame({
            'feature': X.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False).head(10)
        
        print("\nTop 10 Most Important Features (by SHAP):")
        print(top_features.to_string(index=False))
        
        # Plot dependence for top features
        for feature in top_features['feature'].head(3):
            self.plot_dependence(X, feature, shap_values)
        
        print("\n" + "="*50)
        print("Explainability report generation complete!")
        print("="*50)


if __name__ == "__main__":
    # Load model and data
    print("Loading model and data...")
    model = joblib.load('models/xgboost_fraud.pkl')
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Create explainer
    explainer = FraudExplainer(model)
    explainer.create_explainer(X_train)
    
    # Generate report
    explainer.generate_report(X_test[:100], y_test[:100], y_pred[:100], y_proba[:100])
