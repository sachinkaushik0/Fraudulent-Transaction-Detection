"""
Main Pipeline Script
Orchestrates the complete fraud detection workflow
"""

import os
import argparse
import pandas as pd
from data_processing import FraudDataProcessor
from model import FraudDetectionModel
from explainer import FraudExplainer


def create_directories():
    """Create necessary directories"""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'outputs'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Directories created successfully!")


def run_pipeline(data_path=None, train=True, evaluate=True, explain=True):
    """Execute the complete fraud detection pipeline"""
    print("\n" + "="*70)
    print(" "*15 + "FRAUD DETECTION ML PIPELINE")
    print("="*70 + "\n")
    
    # Create directories
    create_directories()
    
    # Step 1: Data Processing
    print("\n[STEP 1/4] DATA PROCESSING")
    print("-"*70)
    processor = FraudDataProcessor()
    X_train, X_test, y_train, y_test = processor.process_pipeline(data_path)
    
    # Save processed data
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    if not train:
        print("\nSkipping training (--no-train flag set)")
        return
    
    # Step 2: Model Training
    print("\n[STEP 2/4] MODEL TRAINING")
    print("-"*70)
    model = FraudDetectionModel()
    model.train(X_train, y_train, use_smote=True)
    model.save_model()
    
    if not evaluate:
        print("\nSkipping evaluation (--no-evaluate flag set)")
        return
    
    # Step 3: Model Evaluation
    print("\n[STEP 3/4] MODEL EVALUATION")
    print("-"*70)
    metrics, cm, y_proba = model.evaluate(X_test, y_test)
    
    # Create visualizations
    model.plot_confusion_matrix(cm)
    model.plot_roc_curve(y_test, y_proba)
    model.plot_precision_recall_curve(y_test, y_proba)
    model.plot_feature_importance(X_train)
    
    if not explain:
        print("\nSkipping explainability (--no-explain flag set)")
        return
    
    # Step 4: Model Explainability
    print("\n[STEP 4/4] MODEL EXPLAINABILITY")
    print("-"*70)
    explainer = FraudExplainer(model.model)
    explainer.create_explainer(X_train)
    
    y_pred = model.predict(X_test)
    explainer.generate_report(
        X_test.head(100), 
        y_test[:100], 
        y_pred[:100], 
        y_proba[:100]
    )
    
    # Final Summary
    print("\n" + "="*70)
    print(" "*20 + "PIPELINE COMPLETE!")
    print("="*70)
    print("\n📊 Results Summary:")
    print(f"  • Precision: {metrics['precision']:.4f}")
    print(f"  • Recall: {metrics['recall']:.4f}")
    print(f"  • F1-Score: {metrics['f1_score']:.4f}")
    print(f"  • ROC-AUC: {metrics['roc_auc']:.4f}")
    print("\n📁 Outputs:")
    print("  • Model: models/xgboost_fraud.pkl")
    print("  • Visualizations: outputs/")
    print("  • Processed Data: data/processed/")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fraud Detection ML Pipeline'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default=None,
        help='Path to input data file'
    )
    parser.add_argument(
        '--train', 
        action='store_true',
        help='Train the model'
    )
    parser.add_argument(
        '--no-train', 
        action='store_true',
        help='Skip training'
    )
    parser.add_argument(
        '--evaluate', 
        action='store_true',
        help='Evaluate the model'
    )
    parser.add_argument(
        '--no-evaluate', 
        action='store_true',
        help='Skip evaluation'
    )
    parser.add_argument(
        '--explain', 
        action='store_true',
        help='Generate explainability report'
    )
    parser.add_argument(
        '--no-explain', 
        action='store_true',
        help='Skip explainability'
    )
    
    args = parser.parse_args()
    
    # Default to all steps if no flags provided
    train = not args.no_train if args.no_train else True
    evaluate = not args.no_evaluate if args.no_evaluate else True
    explain = not args.no_explain if args.no_explain else True
    
    run_pipeline(
        data_path=args.data,
        train=train,
        evaluate=evaluate,
        explain=explain
    )
