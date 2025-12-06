# Fraudulent Transaction Detection
## End-to-End Machine Learning Pipeline for Financial Fraud Detection

### 🎯 Project Overview
This project implements a complete machine learning pipeline to detect fraudulent financial transactions using XGBoost. It addresses real-world challenges, including extreme class imbalance, feature engineering, and model interpretability.

### 📋 Features
- **Supervised Classification**: Binary classification (fraud vs legitimate)
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Feature Engineering**: Domain-specific features for anomaly detection
- **Model Evaluation**: Comprehensive metrics (Precision, Recall, F1, ROC-AUC)
- **Explainability**: SHAP values for model interpretability

### 🏗️ Project Structure
```
fraud-detection/
├── data/
│   ├── raw/                      # Original dataset
│   ├── processed/                # Cleaned and engineered features
│   └── synthetic/                # SMOTE-balanced data
├── src/
│   ├── data_processing.py       # Data cleaning and feature engineering
│   ├── model.py                 # Model training and evaluation
│   ├── utils.py                 # Helper functions
│   └── explainer.py             # SHAP interpretability
├── models/
│   └── xgboost_fraud.pkl        # Saved trained model
├── config/
│   └── config.yaml              # Configuration parameters
├── requirements.txt
└── README.md
```
### 🚀 Quick Start

#### 1. Create Directory Structure
```bash
mkdir -p fraud-detection/{data/{raw,processed,synthetic},notebooks,src,models,config}
cd fraud-detection
```

#### 2. Installation
```bash
pip install -r requirements.txt
```

#### 3. Download Dataset
Use the Kaggle Credit Card Fraud dataset:
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/raw/
```

#### 4. Run Pipeline
```bash
# Option 1: Run full pipeline
python src/main.py

# Option 2: Run step by step
python src/data_processing.py
python src/model.py
python src/explainer.py
```

### 📊 Dataset
The project uses the **Credit Card Fraud Detection** dataset from Kaggle:
- **284,807 transactions** (492 fraudulent)
- **Class imbalance**: 0.172% fraud rate
- **Features**: 30 anonymized features (V1-V28, Time, Amount)

### 🔧 Key Components

#### Feature Engineering
- Transaction velocity features
- Amount percentile rankings
- Time-based patterns (hour, day/night)
- Statistical aggregations from V features

#### Model Selection
- **XGBoost Classifier**: Handles imbalanced data with scale_pos_weight
- SMOTE for synthetic minority oversampling
- Hyperparameter tuning via configuration
- Cross-validation for robust evaluation

#### Evaluation Metrics
- Confusion Matrix
- Precision-Recall Curve
- ROC-AUC Score
- Classification Report
- Business Metrics (cost-benefit analysis)

### 📈 Expected Results
- **Precision**: ~85-90%
- **Recall**: ~75-85%
- **F1-Score**: ~80-87%
- **ROC-AUC**: ~95-98%

### 🔍 Model Interpretability
SHAP (SHapley Additive exPlanations) provides:
- Feature importance rankings
- Individual prediction explanations
- Decision plots for stakeholder communication
- Dependence plots for feature interactions

### 📓 Jupyter Notebooks

#### 01_eda.ipynb
- Data exploration and visualization
- Class distribution analysis
- Feature correlation analysis
- Temporal and amount patterns

#### 02_preprocessing.ipynb
- Data cleaning pipeline
- Feature engineering workflow
- Train-test split
- Feature scaling

#### 03_modeling.ipynb
- SMOTE application
- Model training
- Hyperparameter tuning
- Model evaluation
- SHAP analysis

### 🤝 Contributing
Contributions welcome! Please follow standard PR procedures.

### 📄 License
MIT License

### 👥 Authors
github/skaus0 - Fraud Detection ML Pipeline

---
**Note**: This is a learning project. Real-world fraud detection requires additional security measures, regulatory compliance, and continuous monitoring.
