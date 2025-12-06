"""
Data Processing Module for Fraud Detection
Handles data loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml


class FraudDataProcessor:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.scaler = StandardScaler()
    
    def load_data(self, file_path=None):
        """Load raw transaction data"""
        if file_path is None:
            file_path = self.config['data']['raw_path']
        
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Data shape: {df.shape}")
        print(f"Fraud rate: {df['Class'].mean()*100:.3f}%")
        return df
    
    def engineer_features(self, df):
        """Create domain-specific features for fraud detection"""
        print("Engineering features...")
        df = df.copy()
        
        # Time-based features
        if self.config['features']['time_based']:
            df['Hour'] = (df['Time'] / 3600) % 24
            df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
            df['Time_Since_Start'] = df['Time'] - df['Time'].min()
        
        # Amount-based features
        if self.config['features']['amount_based']:
            df['Amount_Log'] = np.log1p(df['Amount'])
            df['Amount_Squared'] = df['Amount'] ** 2
            df['Amount_Percentile'] = df['Amount'].rank(pct=True)
            
            # Binning amounts
            df['Amount_Bin'] = pd.qcut(df['Amount'], q=10, labels=False, duplicates='drop')
        
        # Velocity features (transactions in time windows)
        if self.config['features']['velocity']:
            df = df.sort_values('Time')
            df['Trans_Last_1h'] = df.groupby('Time')['Time'].transform(
                lambda x: ((df['Time'] - x.iloc[0]) <= 3600).sum()
            )
        
        # Statistical features from V columns
        if self.config['features']['statistical']:
            v_cols = [col for col in df.columns if col.startswith('V')]
            df['V_Mean'] = df[v_cols].mean(axis=1)
            df['V_Std'] = df[v_cols].std(axis=1)
            df['V_Max'] = df[v_cols].max(axis=1)
            df['V_Min'] = df[v_cols].min(axis=1)
        
        print(f"Features after engineering: {df.shape[1]}")
        return df
    
    def clean_data(self, df):
        """Clean and handle missing values"""
        print("Cleaning data...")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print(f"Missing values found:\n{missing[missing > 0]}")
            df = df.fillna(df.median())
        
        # Remove duplicates
        initial_shape = df.shape[0]
        df = df.drop_duplicates()
        print(f"Removed {initial_shape - df.shape[0]} duplicate rows")
        
        return df
    
    def split_data(self, df):
        """Split data into train and test sets"""
        print("Splitting data...")
        
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        test_size = self.config['preprocessing']['test_size']
        random_state = self.config['preprocessing']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        print(f"Train fraud rate: {y_train.mean()*100:.3f}%")
        print(f"Test fraud rate: {y_test.mean()*100:.3f}%")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Scale numerical features"""
        if not self.config['preprocessing']['scale_features']:
            return X_train, X_test
        
        print("Scaling features...")
        
        # Don't scale 'Time' if it exists
        cols_to_scale = [col for col in X_train.columns if col != 'Time']
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[cols_to_scale] = self.scaler.fit_transform(X_train[cols_to_scale])
        X_test_scaled[cols_to_scale] = self.scaler.transform(X_test[cols_to_scale])
        
        return X_train_scaled, X_test_scaled
    
    def process_pipeline(self, file_path=None):
        """Execute full processing pipeline"""
        print("="*50)
        print("Starting Data Processing Pipeline")
        print("="*50)
        
        # Load data
        df = self.load_data(file_path)
        
        # Clean data
        df = self.clean_data(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # Scale features
        X_train, X_test = self.scale_features(X_train, X_test)
        
        print("="*50)
        print("Data Processing Complete!")
        print("="*50)
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    processor = FraudDataProcessor()
    X_train, X_test, y_train, y_test = processor.process_pipeline()
    
    # Save processed data
    print("\nSaving processed data...")
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    print("Processed data saved!")
