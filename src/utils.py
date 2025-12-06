"""
Utility Functions for Fraud Detection Project
Helper functions for data analysis and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def analyze_class_imbalance(y):
    """Analyze and visualize class distribution"""
    class_counts = pd.Series(y).value_counts()
    class_percentages = pd.Series(y).value_counts(normalize=True) * 100
    
    print("Class Distribution:")
    print(f"Legitimate (0): {class_counts[0]:,} ({class_percentages[0]:.3f}%)")
    print(f"Fraud (1): {class_counts[1]:,} ({class_percentages[1]:.3f}%)")
    print(f"Imbalance Ratio: 1:{class_counts[0]/class_counts[1]:.1f}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Bar plot
    class_counts.plot(kind='bar', ax=ax1, color=['green', 'red'])
    ax1.set_title('Class Distribution (Count)')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_xticklabels(['Legitimate', 'Fraud'], rotation=0)
    
    # Pie chart
    ax2.pie(class_counts, labels=['Legitimate', 'Fraud'], 
            autopct='%1.3f%%', colors=['green', 'red'])
    ax2.set_title('Class Distribution (Percentage)')
    
    plt.tight_layout()
    plt.savefig('outputs/class_distribution.png', dpi=300, bbox_inches='tight')
    print("Class distribution plot saved!")


def plot_amount_distribution(df):
    """Analyze transaction amount distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Overall distribution
    axes[0, 0].hist(df['Amount'], bins=50, edgecolor='black')
    axes[0, 0].set_title('Transaction Amount Distribution')
    axes[0, 0].set_xlabel('Amount')
    axes[0, 0].set_ylabel('Frequency')
    
    # Log scale
    axes[0, 1].hist(np.log1p(df['Amount']), bins=50, edgecolor='black', color='orange')
    axes[0, 1].set_title('Transaction Amount Distribution (Log Scale)')
    axes[0, 1].set_xlabel('Log(Amount + 1)')
    axes[0, 1].set_ylabel('Frequency')
    
    # By class
    fraud = df[df['Class'] == 1]['Amount']
    legit = df[df['Class'] == 0]['Amount']
    
    axes[1, 0].hist([legit, fraud], bins=50, label=['Legitimate', 'Fraud'], 
                     color=['green', 'red'], alpha=0.7)
    axes[1, 0].set_title('Amount Distribution by Class')
    axes[1, 0].set_xlabel('Amount')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Box plot
    df.boxplot(column='Amount', by='Class', ax=axes[1, 1])
    axes[1, 1].set_title('Amount Distribution by Class (Box Plot)')
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('Amount')
    
    plt.suptitle('Transaction Amount Analysis', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig('outputs/amount_distribution.png', dpi=300, bbox_inches='tight')
    print("Amount distribution plot saved!")


def plot_time_analysis(df):
    """Analyze temporal patterns in transactions"""
    df['Hour'] = (df['Time'] / 3600) % 24
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Transactions over time
    axes[0, 0].plot(df['Time'] / 3600, alpha=0.5)
    axes[0, 0].set_title('Transactions Over Time')
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('Transaction Index')
    
    # Hourly distribution
    hour_counts = df.groupby('Hour').size()
    axes[0, 1].bar(hour_counts.index, hour_counts.values, edgecolor='black')
    axes[0, 1].set_title('Transaction Count by Hour of Day')
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].set_ylabel('Count')
    
    # Fraud by hour
    fraud_by_hour = df[df['Class'] == 1].groupby('Hour').size()
    axes[1, 0].bar(fraud_by_hour.index, fraud_by_hour.values, 
                   color='red', edgecolor='black')
    axes[1, 0].set_title('Fraud Transactions by Hour')
    axes[1, 0].set_xlabel('Hour')
    axes[1, 0].set_ylabel('Fraud Count')
    
    # Fraud rate by hour
    fraud_rate = df.groupby('Hour')['Class'].mean() * 100
    axes[1, 1].plot(fraud_rate.index, fraud_rate.values, marker='o', color='red')
    axes[1, 1].set_title('Fraud Rate by Hour')
    axes[1, 1].set_xlabel('Hour')
    axes[1, 1].set_ylabel('Fraud Rate (%)')
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('Temporal Analysis', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig('outputs/time_analysis.png', dpi=300, bbox_inches='tight')
    print("Time analysis plot saved!")


def calculate_business_metrics(y_true, y_pred, amounts, 
                                 fraud_investigation_cost=50,
                                 fraud_loss_multiplier=1.0):
    """Calculate business-relevant metrics for fraud detection"""
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Get amounts for each category
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    amounts = np.array(amounts)
    
    # Calculate costs
    investigation_costs = fp * fraud_investigation_cost
    fraud_losses = np.sum(amounts[(y_true == 1) & (y_pred == 0)]) * fraud_loss_multiplier
    prevented_fraud = np.sum(amounts[(y_true == 1) & (y_pred == 1)])
    
    total_cost = investigation_costs + fraud_losses
    net_benefit = prevented_fraud - total_cost
    
    metrics = {
        'True Positives': tp,
        'False Positives': fp,
        'True Negatives': tn,
        'False Negatives': fn,
        'Investigation Costs': f'${investigation_costs:,.2f}',
        'Fraud Losses': f'${fraud_losses:,.2f}',
        'Prevented Fraud': f'${prevented_fraud:,.2f}',
        'Total Cost': f'${total_cost:,.2f}',
        'Net Benefit': f'${net_benefit:,.2f}'
    }
    
    print("\nBusiness Metrics:")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key:.<40} {value}")
    print("="*50)
    
    return metrics


def generate_data_quality_report(df):
    """Generate comprehensive data quality report"""
    print("\n" + "="*70)
    print(" "*20 + "DATA QUALITY REPORT")
    print("="*70)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of Features: {df.shape[1]}")
    print(f"Number of Samples: {df.shape[0]:,}")
    
    # Missing values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✓ No missing values found")
    else:
        print(missing[missing > 0])
    
    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates:,}")
    
    # Data types
    print("\nData Types:")
    print(df.dtypes.value_counts())
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"\nMemory Usage: {memory_mb:.2f} MB")
    
    # Basic statistics
    print("\nNumerical Features Summary:")
    print(df.describe())
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Utility functions loaded successfully!")
    print("Available functions:")
    print("  - analyze_class_imbalance()")
    print("  - plot_amount_distribution()")
    print("  - plot_time_analysis()")
    print("  - calculate_business_metrics()")
    print("  - generate_data_quality_report()")
