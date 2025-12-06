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
    print(f"Fraud (1):
