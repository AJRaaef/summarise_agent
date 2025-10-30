import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def numeric_summary(df, numeric_cols):
    """Vectorized numeric summary computation"""
    if not numeric_cols:
        return {}
    
    summary = {}
    numeric_df = df[numeric_cols]
    
    for col in numeric_cols:
        data = numeric_df[col].dropna()
        if len(data) == 0:
            summary[col] = {"mean": 0, "sum": 0, "outliers": [], "skewness": 0}
            continue
            
        # Vectorized statistics
        mean, std = data.mean(), data.std()
        
        # Efficient outlier detection using IQR (more robust than std)
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)].tolist()
        
        summary[col] = {
            "mean": mean,
            "sum": data.sum(),
            "outliers": outliers,
            "std": std,
            "min": data.min(),
            "max": data.max(),
            "skewness": data.skew()
        }
    
    return summary

def categorical_summary(df, categorical_cols):
    """Optimized categorical summary with top-N values"""
    if not categorical_cols:
        return {}
    
    summary = {}
    MAX_CATEGORIES = 20  # Limit to prevent memory issues
    
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        
        # Limit to top categories for large cardinality
        if len(value_counts) > MAX_CATEGORIES:
            top_values = value_counts.head(MAX_CATEGORIES)
            others_count = value_counts.iloc[MAX_CATEGORIES:].sum()
            freq = top_values.to_dict()
            freq['_others'] = others_count
        else:
            freq = value_counts.to_dict()
        
        summary[col] = freq
    
    return summary