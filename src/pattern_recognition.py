import numpy as np
import pandas as pd

def numeric_summary(df, numeric_cols):
    """Vectorized numeric summary computation without external dependencies"""
    if not numeric_cols:
        return {}
    
    summary = {}
    
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) == 0:
            summary[col] = {
                "mean": 0, 
                "sum": 0, 
                "outliers": [], 
                "std": 0,
                "min": 0,
                "max": 0,
                "skewness": 0
            }
            continue
            
        # Basic statistics
        mean = data.mean()
        std = data.std()
        data_sum = data.sum()
        data_min = data.min()
        data_max = data.max()
        
        # Efficient outlier detection using IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)].tolist()
        
        # Manual skewness calculation
        if std > 0 and len(data) > 0:
            skewness = ((data - mean) ** 3).mean() / (std ** 3)
        else:
            skewness = 0
        
        summary[col] = {
            "mean": mean,
            "sum": data_sum,
            "outliers": outliers,
            "std": std,
            "min": data_min,
            "max": data_max,
            "skewness": skewness
        }
    
    return summary

def categorical_summary(df, categorical_cols):
    """Optimized categorical summary with top-N values"""
    if not categorical_cols:
        return {}
    
    summary = {}
    MAX_CATEGORIES = 50  # Increased limit for better analysis
    
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
