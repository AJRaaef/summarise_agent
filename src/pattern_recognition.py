import numpy as np
import pandas as pd
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

def numeric_summary(df, numeric_cols):
    """Enhanced numeric summary computation"""
    if not numeric_cols:
        return {}
    
    summary = {}
    
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) == 0:
            summary[col] = {
                "mean": 0, "sum": 0, "outliers": [], "std": 0,
                "min": 0, "max": 0, "skewness": 0, "kurtosis": 0,
                "cv": 0, "q1": 0, "median": 0, "q3": 0, "count": 0
            }
            continue
            
        # Basic statistics
        mean = data.mean()
        std = data.std()
        data_sum = data.sum()
        data_min = data.min()
        data_max = data.max()
        median = data.median()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        
        # Efficient outlier detection using IQR
        IQR = q3 - q1
        lower_bound = q1 - 1.5 * IQR
        upper_bound = q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)].tolist()
        
        # Manual skewness calculation
        if std > 0 and len(data) > 0:
            skewness = ((data - mean) ** 3).mean() / (std ** 3)
            # Kurtosis calculation
            kurtosis = ((data - mean) ** 4).mean() / (std ** 4) - 3
        else:
            skewness = 0
            kurtosis = 0
        
        # Coefficient of variation
        cv = (std / mean * 100) if mean != 0 else 0
        
        summary[col] = {
            "mean": mean,
            "sum": data_sum,
            "outliers": outliers,
            "std": std,
            "min": data_min,
            "max": data_max,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "cv": cv,
            "q1": q1,
            "median": median,
            "q3": q3,
            "count": len(data)
        }
    
    return summary

def categorical_summary(df, categorical_cols):
    """Enhanced categorical summary"""
    if not categorical_cols:
        return {}
    
    summary = {}
    MAX_CATEGORIES = 50
    
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        total_count = df[col].count()
        
        # Calculate additional statistics
        unique_count = len(value_counts)
        mode = value_counts.index[0] if len(value_counts) > 0 else None
        mode_count = value_counts.iloc[0] if len(value_counts) > 0 else 0
        mode_percentage = (mode_count / total_count * 100) if total_count > 0 else 0
        
        # Limit to top categories for large cardinality
        if len(value_counts) > MAX_CATEGORIES:
            top_values = value_counts.head(MAX_CATEGORIES)
            others_count = value_counts.iloc[MAX_CATEGORIES:].sum()
            freq = top_values.to_dict()
            freq['_others'] = others_count
        else:
            freq = value_counts.to_dict()
        
        summary[col] = {
            'frequency': freq,
            'unique_count': unique_count,
            'mode': mode,
            'mode_count': mode_count,
            'mode_percentage': mode_percentage,
            'total_count': total_count
        }
    
    return summary

def correlation_analysis(df):
    """Advanced correlation analysis"""
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr = abs(corr_matrix.iloc[i, j])
            if corr > 0.7:
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                high_corr_pairs.append({
                    'columns': (col1, col2),
                    'correlation': corr,
                    'type': 'positive' if corr_matrix.iloc[i, j] > 0 else 'negative'
                })
    
    return {
        'matrix': corr_matrix,
        'high_correlations': high_corr_pairs
    }

def trend_analysis(df):
    """Analyze trends in numeric data"""
    trends = {}
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) < 5:  # Need minimum data points
            continue
            
        # Simple linear trend
        x = np.arange(len(data))
        y = data.values
        slope = np.polyfit(x, y, 1)[0]
        
        # Determine trend direction and strength
        if abs(slope) < 0.01 * np.std(y):
            direction = "stable"
            strength = "weak"
        else:
            direction = "increasing" if slope > 0 else "decreasing"
            strength = "strong" if abs(slope) > 0.05 * np.std(y) else "moderate"
        
        trends[col] = {
            'direction': direction,
            'strength': strength,
            'slope': slope,
            'data_points': len(data)
        }
    
    return trends

def anomaly_detection(df):
    """Advanced anomaly detection using multiple methods"""
    anomalies = {}
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) < 10:  # Need enough data
            continue
            
        # Method 1: IQR-based outliers
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_outliers = data[(data < lower_bound) | (data > upper_bound)].tolist()
        
        # Method 2: Z-score based outliers (for normal distributions)
        z_scores = np.abs((data - data.mean()) / data.std())
        z_outliers = data[z_scores > 3].tolist()
        
        # Combine methods
        all_outliers = list(set(iqr_outliers + z_outliers))
        
        if all_outliers:
            anomalies[col] = {
                'outliers': all_outliers,
                'count': len(all_outliers),
                'percentage': (len(all_outliers) / len(data)) * 100,
                'min_outlier': min(all_outliers) if all_outliers else None,
                'max_outlier': max(all_outliers) if all_outliers else None
            }
    
    return anomalies
