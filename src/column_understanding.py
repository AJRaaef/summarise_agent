from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Load once globally
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')  # free and lightweight
    return _model

# Expanded concepts for better matching
COMMON_CONCEPTS = [
    "Revenue", "Sales", "Income", "Profit", "Cost", "Price", "Amount", "Fee", "Charge",
    "Passengers", "Customers", "Users", "Visitors", "Clients",
    "Delay", "Wait", "Duration", "Time", "Hours", "Minutes", "Seconds",
    "Location", "Address", "City", "Country", "Region", "Area", "Zone",
    "Quantity", "Count", "Number", "Total", "Volume", "Capacity",
    "Date", "Month", "Year", "Day", "Time", "Timestamp",
    "Status", "Type", "Category", "Class", "Group", "Segment",
    "Score", "Rating", "Grade", "Rank", "Percentage", "Ratio",
    "Name", "ID", "Identifier", "Code", "Key",
    "Description", "Note", "Comment", "Remark",
    "Weight", "Height", "Length", "Width", "Size",
    "Temperature", "Speed", "Velocity", "Acceleration",
    "Price", "Cost", "Value", "Worth", "Amount"
]

def infer_columns(df):
    """
    Optimized column meaning inference using semantic similarity
    """
    columns = [str(col) for col in df.columns.tolist()]
    model = get_model()
    
    # Batch encode for efficiency
    col_embeddings = model.encode(columns, convert_to_tensor=True, show_progress_bar=False)
    concept_embeddings = model.encode(COMMON_CONCEPTS, convert_to_tensor=True, show_progress_bar=False)
    
    # Batch similarity computation
    similarities = util.pytorch_cos_sim(col_embeddings, concept_embeddings)
    best_matches = similarities.argmax(dim=1)
    
    col_mapping = {}
    for i, col in enumerate(columns):
        best_match = COMMON_CONCEPTS[best_matches[i]]
        confidence = similarities[i][best_matches[i]].item()
        
        # Only use inferred concept if confidence is high enough
        if confidence > 0.3:
            col_mapping[col] = best_match
        else:
            col_mapping[col] = col  # Fallback to original column name
    
    return col_mapping

def detect_data_quality_issues(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Comprehensive data quality assessment
    Returns dictionary of issue types with specific problems
    """
    issues = {
        'missing_data': [],
        'data_type_issues': [],
        'consistency_issues': [],
        'outlier_issues': [],
        'cardinality_issues': [],
        'format_issues': []
    }
    
    # Check for missing values
    missing_series = df.isnull().sum()
    high_missing_cols = missing_series[missing_series > 0]
    for col, missing_count in high_missing_cols.items():
        missing_pct = (missing_count / len(df)) * 100
        if missing_pct > 50:
            issues['missing_data'].append(f"{col}: {missing_pct:.1f}% missing values (CRITICAL)")
        elif missing_pct > 10:
            issues['missing_data'].append(f"{col}: {missing_pct:.1f}% missing values (HIGH)")
        elif missing_pct > 0:
            issues['missing_data'].append(f"{col}: {missing_pct:.1f}% missing values")
    
    # Check data type consistency
    for col in df.columns:
        # Check for mixed data types
        if df[col].dtype == 'object':
            # Check if numeric data stored as string
            numeric_count = 0
            total_count = 0
            for val in df[col].dropna():
                total_count += 1
                try:
                    float(str(val))
                    numeric_count += 1
                except:
                    pass
            
            if total_count > 0 and numeric_count / total_count > 0.8:
                issues['data_type_issues'].append(f"{col}: Mostly numeric data stored as text")
        
        # Check for inconsistent formats in text columns
        if df[col].dtype == 'object':
            sample_values = df[col].dropna().head(100)
            if len(sample_values) > 0:
                # Check for mixed case inconsistencies
                upper_count = sum(1 for val in sample_values if str(val).isupper())
                lower_count = sum(1 for val in sample_values if str(val).islower())
                if 0 < upper_count < len(sample_values) and 0 < lower_count < len(sample_values):
                    issues['format_issues'].append(f"{col}: Inconsistent text casing")
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        duplicate_pct = (duplicate_count / len(df)) * 100
        issues['consistency_issues'].append(f"Duplicate rows: {duplicate_count} ({duplicate_pct:.1f}%)")
    
    # Check for constant columns (no variation)
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        issues['cardinality_issues'].append(f"Constant columns with no variation: {', '.join(constant_cols)}")
    
    # Check for high cardinality categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.9 and len(df) > 100:
            issues['cardinality_issues'].append(f"{col}: High cardinality ({df[col].nunique()} unique values)")
    
    # Check for potential outliers in numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if len(df[col].dropna()) > 10:  # Need enough data for outlier detection
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_pct = (len(outliers) / len(df[col].dropna())) * 100
            
            if outlier_pct > 10:
                issues['outlier_issues'].append(f"{col}: {outlier_pct:.1f}% potential outliers")
    
    # Remove empty issue categories
    return {k: v for k, v in issues.items() if v}
