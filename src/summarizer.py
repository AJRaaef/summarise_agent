from src.utils import format_number

def generate_summary(col_mapping, df, numeric_summary_data, categorical_summary_data):
    """Generate concise, insightful summary"""
    
    summary_parts = []
    
    # Dataset overview
    summary_parts.append(f"Dataset Overview:")
    summary_parts.append(f"- Total Records: {df.shape[0]:,}")
    summary_parts.append(f"- Total Features: {df.shape[1]}")
    summary_parts.append(f"- Numeric Columns: {len(numeric_summary_data)}")
    summary_parts.append(f"- Categorical Columns: {len(categorical_summary_data)}")
    summary_parts.append("")
    
    # Key numeric insights
    if numeric_summary_data:
        summary_parts.append("Key Numeric Insights:")
        for col, info in list(numeric_summary_data.items())[:5]:  # Limit to top 5
            concept = col_mapping.get(col, col)
            summary_parts.append(f"• {concept}: Avg {format_number(info['mean'])}, "
                               f"Total {format_number(info['sum'])}, "
                               f"{len(info['outliers'])} outliers")
        summary_parts.append("")
    
    # Key categorical insights
    if categorical_summary_data:
        summary_parts.append("Key Categorical Insights:")
        for col, freq in list(categorical_summary_data.items())[:5]:  # Limit to top 5
            concept = col_mapping.get(col, col)
            top_category = max(freq.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
            unique_count = len(freq)
            summary_parts.append(f"• {concept}: {unique_count} unique values, "
                               f"most common '{top_category[0]}' ({top_category[1]} occurrences)")
    
    # Data quality notes
    missing_data = df.isnull().sum()
    high_missing = missing_data[missing_data > 0]
    if len(high_missing) > 0:
        summary_parts.append("")
        summary_parts.append("Data Quality Notes:")
        for col, missing_count in high_missing.items():
            missing_pct = (missing_count / len(df)) * 100
            summary_parts.append(f"• {col}: {missing_count} missing values ({missing_pct:.1f}%)")
    
    return "\n".join(summary_parts)