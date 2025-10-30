import pandas as pd
import numpy as np
from datetime import datetime
import base64
from io import BytesIO

def generate_comprehensive_report(df, analysis_results, ai_insights, quality_issues):
    """Generate comprehensive analysis report"""
    
    report = {
        'executive_summary': generate_executive_summary(df, ai_insights),
        'detailed_report': generate_detailed_report(df, analysis_results, ai_insights, quality_issues),
        'filename': f"data_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        'pdf_content': generate_pdf_content(df, analysis_results, ai_insights)
    }
    
    return report

def generate_executive_summary(df, ai_insights):
    """Generate executive summary"""
    summary = f"""
    DATA ANALYSIS EXECUTIVE SUMMARY
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    Dataset Overview:
    - Records: {df.shape[0]:,}
    - Features: {df.shape[1]}
    - Analysis Period: Complete dataset
    
    Key Insights:
    {chr(10).join(['• ' + insight for insight in ai_insights.get('insights', [])[:3]])}
    
    Recommendations:
    {chr(10).join(['• ' + rec for rec in ai_insights.get('recommendations', [])[:2]])}
    
    This analysis reveals significant patterns and opportunities for data-driven decision making.
    """
    return summary

def generate_detailed_report(df, analysis_results, ai_insights, quality_issues):
    """Generate detailed analysis report"""
    report = f"""
    COMPREHENSIVE DATA ANALYSIS REPORT
    ==================================
    
    Executive Summary
    {generate_executive_summary(df, ai_insights)}
    
    Detailed Analysis
    -----------------
    
    Data Quality Assessment:
    {generate_quality_section(quality_issues)}
    
    Statistical Summary:
    {generate_statistical_summary(df)}
    
    AI Insights:
    {chr(10).join(['• ' + insight for insight in ai_insights.get('insights', [])])}
    
    Predictive Analysis:
    {chr(10).join(['• ' + pred for pred in ai_insights.get('predictions', {}).get('short_term', []))]}
    
    Actionable Recommendations:
    {chr(10).join(['• ' + rec for rec in ai_insights.get('recommendations', [])])}
    """
    return report

def generate_quality_section(quality_issues):
    """Generate data quality section"""
    if not quality_issues:
        return "No significant data quality issues detected."
    
    section = "Data Quality Issues Found:\n"
    for issue_type, issues in quality_issues.items():
        section += f"\n{issue_type}:\n"
        for issue in issues[:5]:
            section += f"  - {issue}\n"
    return section

def generate_statistical_summary(df):
    """Generate statistical summary"""
    numeric_cols = df.select_dtypes(include=np.number).columns
    summary = "Numeric Columns Summary:\n"
    
    for col in numeric_cols[:5]:  # Limit to first 5 columns
        stats = df[col].describe()
        summary += f"\n{col}:\n"
        summary += f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}\n"
        summary += f"  Min: {stats['min']:.2f}, Max: {stats['max']:.2f}\n"
    
    return summary

def generate_pdf_content(df, analysis_results, ai_insights):
    """Generate PDF content (simplified - returns text for now)"""
    # In a real implementation, this would generate actual PDF
    # For now, return base64 encoded text
    report_text = generate_detailed_report(df, analysis_results, ai_insights, {})
    return base64.b64encode(report_text.encode()).decode()
