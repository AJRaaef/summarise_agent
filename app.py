import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import io
from src.data_loader import load_data
from src.column_understanding import infer_columns, detect_data_quality_issues
from src.pattern_recognition import (
    numeric_summary, 
    categorical_summary,
    correlation_analysis,
    trend_analysis,
    anomaly_detection
)
from concurrent.futures import ThreadPoolExecutor, as_completed

# Page config
st.set_page_config(
    page_title="AI Data Analysis Agent", 
    layout="wide",
    page_icon="🧠"
)

# Custom CSS for advanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
    }
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #4ECDC4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .summary-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .correlation-high {
        background-color: #ff6b6b !important;
        color: white !important;
        font-weight: bold;
    }
    .correlation-medium {
        background-color: #ffa500 !important;
        color: white !important;
    }
    .correlation-low {
        background-color: #4ecdc4 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header with animated elements
    st.markdown('<h1 class="main-header">🧠 Advanced AI Data Analysis Agent</h1>', unsafe_allow_html=True)
    
    # Feature showcase
    with st.expander("🚀 **AI-Powered Features**", expanded=True):
        cols = st.columns(4)
        features = [
            "🤖 Smart Pattern Detection",
            "📈 Predictive Analytics", 
            "🔍 Anomaly Detection",
            "📊 Automated Insights",
            "🎯 Action Recommendations",
            "📖 Data Storytelling",
            "🔄 Real-time Analysis",
            "📋 Professional Reports"
        ]
        for i, feature in enumerate(features):
            cols[i % 4].markdown(f'<div class="feature-card">{feature}</div>', unsafe_allow_html=True)
    
    # File upload with drag & drop style
    uploaded_file = st.file_uploader(
        "**Drag & Drop your data file here**", 
        type=["csv", "xlsx", "xls", "json", "parquet"],
        help="Supports CSV, Excel, JSON, and Parquet files"
    )
    
    if uploaded_file is not None:
        try:
            # Initialize session state for analysis results
            if 'analysis_complete' not in st.session_state:
                st.session_state.analysis_complete = False
            if 'ai_insights' not in st.session_state:
                st.session_state.ai_insights = None
            
            # Analysis pipeline
            run_advanced_analysis(uploaded_file)
            
        except Exception as e:
            st.error(f"🚨 Analysis Error: {str(e)}")
            st.info("💡 **Tip**: Try uploading a different file format or check data quality")

def run_advanced_analysis(uploaded_file):
    """Advanced analysis pipeline with AI capabilities"""
    
    # Progress tracking with multiple stages
    progress_stages = {
        "Loading Data": 10,
        "Data Quality Check": 20,
        "Pattern Recognition": 30,
        "AI Analysis": 50,
        "Advanced Analytics": 70,
        "Report Generation": 90,
        "Complete": 100
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Stage 1: Data Loading & Validation
    status_text.text(f"🔄 {list(progress_stages.keys())[0]}...")
    df = load_data(uploaded_file)
    progress_bar.progress(progress_stages["Loading Data"])
    time.sleep(0.5)
    
    # Display quick overview
    show_data_overview(df)
    
    # Stage 2: Data Quality Analysis
    status_text.text(f"🔍 {list(progress_stages.keys())[1]}...")
    quality_issues = detect_data_quality_issues(df)
    progress_bar.progress(progress_stages["Data Quality Check"])
    time.sleep(0.5)
    
    # Stage 3: Advanced Analysis (Parallel Processing)
    status_text.text(f"📊 {list(progress_stages.keys())[2]}...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all analysis tasks
        futures = {
            executor.submit(infer_columns, df): "column_analysis",
            executor.submit(numeric_summary, df, df.select_dtypes(include=np.number).columns.tolist()): "numeric_analysis",
            executor.submit(categorical_summary, df, df.select_dtypes(exclude=np.number).columns.tolist()): "categorical_analysis",
            executor.submit(correlation_analysis, df): "correlation_analysis",
            executor.submit(trend_analysis, df): "trend_analysis",
            executor.submit(anomaly_detection, df): "anomaly_detection"
        }
        
        results = {}
        for future in as_completed(futures):
            task_name = futures[future]
            try:
                results[task_name] = future.result()
            except Exception as e:
                st.warning(f"⚠️ {task_name} had issues: {str(e)}")
    
    progress_bar.progress(progress_stages["Pattern Recognition"])
    
    # Stage 4: AI-Powered Insights
    status_text.text(f"🤖 {list(progress_stages.keys())[3]}...")
    
    # Use simplified AI analysis
    ai_insights = generate_basic_insights(df, results)
    predictions = {'short_term': [], 'long_term': []}
    recommendations = generate_recommendations(df, results, quality_issues)
    story = generate_basic_story(df, results)
    
    st.session_state.ai_insights = {
        'insights': ai_insights,
        'predictions': predictions,
        'recommendations': recommendations,
        'story': story
    }
    
    progress_bar.progress(progress_stages["AI Analysis"])
    time.sleep(0.5)
    
    # Stage 5: Advanced Analytics
    status_text.text(f"📈 {list(progress_stages.keys())[4]}...")
    progress_bar.progress(progress_stages["Advanced Analytics"])
    
    # Stage 6: Report Generation
    status_text.text(f"📋 {list(progress_stages.keys())[5]}...")
    comprehensive_report = generate_comprehensive_report(df, results, st.session_state.ai_insights, quality_issues)
    progress_bar.progress(progress_stages["Report Generation"])
    
    # Complete
    status_text.text(f"✅ {list(progress_stages.keys())[6]}")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    st.session_state.analysis_complete = True
    st.session_state.results = results
    st.session_state.df = df
    
    # Display all results
    display_advanced_results(df, results, st.session_state.ai_insights, quality_issues, comprehensive_report)

def generate_comprehensive_summary(df, results):
    """Generate comprehensive summary in the requested format"""
    summary_parts = []
    
    # Dataset Overview
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    
    summary_parts.append("**Dataset Overview:**")
    summary_parts.append(f"- Total Records: {df.shape[0]:,}")
    summary_parts.append(f"- Total Features: {df.shape[1]}")
    summary_parts.append(f"- Numeric Columns: {len(numeric_cols)}")
    summary_parts.append(f"- Categorical Columns: {len(categorical_cols)}")
    summary_parts.append("")
    
    # Key Numeric Insights
    if 'numeric_analysis' in results and results['numeric_analysis']:
        summary_parts.append("**Key Numeric Insights:**")
        numeric_results = results['numeric_analysis']
        
        # Show top 5 numeric columns
        for col, stats in list(numeric_results.items())[:5]:
            mean_val = stats.get('mean', 0)
            sum_val = stats.get('sum', 0)
            outliers_count = len(stats.get('outliers', []))
            
            # Format numbers nicely
            if abs(mean_val) >= 1000:
                mean_str = f"{mean_val:,.2f}"
            else:
                mean_str = f"{mean_val:.2f}"
                
            if abs(sum_val) >= 1000:
                sum_str = f"{sum_val:,.2f}"
            else:
                sum_str = f"{sum_val:.2f}"
            
            summary_parts.append(f"• {col}: Avg {mean_str}, Total {sum_str}, {outliers_count} outliers")
        summary_parts.append("")
    
    # Key Categorical Insights
    if 'categorical_analysis' in results and results['categorical_analysis']:
        summary_parts.append("**Key Categorical Insights:**")
        categorical_results = results['categorical_analysis']
        
        # Show top 5 categorical columns
        for col, stats in list(categorical_results.items())[:5]:
            freq_data = stats.get('frequency', {})
            unique_count = stats.get('unique_count', 0)
            mode = stats.get('mode', 'N/A')
            mode_count = stats.get('mode_count', 0)
            
            # Handle cases where mode might be None or not available
            if mode is None or mode == 'N/A':
                mode_display = 'N/A'
                mode_count_display = 0
            else:
                mode_display = str(mode)[:30]  # Truncate long values
                mode_count_display = mode_count
            
            summary_parts.append(f"• {col}: {unique_count} unique values, most common '{mode_display}' ({mode_count_display} occurrences)")
    
    return "\n".join(summary_parts)

def generate_basic_insights(df, results):
    """Generate basic insights"""
    insights = []
    
    # Basic insights from results
    if 'numeric_analysis' in results:
        for col, stats in list(results['numeric_analysis'].items())[:3]:
            if abs(stats.get('skewness', 0)) > 1:
                direction = "positive" if stats['skewness'] > 0 else "negative"
                insights.append(f"{col} shows strong {direction} skewness")
    
    if 'correlation_analysis' in results:
        corr_data = results['correlation_analysis']
        if 'high_correlations' in corr_data:
            for corr in corr_data['high_correlations'][:2]:
                col1, col2 = corr['columns']
                insights.append(f"Strong correlation between {col1} and {col2}")
    
    if 'trend_analysis' in results:
        for col, trend in list(results['trend_analysis'].items())[:2]:
            if trend.get('strength') in ['strong', 'moderate']:
                insights.append(f"{trend['direction']} trend in {col}")
    
    return insights if insights else ["Dataset loaded successfully. Detailed analysis complete."]

def generate_recommendations(df, results, quality_issues):
    """Generate recommendations"""
    recommendations = []
    
    if quality_issues:
        recommendations.append("Address data quality issues before further analysis")
    
    if 'anomaly_detection' in results:
        total_anomalies = sum(anom.get('count', 0) for anom in results['anomaly_detection'].values())
        if total_anomalies > 0:
            recommendations.append(f"Review {total_anomalies} detected anomalies")
    
    if 'correlation_analysis' in results:
        corr_data = results['correlation_analysis']
        if 'high_correlations' in corr_data and corr_data['high_correlations']:
            recommendations.append("Investigate strong correlations for business insights")
    
    return recommendations if recommendations else ["Continue with deeper analysis based on initial findings"]

def generate_basic_story(df, results):
    """Generate basic data story"""
    return f"This dataset contains {df.shape[0]:,} records with {df.shape[1]} features. Analysis reveals patterns and relationships that can inform data-driven decisions."

def generate_comprehensive_report(df, results, ai_insights, quality_issues):
    """Generate comprehensive analysis report"""
    
    # Generate the new comprehensive summary
    comprehensive_summary = generate_comprehensive_summary(df, results)
    
    report = {
        'executive_summary': f"""
DATA ANALYSIS EXECUTIVE SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{comprehensive_summary}

Key AI Insights:
{chr(10).join(['• ' + insight for insight in ai_insights.get('insights', [])[:3]])}

This analysis provides valuable insights for data-driven decision making.
        """,
        'detailed_report': f"""
COMPREHENSIVE DATA ANALYSIS REPORT
==================================

Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{comprehensive_summary}

DATA QUALITY:
{generate_quality_report(quality_issues)}

KEY AI INSIGHTS:
{chr(10).join(['• ' + insight for insight in ai_insights.get('insights', [])])}

RECOMMENDATIONS:
{chr(10).join(['• ' + rec for rec in ai_insights.get('recommendations', [])])}
        """,
        'filename': f"data_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        'comprehensive_summary': comprehensive_summary
    }
    
    return report

def generate_quality_report(quality_issues):
    """Generate quality issues report"""
    if not quality_issues:
        return "No significant data quality issues detected."
    
    report = ""
    for issue_type, issues in quality_issues.items():
        report += f"\n{issue_type.upper()}:\n"
        for issue in issues[:5]:
            report += f"  - {issue}\n"
    
    return report

def show_data_overview(df):
    """Enhanced data overview with advanced metrics"""
    st.subheader("📊 Dataset Intelligence Overview")
    
    cols = st.columns(6)
    metrics = [
        ("Total Records", f"{df.shape[0]:,}"),
        ("Features", f"{df.shape[1]}"),
        ("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB"),
        ("Complete Cases", f"{(1 - df.isnull().any(axis=1).mean()) * 100:.1f}%"),
        ("Data Density", f"{(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.1f}%"),
        ("Unique Ratio", f"{(df.nunique() / len(df)).mean() * 100:.1f}%")
    ]
    
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)
    
    # Quick insights
    with st.expander("🔍 Quick Data Insights"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Types Distribution**")
            dtype_counts = df.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"- {dtype}: {count} columns")
        
        with col2:
            st.write("**Data Quality Snapshot**")
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            st.write(f"- Missing Values: {missing_cells} ({missing_cells/total_cells*100:.1f}%)")
            st.write(f"- Duplicate Rows: {df.duplicated().sum()}")
            st.write(f"- Constant Columns: {len([col for col in df.columns if df[col].nunique() == 1])}")

def create_simple_histogram(data, bins=20):
    """Create histogram data without using pandas cut (which creates Interval objects)"""
    if len(data) == 0:
        return pd.DataFrame({'bin_range': [], 'count': []})
    
    min_val = data.min()
    max_val = data.max()
    bin_width = (max_val - min_val) / bins
    
    # Create simple bin ranges
    bin_edges = [min_val + i * bin_width for i in range(bins + 1)]
    bin_counts = [0] * bins
    
    # Count values in each bin
    for value in data:
        if pd.notna(value):
            bin_index = min(int((value - min_val) / bin_width), bins - 1)
            bin_counts[bin_index] += 1
    
    # Create bin labels as simple strings
    bin_labels = []
    for i in range(bins):
        if i == bins - 1:
            label = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
        else:
            label = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
        bin_labels.append(label)
    
    return pd.DataFrame({
        'bin_range': bin_labels,
        'count': bin_counts
    })

def style_correlation_matrix(corr_matrix):
    """Style correlation matrix without matplotlib"""
    styled_df = corr_matrix.copy()
    
    # Convert to HTML with custom styling
    html = '<div style="overflow-x: auto;">'
    html += '<table style="border-collapse: collapse; width: 100%;">'
    
    # Header row
    html += '<tr><th style="border: 1px solid #ddd; padding: 8px; background: #f2f2f2;"></th>'
    for col in corr_matrix.columns:
        html += f'<th style="border: 1px solid #ddd; padding: 8px; background: #f2f2f2;">{col}</th>'
    html += '</tr>'
    
    # Data rows
    for i, row in enumerate(corr_matrix.index):
        html += f'<tr><td style="border: 1px solid #ddd; padding: 8px; background: #f2f2f2; font-weight: bold;">{row}</td>'
        for j, col in enumerate(corr_matrix.columns):
            value = corr_matrix.iloc[i, j]
            abs_value = abs(value)
            
            # Determine color intensity based on correlation strength
            if abs_value > 0.7:
                bg_color = "#ff6b6b"  # Strong red
                text_color = "white"
            elif abs_value > 0.5:
                bg_color = "#ffa500"  # Medium orange
                text_color = "white"
            elif abs_value > 0.3:
                bg_color = "#4ecdc4"  # Light teal
                text_color = "white"
            else:
                bg_color = "#f8f9fa"  # Light gray
                text_color = "black"
            
            html += f'<td style="border: 1px solid #ddd; padding: 8px; background: {bg_color}; color: {text_color}; text-align: center;">{value:.2f}</td>'
        html += '</tr>'
    
    html += '</table></div>'
    return html

def display_advanced_results(df, results, ai_insights, quality_issues, report):
    """Display all advanced analysis results"""
    
    # Create main navigation tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📋 Summary", "🤖 AI Insights", "📈 Analytics", "🔍 Patterns", 
        "📊 Visualizations", "📋 Data Quality", "📖 Story", "📄 Report"
    ])
    
    with tab1:
        display_comprehensive_summary(report['comprehensive_summary'])
    
    with tab2:
        display_ai_insights(ai_insights)
    
    with tab3:
        display_advanced_analytics(df, results)
    
    with tab4:
        display_pattern_analysis(results)
    
    with tab5:
        display_advanced_visualizations(df, results)
    
    with tab6:
        display_data_quality(quality_issues)
    
    with tab7:
        display_data_story(ai_insights['story'])
    
    with tab8:
        display_comprehensive_report(report)

def display_comprehensive_summary(summary_text):
    """Display the comprehensive summary in a beautiful format"""
    st.header("📋 Comprehensive Summary")
    
    # Display summary in a nice box
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 15px; margin: 1rem 0;">
        <h3 style="color: white; margin-bottom: 1rem;">📊 Dataset Overview & Key Insights</h3>
        <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 10px;">
            {summary_text.replace(chr(10), '<br>').replace('•', '•')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Also show as raw text for copying
    with st.expander("📝 Copy Summary Text"):
        st.text(summary_text)

def display_ai_insights(ai_insights):
    """Display AI-generated insights"""
    st.header("🤖 AI-Powered Insights")
    
    # Key Insights
    st.subheader("💡 Key Insights")
    for i, insight in enumerate(ai_insights['insights'][:5], 1):
        st.markdown(f'<div class="insight-box"><strong>Insight {i}:</strong> {insight}</div>', unsafe_allow_html=True)
    
    # Recommendations
    st.subheader("🎯 Actionable Recommendations")
    for i, recommendation in enumerate(ai_insights['recommendations'][:5], 1):
        st.markdown(f'<div class="warning-box"><strong>Recommendation {i}:</strong> {recommendation}</div>', unsafe_allow_html=True)

def display_advanced_analytics(df, results):
    """Display advanced analytical results"""
    st.header("📈 Advanced Analytics")
    
    # Correlation Heatmap
    if 'correlation_analysis' in results:
        st.subheader("🔄 Correlation Matrix")
        corr_data = results['correlation_analysis']
        if 'matrix' in corr_data:
            corr_matrix = corr_data['matrix']
            
            # Display styled correlation matrix
            st.markdown(style_correlation_matrix(corr_matrix), unsafe_allow_html=True)
        
        # High correlations
        if corr_data.get('high_correlations'):
            st.subheader("🔗 Strong Correlations")
            for corr in corr_data['high_correlations'][:5]:
                col1, col2 = corr['columns']
                correlation_value = corr['correlation']
                correlation_type = corr['type']
                
                # Color code based on strength
                if correlation_value > 0.7:
                    color = "🔴"
                elif correlation_value > 0.5:
                    color = "🟠"
                else:
                    color = "🟢"
                
                st.write(f"{color} **{col1}** ↔ **{col2}**: {correlation_value:.3f} ({correlation_type})")
    
    # Trend Analysis
    if 'trend_analysis' in results:
        st.subheader("📊 Trend Analysis")
        trends = results['trend_analysis']
        for col, trend_info in trends.items():
            with st.expander(f"Trend Analysis: {col}"):
                direction = trend_info.get('direction', 'N/A')
                strength = trend_info.get('strength', 'N/A')
                
                # Add emojis for better visualization
                if direction == "increasing":
                    arrow = "📈"
                elif direction == "decreasing":
                    arrow = "📉"
                else:
                    arrow = "➡️"
                
                st.write(f"**Direction:** {arrow} {direction}")
                st.write(f"**Strength:** {strength}")
                st.write(f"**Slope:** {trend_info.get('slope', 0):.4f}")
    
    # Anomaly Detection
    if 'anomaly_detection' in results:
        st.subheader("🚨 Anomaly Detection")
        anomalies = results['anomaly_detection']
        total_anomalies = sum(anom.get('count', 0) for anom in anomalies.values())
        
        # Color code total anomalies
        if total_anomalies > 100:
            anomaly_color = "🔴"
        elif total_anomalies > 50:
            anomaly_color = "🟠"
        else:
            anomaly_color = "🟢"
            
        st.metric("Total Anomalies Detected", f"{anomaly_color} {total_anomalies}")
        
        for col, anomaly_info in list(anomalies.items())[:3]:
            with st.expander(f"Anomalies in {col}"):
                count = anomaly_info.get('count', 0)
                percentage = anomaly_info.get('percentage', 0)
                
                if percentage > 10:
                    severity = "🔴 High"
                elif percentage > 5:
                    severity = "🟠 Medium"
                else:
                    severity = "🟢 Low"
                    
                st.write(f"**Count:** {count}")
                st.write(f"**Percentage:** {percentage:.1f}%")
                st.write(f"**Severity:** {severity}")

def display_pattern_analysis(results):
    """Display pattern recognition results"""
    st.header("🔍 Pattern Recognition")
    
    # Numeric Patterns
    if 'numeric_analysis' in results:
        st.subheader("🔢 Numeric Patterns")
        numeric_results = results['numeric_analysis']
        for col, stats in list(numeric_results.items())[:5]:
            with st.expander(f"Patterns in {col}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Distribution**")
                    skewness = stats.get('skewness', 0)
                    if abs(skewness) > 1:
                        skew_icon = "⚠️"
                    else:
                        skew_icon = "✅"
                    st.write(f"- Skewness: {skew_icon} {skewness:.2f}")
                    st.write(f"- Kurtosis: {stats.get('kurtosis', 0):.2f}")
                    st.write(f"- CV: {stats.get('cv', 0):.2f}%")
                
                with col2:
                    st.write("**Outliers**")
                    outlier_count = len(stats.get('outliers', []))
                    outlier_percentage = outlier_count/stats.get('count', 1)*100
                    
                    if outlier_percentage > 10:
                        outlier_icon = "🚨"
                    elif outlier_percentage > 5:
                        outlier_icon = "⚠️"
                    else:
                        outlier_icon = "✅"
                        
                    st.write(f"- Count: {outlier_icon} {outlier_count}")
                    st.write(f"- % of data: {outlier_percentage:.1f}%")

def display_advanced_visualizations(df, results):
    """Display interactive visualizations"""
    st.header("📊 Advanced Visualizations")
    
    # Simple visualizations using Streamlit
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if numeric_cols:
        selected_col = st.selectbox("Select Column for Visualization", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Line Chart")
            st.line_chart(df[selected_col])
        
        with col2:
            st.subheader("📊 Distribution")
            # Use our custom histogram function instead of pd.cut
            data = df[selected_col].dropna()
            if len(data) > 0:
                hist_df = create_simple_histogram(data, bins=15)
                if not hist_df.empty:
                    # Display as bar chart with proper labels
                    chart_data = hist_df.set_index('bin_range')['count']
                    st.bar_chart(chart_data)
                else:
                    st.info("No data available for histogram")
            else:
                st.info("No data available for histogram")

def display_data_quality(quality_issues):
    """Display data quality assessment"""
    st.header("📋 Data Quality Assessment")
    
    if quality_issues:
        st.subheader("🚨 Quality Issues Found")
        for issue_type, issues in quality_issues.items():
            with st.expander(f"{issue_type.replace('_', ' ').title()} ({len(issues)} issues)"):
                for issue in issues[:10]:
                    # Add severity indicators
                    if "CRITICAL" in issue or "HIGH" in issue:
                        icon = "🔴"
                    elif "missing" in issue.lower():
                        icon = "🟠"
                    else:
                        icon = "🟡"
                    st.write(f"{icon} {issue}")
    else:
        st.success("✅ No major data quality issues detected!")

def display_data_story(story):
    """Display data storytelling"""
    st.header("📖 Data Story")
    st.markdown(f"""
    <div style="background: #f8f9fa; padding: 2rem; border-radius: 10px; border-left: 5px solid #4ECDC4;">
    {story}
    </div>
    """, unsafe_allow_html=True)

def display_comprehensive_report(report):
    """Display comprehensive analysis report"""
    st.header("📄 Comprehensive Analysis Report")
    
    st.download_button(
        "📥 Download Full Report (TXT)",
        data=report['detailed_report'],
        file_name=report['filename'],
        mime="text/plain"
    )
    
    st.subheader("Executive Summary")
    st.write(report['executive_summary'])
    
    with st.expander("View Detailed Report"):
        st.text(report['detailed_report'])

if __name__ == "__main__":
    main()
