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
from src.advanced_ai import (
    generate_ai_insights,
    predict_trends,
    suggest_actions,
    data_storytelling
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
    
    # Use simplified AI analysis if advanced_ai is not available
    try:
        ai_insights = generate_ai_insights(df, results)
        predictions = predict_trends(df, results)
        recommendations = suggest_actions(df, results, quality_issues)
        story = data_storytelling(df, results, ai_insights)
    except Exception as e:
        # Fallback to basic insights
        st.warning("Using simplified AI analysis")
        ai_insights = generate_basic_insights(df, results)
        predictions = {'short_term': [], 'long_term': []}
        recommendations = ["Analyze data quality issues first"]
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

def generate_basic_insights(df, results):
    """Generate basic insights when advanced AI is not available"""
    insights = []
    
    # Basic insights from results
    if 'numeric_analysis' in results:
        for col, stats in results['numeric_analysis'].items():
            if stats.get('skewness', 0) > 1:
                insights.append(f"{col} shows strong positive skewness")
            elif stats.get('skewness', 0) < -1:
                insights.append(f"{col} shows strong negative skewness")
    
    if 'correlation_analysis' in results:
        high_corrs = results['correlation_analysis'].get('high_correlations', [])
        for corr in high_corrs[:3]:
            col1, col2 = corr['columns']
            insights.append(f"Strong {corr['type']} correlation between {col1} and {col2}")
    
    return insights if insights else ["Dataset loaded successfully. Run detailed analysis for more insights."]

def generate_basic_story(df, results):
    """Generate basic data story"""
    return f"This dataset contains {df.shape[0]:,} records with {df.shape[1]} features. Basic analysis reveals patterns in the data that can inform decision-making."

def generate_comprehensive_report(df, results, ai_insights, quality_issues):
    """Generate comprehensive analysis report"""
    report = {
        'executive_summary': f"""
DATA ANALYSIS EXECUTIVE SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Dataset Overview:
- Records: {df.shape[0]:,}
- Features: {df.shape[1]}
- Analysis Complete: Yes

Key Insights:
{chr(10).join(['• ' + insight for insight in ai_insights.get('insights', [])[:3]])}

This analysis provides valuable insights for data-driven decision making.
        """,
        'detailed_report': f"""
COMPREHENSIVE DATA ANALYSIS REPORT
==================================

Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA QUALITY:
{generate_quality_report(quality_issues)}

KEY INSIGHTS:
{chr(10).join(['• ' + insight for insight in ai_insights.get('insights', [])])}

RECOMMENDATIONS:
{chr(10).join(['• ' + rec for rec in ai_insights.get('recommendations', [])])}
        """,
        'filename': f"data_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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

def display_advanced_results(df, results, ai_insights, quality_issues, report):
    """Display all advanced analysis results"""
    
    # Create main navigation tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🤖 AI Insights", "📈 Analytics", "🔍 Patterns", 
        "📊 Visualizations", "📋 Data Quality", "📖 Story", "📄 Report"
    ])
    
    with tab1:
        display_ai_insights(ai_insights)
    
    with tab2:
        display_advanced_analytics(df, results)
    
    with tab3:
        display_pattern_analysis(results)
    
    with tab4:
        display_advanced_visualizations(df, results)
    
    with tab5:
        display_data_quality(quality_issues)
    
    with tab6:
        display_data_story(ai_insights['story'])
    
    with tab7:
        display_comprehensive_report(report)

def display_ai_insights(ai_insights):
    """Display AI-generated insights"""
    st.header("🤖 AI-Powered Insights")
    
    # Key Insights
    st.subheader("💡 Key Insights")
    for i, insight in enumerate(ai_insights['insights'][:5], 1):
        st.markdown(f'<div class="insight-box"><strong>Insight {i}:</strong> {insight}</div>', unsafe_allow_html=True)
    
    # Predictions
    st.subheader("🔮 Predictive Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Short-term Trends**")
        for pred in ai_insights['predictions'].get('short_term', [])[:3]:
            st.write(f"📈 {pred}")
    
    with col2:
        st.write("**Long-term Projections**")
        for pred in ai_insights['predictions'].get('long_term', [])[:3]:
            st.write(f"🎯 {pred}")
    
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
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None), use_container_width=True)
        
        # High correlations
        if corr_data.get('high_correlations'):
            st.subheader("🔗 Strong Correlations")
            for corr in corr_data['high_correlations'][:5]:
                col1, col2 = corr['columns']
                st.write(f"**{col1}** ↔ **{col2}**: {corr['correlation']:.3f} ({corr['type']})")
    
    # Trend Analysis
    if 'trend_analysis' in results:
        st.subheader("📊 Trend Analysis")
        trends = results['trend_analysis']
        for col, trend_info in trends.items():
            with st.expander(f"Trend Analysis: {col}"):
                st.write(f"**Direction:** {trend_info.get('direction', 'N/A')}")
                st.write(f"**Strength:** {trend_info.get('strength', 'N/A')}")
                st.write(f"**Slope:** {trend_info.get('slope', 0):.4f}")
    
    # Anomaly Detection
    if 'anomaly_detection' in results:
        st.subheader("🚨 Anomaly Detection")
        anomalies = results['anomaly_detection']
        total_anomalies = sum(anom.get('count', 0) for anom in anomalies.values())
        st.metric("Total Anomalies Detected", total_anomalies)
        
        for col, anomaly_info in list(anomalies.items())[:3]:
            with st.expander(f"Anomalies in {col}"):
                st.write(f"Count: {anomaly_info.get('count', 0)}")
                st.write(f"Percentage: {anomaly_info.get('percentage', 0):.1f}%")
                if anomaly_info.get('outliers'):
                    st.write(f"Range: {anomaly_info.get('min_outlier'):.2f} to {anomaly_info.get('max_outlier'):.2f}")

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
                    st.write(f"- Skewness: {stats.get('skewness', 0):.2f}")
                    st.write(f"- Kurtosis: {stats.get('kurtosis', 0):.2f}")
                    st.write(f"- CV: {stats.get('cv', 0):.2f}%")
                
                with col2:
                    st.write("**Outliers**")
                    st.write(f"- Count: {len(stats.get('outliers', []))}")
                    st.write(f"- % of data: {len(stats.get('outliers', []))/stats.get('count', 1)*100:.1f}%")

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
            # Simple histogram using value_counts on binned data
            if len(df[selected_col].dropna()) > 0:
                hist_data = pd.cut(df[selected_col], bins=20).value_counts().sort_index()
                hist_df = pd.DataFrame({
                    'Bin': [f"{interval.left:.1f}-{interval.right:.1f}" for interval in hist_data.index],
                    'Count': hist_data.values
                })
                st.bar_chart(hist_data)

def display_data_quality(quality_issues):
    """Display data quality assessment"""
    st.header("📋 Data Quality Assessment")
    
    if quality_issues:
        st.subheader("🚨 Quality Issues Found")
        for issue_type, issues in quality_issues.items():
            with st.expander(f"{issue_type.replace('_', ' ').title()} ({len(issues)} issues)"):
                for issue in issues[:10]:
                    st.write(f"- {issue}")
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
