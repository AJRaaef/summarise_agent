import streamlit as st
import pandas as pd
import numpy as np
from src.data_loader import load_data
from src.column_understanding import infer_columns
from src.pattern_recognition import numeric_summary, categorical_summary
from src.summarizer import generate_summary
from concurrent.futures import ThreadPoolExecutor
import time

# Cache everything possible
st.set_page_config(page_title="Smart Data Summarizer", layout="wide")
st.title("📊 Smart Data Summarization Agent")

@st.cache_data(show_spinner=False, ttl=3600)
def cached_load_data(file):
    return load_data(file)

@st.cache_data(show_spinner=False, ttl=3600)
def cached_infer_columns(df):
    return infer_columns(df)

@st.cache_data(show_spinner=False, ttl=3600)
def cached_numeric_summary(df, numeric_cols):
    return numeric_summary(df, numeric_cols)

@st.cache_data(show_spinner=False, ttl=3600)
def cached_categorical_summary(df, categorical_cols):
    return categorical_summary(df, categorical_cols)

def main():
    st.sidebar.header("📁 File Requirements")
    st.sidebar.markdown("""
    **Supported formats:**
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    
    **File should contain:**
    - At least 1 column of data
    - Header row (column names)
    - Actual data values
    """)
    
    uploaded_file = st.file_uploader(
        "Upload your CSV or Excel file", 
        type=["csv", "xlsx", "xls"],
        help="Upload a CSV or Excel file with data to analyze"
    )
    
    if uploaded_file is not None:
        # Validate file first
        is_valid, validation_message = validate_file(uploaded_file)
        
        if not is_valid:
            st.error(f"Invalid file: {validation_message}")
            return
        
        try:
            # Show file info
            st.info(f"**File:** {uploaded_file.name} | **Size:** {uploaded_file.size:,} bytes")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Load data with progress
            status_text.text("Loading and validating data...")
            df = cached_load_data(uploaded_file)
            progress_bar.progress(25)
            
            # Show basic file info
            st.subheader("📋 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                st.metric("Numeric Columns", len(numeric_cols))
            with col4:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            st.subheader("Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Show column information
            with st.expander("View Column Details"):
                st.write("**Column Types:**")
                col_types = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.values,
                    'Non-Null Count': df.notnull().sum().values,
                    'Null Count': df.isnull().sum().values
                })
                st.dataframe(col_types, use_container_width=True)
            
            # Get categorical columns
            categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
            
            # Parallel processing for expensive operations
            with ThreadPoolExecutor(max_workers=2) as executor:
                status_text.text("Analyzing columns...")
                col_mapping_future = executor.submit(cached_infer_columns, df)
                
                progress_bar.progress(50)
                
                # Get results
                col_mapping = col_mapping_future.result()
                progress_bar.progress(70)
                
                status_text.text("Analyzing patterns...")
                num_summary_future = executor.submit(cached_numeric_summary, df, numeric_cols)
                cat_summary_future = executor.submit(cached_categorical_summary, df, categorical_cols)
                
                num_summary = num_summary_future.result()
                cat_summary = cat_summary_future.result()
                progress_bar.progress(90)
            
            # Generate summary
            status_text.text("Generating insights...")
            summary = generate_summary(col_mapping, df, num_summary, cat_summary)
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
            
            # Display results
            display_results(df, col_mapping, numeric_cols, categorical_cols, num_summary, cat_summary, summary)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("""
            **Troubleshooting tips:**
            1. Ensure your file is not empty
            2. For CSV files, check the delimiter (comma, semicolon, tab)
            3. For Excel files, ensure data exists in the first sheet
            4. Try saving your file as UTF-8 encoded CSV
            """)

def validate_file(uploaded_file):
    """Validate the uploaded file before processing"""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file size
    if uploaded_file.size == 0:
        return False, "Uploaded file is empty"
    
    # Check file type
    if not uploaded_file.name.lower().endswith(('.csv', '.xlsx', '.xls')):
        return False, "Please upload a CSV or Excel file"
    
    return True, "File is valid"

def display_results(df, col_mapping, numeric_cols, categorical_cols, num_summary, cat_summary, summary):
    """Efficiently display all results with tabs for better organization"""
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "📈 Numeric Analysis", "📊 Categorical Analysis", "📁 Data"])
    
    with tab1:
        st.subheader("Generated Summary")
        st.text_area("Summary Text", summary, height=300, key="summary_text")
        st.download_button("📥 Download Summary", data=summary, file_name="data_summary.txt")
        
        st.subheader("Inferred Column Meanings")
        st.json(col_mapping, expanded=False)
    
    with tab2:
        if numeric_cols:
            st.subheader("📈 Numeric Data Overview")
            for col in numeric_cols:
                display_numeric_column(df, col, col_mapping, num_summary)
        else:
            st.info("No numeric columns found in the dataset.")
    
    with tab3:
        if categorical_cols:
            st.subheader("📊 Categorical Data Overview")
            for col in categorical_cols:
                display_categorical_column(df, col, col_mapping, cat_summary)
        else:
            st.info("No categorical columns found in the dataset.")
    
    with tab4:
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", df.shape[0])
            st.metric("Total Columns", df.shape[1])
        with col2:
            st.metric("Numeric Columns", len(numeric_cols))
            st.metric("Categorical Columns", len(categorical_cols))
        
        st.subheader("Full Dataset")
        st.dataframe(df, use_container_width=True)

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
    
    # Create bin labels
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

def display_numeric_column(df, col, col_mapping, num_summary):
    """Efficient display for a single numeric column using only Streamlit charts"""
    info = num_summary[col]
    concept = col_mapping.get(col, col)
    
    with st.container():
        st.markdown(f"### {concept} (`{col}`)")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Average", f"{info['mean']:,.2f}")
        with col2:
            st.metric("Total", f"{info['sum']:,.2f}")
        with col3:
            st.metric("Min", f"{info.get('min', 0):,.2f}")
        with col4:
            st.metric("Max", f"{info.get('max', 0):,.2f}")
        with col5:
            st.metric("Outliers", len(info['outliers']))
        
        # Visualizations in tabs
        tab1, tab2, tab3 = st.tabs(["📈 Trend", "📊 Distribution", "📋 Statistics"])
        
        with tab1:
            # Line chart for trend
            st.line_chart(df[col], use_container_width=True)
            
        with tab2:
            # Fixed histogram using our custom function
            data = df[col].dropna()
            if len(data) > 0:
                hist_df = create_simple_histogram(data, bins=15)
                if not hist_df.empty:
                    # Display as bar chart with proper labels
                    chart_data = hist_df.set_index('bin_range')['count']
                    st.bar_chart(chart_data, use_container_width=True)
                else:
                    st.info("No data available for histogram")
            else:
                st.info("No data available for histogram")
            
        with tab3:
            # Detailed statistics
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.write("**Basic Statistics:**")
                st.write(f"- Standard Deviation: {info.get('std', 0):,.2f}")
                st.write(f"- Skewness: {info.get('skewness', 0):.2f}")
                st.write(f"- Non-null values: {df[col].notnull().sum()}")
                st.write(f"- Null values: {df[col].isnull().sum()}")
            
            with stats_col2:
                if info['outliers']:
                    st.write("**Outlier Info:**")
                    st.write(f"- Count: {len(info['outliers'])}")
                    if info['outliers']:
                        st.write(f"- Min outlier: {min(info['outliers']):.2f}")
                        st.write(f"- Max outlier: {max(info['outliers']):.2f}")
                else:
                    st.write("**No outliers detected**")

def display_categorical_column(df, col, col_mapping, cat_summary):
    """Efficient display for a single categorical column using only Streamlit charts"""
    freq = cat_summary[col]
    concept = col_mapping.get(col, col)
    
    with st.container():
        st.markdown(f"### {concept} (`{col}`)")
        
        # Summary stats
        unique_count = len(freq)
        total_count = df[col].count()
        most_common_item, most_common_count = max(freq.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
        most_common_pct = (most_common_count / total_count * 100) if total_count > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Unique Values", unique_count)
        with col2:
            # Truncate long category names for display
            display_name = str(most_common_item)[:20] + '...' if len(str(most_common_item)) > 20 else str(most_common_item)
            st.metric("Most Common", display_name)
        with col3:
            st.metric("Occurrences", most_common_count)
        with col4:
            st.metric("Frequency", f"{most_common_pct:.1f}%")
        
        # Visualization - bar chart of top categories
        top_n = min(15, len(freq))
        top_categories = dict(sorted(freq.items(), 
                                   key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, 
                                   reverse=True)[:top_n])
        
        # Create proper dataframe for bar chart with string labels
        chart_data = {}
        for category, count in top_categories.items():
            # Ensure category is string and truncate if too long
            category_str = str(category)
            if len(category_str) > 30:
                category_str = category_str[:27] + '...'
            chart_data[category_str] = count
        
        # Display bar chart
        if chart_data:
            st.bar_chart(chart_data, use_container_width=True)
        else:
            st.info("No data available for chart")
        
        # Show frequency table in expander
        with st.expander("View Full Frequency Table"):
            freq_df = pd.DataFrame(list(freq.items()), columns=['Value', 'Count'])
            freq_df['Percentage'] = (freq_df['Count'] / total_count * 100).round(1)
            freq_df = freq_df.sort_values('Count', ascending=False)
            
            # Format values for display
            freq_df['Value'] = freq_df['Value'].apply(lambda x: str(x)[:50] + '...' if len(str(x)) > 50 else str(x))
            
            st.dataframe(freq_df, use_container_width=True, height=300)

if __name__ == "__main__":
    main()
