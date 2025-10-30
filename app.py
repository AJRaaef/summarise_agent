import streamlit as st
import pandas as pd
import numpy as np
from src.data_loader import load_data
from src.column_understanding import infer_columns
from src.pattern_recognition import numeric_summary, categorical_summary
from src.summarizer import generate_summary
import matplotlib.pyplot as plt
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
    
    **Troubleshooting:**
    - Ensure file is not empty
    - Check file encoding (UTF-8 recommended)
    - Verify data exists in the first sheet (Excel)
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
                st.metric("Numeric Columns", len(df.select_dtypes(include=np.number).columns))
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
            
            # Parallel processing for expensive operations
            with ThreadPoolExecutor(max_workers=2) as executor:
                status_text.text("Analyzing columns...")
                col_mapping_future = executor.submit(cached_infer_columns, df)
                
                # Separate columns
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
                
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
            5. Check that your file contains actual data, not just headers
            """)

def display_results(df, col_mapping, numeric_cols, categorical_cols, num_summary, cat_summary, summary):
    """Efficiently display all results with tabs for better organization"""
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "📈 Numeric Analysis", "📊 Categorical Analysis", "📁 Data"])
    
    with tab1:
        st.subheader("Generated Summary")
        st.text_area("Summary Text", summary, height=300, key="summary_text")
        st.download_button("Download Summary", data=summary, file_name="summary.txt")
        
        st.subheader("Inferred Column Meanings")
        st.json(col_mapping, expanded=False)
    
    with tab2:
        if numeric_cols:
            st.subheader("📈 Numeric Data Overview")
            # Batch processing for numeric columns
            cols_per_row = 2
            for i in range(0, len(numeric_cols), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(numeric_cols[i:i+cols_per_row]):
                    with cols[j]:
                        display_numeric_column(df, col, col_mapping, num_summary)
        else:
            st.info("No numeric columns found in the dataset.")
    
    with tab3:
        if categorical_cols:
            st.subheader("📊 Categorical Data Overview")
            # Batch processing for categorical columns
            cols_per_row = 2
            for i in range(0, len(categorical_cols), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(categorical_cols[i:i+cols_per_row]):
                    with cols[j]:
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

def display_numeric_column(df, col, col_mapping, num_summary):
    """Efficient display for a single numeric column"""
    info = num_summary[col]
    concept = col_mapping.get(col, col)
    
    with st.container():
        st.markdown(f"**{concept}** (`{col}`)")
        
        # Key metrics in columns
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Average", f"{info['mean']:,.2f}")
        with m2:
            st.metric("Total", f"{info['sum']:,.2f}")
        with m3:
            st.metric("Outliers", len(info['outliers']))
        with m4:
            st.metric("Std Dev", f"{info.get('std', 0):,.2f}")
        
        # Visualizations in tabs
        viz_tab1, viz_tab2 = st.tabs(["📊 Distribution", "📈 Trend"])
        
        with viz_tab1:
            # Histogram using Streamlit
            fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
            df[col].hist(ax=ax_hist, bins=30, alpha=0.7)
            ax_hist.set_title(f"{concept} Distribution")
            ax_hist.set_ylabel('Frequency')
            st.pyplot(fig_hist)
            plt.close(fig_hist)
            
        with viz_tab2:
            # Line chart using Streamlit
            st.line_chart(df[col])

def display_categorical_column(df, col, col_mapping, cat_summary):
    """Efficient display for a single categorical column"""
    freq = cat_summary[col]
    concept = col_mapping.get(col, col)
    
    with st.container():
        st.markdown(f"**{concept}** (`{col}`)")
        
        # Summary stats
        unique_count = len(freq)
        total_count = df[col].count()
        most_common = max(freq.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Unique Values", unique_count)
        with col2:
            st.metric("Most Common", most_common[0])
        with col3:
            st.metric("Occurrences", most_common[1])
        
        # Visualization - bar chart of top categories
        top_n = 10
        top_categories = dict(sorted(freq.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)[:top_n])
        
        # Create bar chart using matplotlib
        if top_categories:
            fig, ax = plt.subplots(figsize=(8, 4))
            categories = list(top_categories.keys())
            counts = list(top_categories.values())
            
            # Truncate long category names
            short_categories = [str(cat)[:20] + '...' if len(str(cat)) > 20 else str(cat) for cat in categories]
            
            bars = ax.bar(short_categories, counts, alpha=0.7)
            ax.set_title(f"Top {len(top_categories)} Categories")
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

if __name__ == "__main__":
    main()