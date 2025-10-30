import pandas as pd
import numpy as np
import io

def load_data(file):
    """
    Robust data loading with comprehensive error handling
    """
    try:
        # Read file content first to check if it's empty
        file_content = file.getvalue()
        if len(file_content) == 0:
            raise ValueError("Uploaded file is empty")
        
        # Reset file pointer to beginning
        file.seek(0)
        
        if file.name.endswith('.csv'):
            # Try different encodings and delimiters
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    file.seek(0)  # Reset file pointer
                    # Try to detect delimiter and read
                    sample_content = file.read(1024).decode(encoding)
                    file.seek(0)
                    
                    # Common delimiters to try
                    delimiters = [',', ';', '\t', '|']
                    detected_delimiter = ','
                    
                    for delimiter in delimiters:
                        if sample_content.count(delimiter) > sample_content.count(','):
                            detected_delimiter = delimiter
                            break
                    
                    df = pd.read_csv(
                        file, 
                        delimiter=detected_delimiter,
                        encoding=encoding,
                        engine='python',  # More flexible parser
                        on_bad_lines='skip',  # Skip problematic lines
                        skip_blank_lines=True
                    )
                    
                    # Check if dataframe is empty
                    if df.empty:
                        continue
                        
                    break  # Successfully read file
                    
                except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
                    continue
            else:
                # If all encodings failed, try without specifying encoding
                file.seek(0)
                df = pd.read_csv(file, engine='python', on_bad_lines='skip')
                
        elif file.name.endswith(('.xlsx', '.xls')):
            # For Excel files, try reading first sheet
            df = pd.read_excel(file, engine='openpyxl')
            
            # If first sheet is empty, try other sheets
            if df.empty:
                excel_file = pd.ExcelFile(file, engine='openpyxl')
                sheet_names = excel_file.sheet_names
                
                for sheet in sheet_names:
                    df = pd.read_excel(file, sheet_name=sheet, engine='openpyxl')
                    if not df.empty:
                        break
        
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel file.")
        
        # Validate the loaded data
        if df.empty:
            raise ValueError("The file appears to be empty or contains no data")
        
        if df.shape[1] == 0:
            raise ValueError("No columns found in the file. Please check the file format.")
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Optimize dataframe memory usage
        df = optimize_dataframe(df)
        
        return df
        
    except Exception as e:
        # Provide more specific error messages
        if "No columns to parse from file" in str(e):
            raise ValueError(
                "Could not read any data from the file. This usually means:\n"
                "1. The file is empty\n"
                "2. The file format is incorrect\n"
                "3. The file uses an unusual encoding or delimiter\n"
                "4. The Excel file has no data in the first sheet\n\n"
                "Please check your file and try again."
            )
        elif "Unsupported format" in str(e) or "cannot read" in str(e):
            raise ValueError(
                "Unsupported file format or corrupted file. "
                "Please ensure you're uploading a valid CSV or Excel file."
            )
        else:
            raise ValueError(f"Error loading file: {str(e)}")

def optimize_dataframe(df):
    """Reduce memory usage of dataframe"""
    # Downcast numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        try:
            # Convert to appropriate numeric type
            if df[col].notna().all():  # No NaN values
                if (df[col] % 1 == 0).all():  # All integers
                    if df[col].min() >= 0:
                        df[col] = pd.to_numeric(df[col], downcast='unsigned')
                    else:
                        df[col] = pd.to_numeric(df[col], downcast='integer')
                else:
                    df[col] = pd.to_numeric(df[col], downcast='float')
        except:
            pass  # Keep original if conversion fails
    
    # Convert object columns to category if they have few unique values
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
            df[col] = df[col].astype('category')
    
    return df