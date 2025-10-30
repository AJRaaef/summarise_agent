import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Cache the model globally
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
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
    "Score", "Rating", "Grade", "Rank", "Percentage", "Ratio"
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