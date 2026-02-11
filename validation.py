# validation.py
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def validate_stock_data(df: pd.DataFrame) -> bool:
    """Validate that stock data has required columns and proper format"""
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Check if DataFrame is empty
    if df.empty:
        logging.error("Stock data DataFrame is empty")
        return False
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check if numeric columns are actually numeric
    non_numeric = [col for col in required_columns if not is_numeric_dtype(df[col])]
    if non_numeric:
        logging.error(f"Non-numeric columns found: {non_numeric}")
        return False
    
    # Check for NaN values in critical columns
    if df['Close'].isna().any():
        logging.error("NaN values found in Close prices")
        return False
    
    return True

def validate_model_input(X: np.ndarray, y: np.ndarray) -> bool:
    """Validate model input data"""
    if X is None or y is None:
        logging.error("Model input X or y is None")
        return False
    
    if len(X) == 0 or len(y) == 0:
        logging.error("Model input X or y is empty")
        return False
    
    if len(X) != len(y):
        logging.error(f"Length mismatch: X has {len(X)} samples, y has {len(y)} samples")
        return False
    
    return True