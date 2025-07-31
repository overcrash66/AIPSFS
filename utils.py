# utils.py
import os
import logging
import functools
import pickle
import time
import random
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any

def setup_logging(debug: bool = False):
    """Configure logging for the application."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('stock_analysis.log'),
            logging.StreamHandler()
        ]
    )

def load_stock_list(file_path: str) -> List[Dict]:
    """Load stock list from CSV file."""
    try:
        df = pd.read_csv(file_path)
        if 'symbol' not in df.columns or 'name' not in df.columns:
            logging.error("Stock list CSV must contain 'symbol' and 'name' columns")
            return []
        
        return df[['symbol', 'name']].to_dict('records')
    
    except Exception as e:
        logging.error(f"Failed to load stock list: {str(e)}")
        return []

def retry_with_backoff(max_retries: int = 3, initial_delay: float = 1.0, 
                      backoff_factor: float = 2.0, exceptions: tuple = (Exception,)):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logging.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logging.error(f"All {max_retries} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator

def cache_result(cache_dir: str = "cache", expiry_days: int = 1):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            
            # Create a unique cache key
            cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            # Check if cache exists and is not expired
            if os.path.exists(cache_file):
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - file_time < timedelta(days=expiry_days):
                    try:
                        with open(cache_file, 'rb') as f:
                            result = pickle.load(f)
                        logging.debug(f"Cache hit for {func.__name__}")
                        return result
                    except Exception as e:
                        logging.warning(f"Failed to load cache for {func.__name__}: {str(e)}")
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                logging.debug(f"Cached result for {func.__name__}")
            except Exception as e:
                logging.warning(f"Failed to cache result for {func.__name__}: {str(e)}")
            
            return result
        return wrapper
    return decorator

def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: str) -> pd.DataFrame:
    """Safely merge two DataFrames with error handling."""
    if right.empty:
        return left
    
    try:
        # Reset indexes to ensure the merge column is a regular column
        left_copy = left.reset_index(drop=False)
        right_copy = right.reset_index(drop=False)
        
        # Ensure all columns are strings
        left_copy.columns = left_copy.columns.astype(str)
        right_copy.columns = right_copy.columns.astype(str)
        
        # Check if merge column exists
        if on not in left_copy.columns:
            logging.error(f"Merge column '{on}' missing in left DataFrame")
            return left
        if on not in right_copy.columns:
            logging.error(f"Merge column '{on}' missing in right DataFrame")
            return left
        
        # Handle duplicate columns
        duplicate_cols = set(left_copy.columns) & set(right_copy.columns) - {on}
        if duplicate_cols:
            suffix = "_right"
            rename_dict = {col: col + suffix for col in duplicate_cols}
            right_copy = right_copy.rename(columns=rename_dict)
        
        # Perform merge
        merged_df = pd.merge(left_copy, right_copy, on=on, how='left')
        
        # Restore original index if it was a date index
        if 'date' in left.index.names and 'date' in merged_df.columns:
            merged_df.set_index('date', inplace=True)
            merged_df.index.name = 'date'
        
        return merged_df
    
    except Exception as e:
        logging.error(f"Safe merge failed: {str(e)}")
        return left