import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from feature_engineering import FeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_date_standardization():
    """Test date standardization with a DataFrame that has a DatetimeIndex."""
    # Create a sample DataFrame with DatetimeIndex
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    data = {
        'Open': np.random.uniform(100, 200, len(dates)),
        'High': np.random.uniform(100, 200, len(dates)),
        'Low': np.random.uniform(100, 200, len(dates)),
        'Close': np.random.uniform(100, 200, len(dates)),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }
    
    df = pd.DataFrame(data, index=dates)
    
    print("Original DataFrame:")
    print(f"  Shape: {df.shape}")
    print(f"  Index type: {type(df.index)}")
    print(f"  Index name: {df.index.name}")
    print(f"  Head:\n{df.head()}")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Test date standardization
    print("\nTesting date standardization...")
    standardized_df = feature_engineer._standardize_dates(df.copy())
    
    if standardized_df is not None:
        print("Date standardization successful!")
        print(f"  Shape: {standardized_df.shape}")
        print(f"  Index type: {type(standardized_df.index)}")
        print(f"  Index name: {standardized_df.index.name}")
        print(f"  Head:\n{standardized_df.head()}")
    else:
        print("Date standardization failed!")

if __name__ == "__main__":
    test_date_standardization()