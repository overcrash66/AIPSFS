import logging
import pandas as pd
from datetime import datetime, timedelta
from feature_engineering import FeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_data_processing():
    """Test data processing with sample data."""
    # Create sample data similar to what yFinance returns
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    sample_data = {
        'Open': [150.0 + i * 0.1 for i in range(len(dates))],
        'High': [152.0 + i * 0.1 for i in range(len(dates))],
        'Low': [148.0 + i * 0.1 for i in range(len(dates))],
        'Close': [151.0 + i * 0.1 for i in range(len(dates))],
        'Volume': [1000000 + i * 1000 for i in range(len(dates))]
    }
    
    df = pd.DataFrame(sample_data, index=dates)
    
    print("Sample DataFrame:")
    print(f"  Shape: {df.shape}")
    print(f"  Index: {df.index}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Head:\n{df.head()}")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Test date standardization
    print("\nTesting date standardization...")
    standardized_df = feature_engineer._standardize_dates(df.copy())
    
    if standardized_df is not None:
        print("Date standardization successful!")
        print(f"  Shape: {standardized_df.shape}")
        print(f"  Index: {standardized_df.index}")
        print(f"  Head:\n{standardized_df.head()}")
    else:
        print("Date standardization failed!")
    
    # Test technical indicators
    print("\nTesting technical indicators...")
    df_with_indicators = feature_engineer.calculate_technical_indicators(df.copy())
    print("Technical indicators calculated!")
    print(f"  New columns: {set(df_with_indicators.columns) - set(df.columns)}")
    print(f"  Head:\n{df_with_indicators.head()}")

if __name__ == "__main__":
    test_data_processing()