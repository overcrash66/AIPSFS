import logging
from datetime import datetime, timedelta
from data_fetcher import DataFetcher
from feature_engineering import FeatureEngineer
from config import ApiConfig

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_real_data():
    """Test with real data fetching."""
    api_config = ApiConfig.from_env()
    data_fetcher = DataFetcher(api_config)
    
    # Initialize feature engineer with api_config
    feature_engineer = FeatureEngineer(api_config=api_config)
    
    # Test with AAPL
    symbol = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Shorter period for testing
    
    print(f"Testing {symbol} from {start_date.date()} to {end_date.date()}")
    
    # Fetch data
    df = data_fetcher.fetch_stock_data(symbol, start_date, end_date)
    
    if df is None:
        print("Failed to fetch data")
        return
    
    print("Data fetched successfully!")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Index: {df.index}")
    
    # Process data
    processed_df = feature_engineer.process_stock_data(symbol, df, start_date, end_date)
    
    if processed_df is None:
        print("Failed to process data")
    else:
        print("Data processed successfully!")
        print(f"  Shape: {processed_df.shape}")
        print(f"  Columns: {processed_df.columns.tolist()}")

if __name__ == "__main__":
    test_real_data()