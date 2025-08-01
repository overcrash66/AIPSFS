import logging
from datetime import datetime, timedelta
from config import ApiConfig, ModelConfig, SystemConfig
from data_fetcher import DataFetcher

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_data_fetch():
    """Test data fetching for a single stock."""
    api_config = ApiConfig.from_env()
    data_fetcher = DataFetcher(api_config)
    
    # Test with AAPL
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"Testing data fetch for AAPL from {start_date.date()} to {end_date.date()}")
    
    # Try yfinance
    df = data_fetcher._fetch_from_yfinance('AAPL', start_date, end_date)
    if df is not None:
        print("yfinance successful")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample data:\n{df.head()}")
    else:
        print("yfinance failed")
    
    # Try FMP
    if api_config.fmp_api_key:
        df = data_fetcher._fetch_from_fmp('AAPL', start_date, end_date)
        if df is not None:
            print("FMP successful")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Sample data:\n{df.head()}")
        else:
            print("FMP failed")
    
    # Try Alpha Vantage
    if api_config.alpha_vantage_api_key:
        df = data_fetcher._fetch_from_alpha_vantage('AAPL', start_date, end_date)
        if df is not None:
            print("Alpha Vantage successful")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Sample data:\n{df.head()}")
        else:
            print("Alpha Vantage failed")

if __name__ == "__main__":
    test_data_fetch()