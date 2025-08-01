# test_multiple_stocks.py
import logging
from config import ModelConfig, ApiConfig, SystemConfig
from utils import load_stock_list

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_multiple_stocks():
    """Test processing multiple stocks."""
    # Create a simple stock list
    stock_list = [
        {'symbol': 'AAPL', 'name': 'Apple Inc.'},
        {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
        {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
        {'symbol': 'META', 'name': 'Meta Platforms Inc.'}
    ]
    
    print(f"Testing with {len(stock_list)} stocks:")
    for stock in stock_list:
        print(f"  - {stock['symbol']}: {stock['name']}")
    
    # Test configuration
    api_config = ApiConfig.from_env()
    model_config = ModelConfig()
    system_config = SystemConfig(model=model_config, api=api_config)
    
    print(f"\nConfiguration:")
    print(f"  batch_processing_size: {system_config.batch_processing_size}")
    print(f"  max_processes: {system_config.max_processes}")
    
    # Calculate batches
    batch_size = system_config.batch_processing_size
    total_batches = (len(stock_list) + batch_size - 1) // batch_size
    
    print(f"\nBatch processing:")
    print(f"  Total batches: {total_batches}")
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch = stock_list[start_idx:end_idx]
        print(f"  Batch {batch_idx + 1}: {len(batch)} stocks")
        for stock in batch:
            print(f"    - {stock['symbol']}")

if __name__ == "__main__":
    test_multiple_stocks()