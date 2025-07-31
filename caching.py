# caching.py
import functools
import pickle
import os
from datetime import datetime, timedelta

def cache_result(cache_dir="cache", expiry_days=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a unique cache key based on function name and arguments
            cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            # Check if cache exists and is not expired
            if os.path.exists(cache_file):
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - file_time < timedelta(days=expiry_days):
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            
            # Save result to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            return result
        return wrapper
    return decorator

# Usage
@cache_result(cache_dir="api_cache", expiry_days=1)
def fetch_news(ticker, start_date, end_date):
    # Current implementation
    pass

# Optimized multiprocessing
def process_stocks_in_parallel(stock_list, config):
    """Process stocks in optimized parallel batches"""
    results = []
    
    # Process in batches to avoid API rate limits
    for i in tqdm(range(0, len(stock_list), config.batch_processing_size)):
        batch = stock_list[i:i + config.batch_processing_size]
        
        # Prepare arguments for each task
        tasks = [
            (symbol, name, start_date, end_date, config.model.lookback_days, 
             config.model.epochs, config.model.batch_size)
            for symbol, name in batch
        ]
        
        # Process batch in parallel
        with Pool(processes=min(config.max_processes, len(batch))) as pool:
            batch_results = pool.map(analyze_stock_task, tasks)
            results.extend([r for r in batch_results if r is not None])
        
        # Random delay between batches to avoid rate limits
        if i + config.batch_processing_size < len(stock_list):
            delay = random.uniform(
                config.batch_processing_delay_min, 
                config.batch_processing_delay_max
            )
            logging.info(f"Waiting {delay:.1f}s before next batch...")
            time.sleep(delay)
    
    return results