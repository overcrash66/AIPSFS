# exceptions.py
class StockAnalysisError(Exception):
    """Base exception for stock analysis errors"""
    pass

class DataFetchError(StockAnalysisError):
    """Exception for data fetching errors"""
    pass

class ModelTrainingError(StockAnalysisError):
    """Exception for model training errors"""
    pass

# Enhanced retry mechanism
def retry_with_backoff(max_retries=3, initial_delay=1, backoff_factor=2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logging.warning(f"Attempt {attempt+1}/{max_retries} failed for {func.__name__}: {str(e)}. Retrying in {delay}s...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logging.error(f"All {max_retries} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator

# Usage
@retry_with_backoff(max_retries=3)
def download_stock_data(symbol, start_date, end_date):
    # Current implementation
    pass