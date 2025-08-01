import logging
import pandas as pd
from utils import load_stock_list

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_stock_list_detailed():
    """Test loading stock list with detailed debugging."""
    file_path = input("Enter the path to your stock list file: ")
    
    print(f"\n=== Testing Stock List Loading ===")
    print(f"File: {file_path}")
    
    # Test raw pandas reading
    print("\n--- Raw Pandas Reading ---")
    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 10 rows:")
        print(df.head(10).to_string())
        
        print("\nLast 5 rows:")
        print(df.tail(5).to_string())
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Test our function
    print("\n--- Our Load Function ---")
    stocks = load_stock_list(file_path)
    
    if stocks:
        print(f"\nSuccessfully loaded {len(stocks)} stocks:")
        for i, stock in enumerate(stocks):
            print(f"  {i+1}. {stock['symbol']} - {stock['name']}")
    else:
        print("No stocks loaded!")
    
    # Show file info
    print("\n--- File Information ---")
    import os
    if os.path.exists(file_path):
        print(f"File size: {os.path.getsize(file_path)} bytes")
        print(f"Last modified: {os.path.getmtime(file_path)}")
    else:
        print("File does not exist!")

if __name__ == "__main__":
    test_stock_list_detailed()