import urllib.request
import pandas as pd

def download_stock_list():
    # URLs for NASDAQ and NYSE stock listings
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    ]
    
    symbols = []
    names = []
    
    for url in urls:
        try:
            # Download and decode the data
            with urllib.request.urlopen(url) as response:
                data = response.read().decode('utf-8')
                lines = data.splitlines()
                
                # Skip header and footer lines (first and last line)
                for line in lines[1:-1]:
                    parts = line.strip().split('|')
                    
                    # Extract symbol and name
                    symbol = parts[0]
                    name = parts[1]
                    
                    symbols.append(symbol)
                    names.append(name)
                
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
    
    return {'symbol': symbols, 'name': names}

if __name__ == "__main__":
    # Download stock data
    stocks_data = download_stock_list()
    
    # Create DataFrame
    df = pd.DataFrame(stocks_data)
    
    # Save to CSV
    csv_filename = 'stock_list.csv'
    df.to_csv(csv_filename, index=False)
    
    # Print confirmation
    print(f"Created {csv_filename} with {len(df)} stocks")
    print("Sample stocks:")
    print(df.head(10))