import urllib.request
import csv

def download_stock_list():
    # URLs for NASDAQ and NYSE stock listings
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    ]
    
    stocks = []
    
    for url in urls:
        try:
            # Download and decode the data
            response = urllib.request.urlopen(url)
            data = response.read().decode('utf-8')
            lines = data.splitlines()
            
            # Skip header and footer lines (first and last line)
            for line in lines[1:-1]:
                parts = line.strip().split('|')
                
                # Extract symbol and name
                if url.endswith("nasdaqlisted.txt"):
                    symbol = parts[0]
                    name = parts[1]
                else:  # otherlisted.txt (NYSE/AMEX)
                    symbol = parts[0]
                    name = parts[1]
                
                stocks.append((symbol, name))
                
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
    
    return stocks

def save_to_csv(stocks, filename='stock_list.csv'):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['symbol', 'name'])
        writer.writerows(stocks)
    print(f"Saved {len(stocks)} stocks to {filename}")

if __name__ == "__main__":
    stock_data = download_stock_list()
    save_to_csv(stock_data)