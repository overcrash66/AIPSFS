# create_proper_stocks.py
import pandas as pd

# Create a proper stock list
stocks_data = {
    'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V'],
    'name': [
        'Apple Inc.',
        'Microsoft Corporation',
        'Alphabet Inc.',
        'Amazon.com Inc.',
        'Meta Platforms Inc.',
        'Tesla Inc.',
        'NVIDIA Corporation',
        'JPMorgan Chase & Co.',
        'Johnson & Johnson',
        'Visa Inc.'
    ]
}

df = pd.DataFrame(stocks_data)
df.to_csv('stocks_proper.csv', index=False)

print("Created stocks_proper.csv with the following stocks:")
print(df)