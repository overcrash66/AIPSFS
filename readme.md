# AI-Powered Stock Forecasting System

A comprehensive machine learning system for predicting stock prices using LSTM neural networks with attention mechanisms, enhanced with sentiment analysis from news and social media, macroeconomic indicators, and technical analysis.

## Features

- **Multi-Source Data Integration**: Fetches stock data from multiple sources (yFinance, Financial Modeling Prep) with automatic fallback
- **Sentiment Analysis**: Analyzes news headlines and tweets to gauge market sentiment
- **Technical Indicators**: Calculates key technical indicators (SMA, RSI, MACD) for better predictions
- **Macroeconomic Factors**: Incorporates economic indicators (GDP, unemployment, inflation, VIX)
- **Advanced LSTM Model**: Uses an LSTM neural network with attention mechanism for accurate forecasting
- **Comprehensive Reporting**: Generates detailed PDF reports with charts and analysis
- **Parallel Processing**: Efficiently processes multiple stocks in parallel
- **Caching System**: Caches API responses to reduce rate limit issues
- **Robust Error Handling**: Implements retry mechanisms and graceful error recovery

## Installation

### Prerequisites

- Python 3.9 or higher
- API keys for various services (see Configuration section)
     
API Key Sources 

     News API: Get a free API key from https://newsapi.org/
     Finnhub: Sign up at https://finnhub.io/  for a free API key
     Twitter: Apply for a developer account at Twitter Developer Platform https://developer.twitter.com/
     FRED: Get an API key from Federal Reserve Economic Data (FRED)  https://fred.stlouisfed.org/docs/api/api_key.html
     Financial Modeling Prep: Get a free API key from Financial Modeling Prep https://site.financialmodelingprep.com/developer/docs
     


# Setup python venv

```
py -3.10 -m venv venv
venv\Scripts\activate
```

# Install dependencies

```
pip install -r requirements.txt
```

# Set up environment variables



# Edit .env with your API keys
Example using CMD:

```
set NEWS_API_KEY=[Add_ME]
set FINNHUB_API_KEY=[Add_ME]
set TWITTER_BEARER_TOKEN=[Add_ME]
set ALPHA_VANTAGE_API_KEY=[Add_ME]
set FMP_API_KEY=[Add_ME]
set FRED_API_KEY=[Add_ME]
```

OR using .env FILE:

```
# API Keys
NEWS_API_KEY=your_news_api_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
FRED_API_KEY=your_fred_api_key_here
FMP_API_KEY=your_fmp_api_key_here

# Logging
LOG_LEVEL=INFO
```

# Run the analysis

```
python main.py --stocks stocks.csv --start-date 2022-01-01 --end-date 2023-12-31 --top-n 10
```

| Option       | Description                               | Default     |
| ------------ | ----------------------------------------- | ----------- |
| --stocks     | Path to CSV file containing stock list    | stocks.csv  |
| --start-date | Start date for analysis (YYYY-MM-DD)      | 2 years ago |
| --end-date   | End date for analysis (YYYY-MM-DD)        | Today       |
| --output     | Output directory for reports              | reports     |
| --top-n      | Number of top stocks to include in report | 10          |
| --min-return | Minimum predicted return percentage       | 50.0        |
| --no-cache   | Disable caching of API responses          | False       |
| --debug      | Enable debug logging                      | False       |


# Project Structure

stock_analysis/
├── main.py                 # Main application entry point
├── config.py               # Configuration classes
├── data_fetcher.py         # Data fetching module
├── feature_engineering.py   # Feature engineering module
├── model.py                # LSTM model with attention
├── reporter.py             # PDF report generation
├── utils.py                # Utility functions
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── stocks.csv             # Example stock list
└── README.md              # This file


## How It Works

### 1. Data Collection
- The system fetches historical stock data from multiple sources with automatic fallback mechanisms.

### 2. Feature Engineering
- Calculates technical indicators (SMA, RSI, MACD)
- Fetches and analyzes news articles for sentiment
- Collects and processes tweets for social sentiment
- Incorporates macroeconomic indicators

### 3. Model Training
- Prepares sequences for LSTM input
- Builds an LSTM model with attention mechanism
- Trains the model with early stopping to prevent overfitting
- Evaluates model performance with multiple metrics

### 4. Forecasting
- Generates future price predictions
- Calculates expected returns
- Filters top-performing stocks based on predicted returns

### 5. Reporting
- Creates comprehensive PDF reports
- Includes summary tables and price charts
- Provides detailed analysis for each stock

## Example Output

The system generates a PDF report containing:

- Cover page with analysis period
- Summary table of top-performing stocks
- Detailed analysis for each stock including:
  - Current and predicted prices
  - Expected returns
  - Model performance metrics
  - Price forecast charts
