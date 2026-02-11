# 📈 AI-Powered Stock Forecasting System

A hybrid deep-learning ensemble that combines **LSTM**, **GRU**, and **CNN** architectures with attention mechanisms to forecast stock prices. The system integrates data from multiple sources — historical prices, news sentiment, social-media sentiment, macroeconomic indicators, and technical analysis — then generates detailed PDF reports.

> **Research Paper:** [Research_Paper.md](Research_Paper.md)

---

## ✨ Features

| Category | Details |
|----------|---------|
| **Multi-Source Data** | yFinance, Financial Modeling Prep, Alpha Vantage — with automatic fallback |
| **Sentiment Analysis** | News headlines (NewsAPI, Finnhub, GNews) and tweets via VADER |
| **Technical Indicators** | SMA (20/50/200), RSI, MACD, Bollinger Bands, and more |
| **Macro Factors** | GDP, unemployment, inflation, and VIX from FRED |
| **Model Architectures** | Standard LSTM + Attention, or advanced ensemble (LSTM, GRU, CNN-LSTM, hybrid) |
| **PDF Reporting** | Cover page, summary table, per-stock charts and metrics |
| **Parallel Processing** | Adaptive batch sizing with multiprocessing |
| **Caching** | Disk-based caching of API responses with configurable expiry |
| **Error Handling** | Retry with exponential backoff, graceful degradation |

---

## 📁 Project Structure

```
AIPSFS/
├── main.py                        # CLI entry point
├── aipsfs/                        # Core package
│   ├── config.py                  # ModelConfig, ApiConfig, SystemConfig
│   ├── exceptions.py              # Custom exception hierarchy
│   ├── data/
│   │   ├── fetcher.py             # DataFetcher — multi-source data collection
│   │   └── engineering.py         # FeatureEngineer — indicators & sentiment
│   ├── models/
│   │   ├── predictor.py           # StockPredictor — LSTM + Attention
│   │   └── advanced.py            # AdvancedStockPredictor — ensemble
│   ├── reporting/
│   │   ├── generator.py           # ReportGenerator — PDF creation
│   │   └── visualization.py       # Charting utilities
│   └── utils/
│       ├── helpers.py             # Logging, retry, caching, stock-list loader
│       └── validation.py          # Data & model input validation
├── tests/                         # Unit tests (pytest)
├── stocks.csv                     # Default stock list
├── generate_list_of_stocks.py     # Download NASDAQ/NYSE symbol lists
├── requirements.txt               # Runtime dependencies
├── requirements-test.txt          # Test dependencies (pytest)
└── .env.example                   # API key template
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- At least one API key (see [Configuration](#-configuration) below)

### 1. Clone & Create Virtual Environment

**Windows:**
```bash
git clone https://github.com/overcrash66/AIPSFS.git
cd AIPSFS
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
git clone https://github.com/overcrash66/AIPSFS.git
cd AIPSFS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

Then edit `.env` with your keys:

```env
NEWS_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
TWITTER_BEARER_TOKEN=your_token_here
FRED_API_KEY=your_key_here
FMP_API_KEY=your_key_here
ALPHA_VANTAGE_API_KEY=your_key_here
GNEWS_API_KEY=your_key_here
```

> **Tip:** The system works with any combination of keys. More keys = richer data.

### 4. Run the Analysis

```bash
# Standard LSTM model
python main.py --stocks stocks.csv

# Advanced ensemble models
python main.py --stocks stocks.csv --use-advanced

# Custom date range and filters
python main.py --stocks stocks.csv --use-advanced --start-date 2022-01-01 --end-date 2024-12-31 --top-n 5 --min-return 30.0
```

---

## ⚙️ CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--stocks` | Path to CSV file containing stock list | `stocks.csv` |
| `--start-date` | Start date for analysis (YYYY-MM-DD) | 2 years ago |
| `--end-date` | End date for analysis (YYYY-MM-DD) | Today |
| `--output` | Output directory for reports | `reports` |
| `--top-n` | Number of top stocks to include in report | `10` |
| `--min-return` | Minimum predicted return percentage | `50.0` |
| `--no-cache` | Disable caching of API responses | `False` |
| `--debug` | Enable debug logging | `False` |
| `--use-advanced` | Use advanced ensemble models | `False` |

---

## 🔑 Configuration

API keys are loaded from a `.env` file or environment variables. The following services are supported:

| Service | Key Variable | Get a Key |
|---------|-------------|-----------|
| NewsAPI | `NEWS_API_KEY` | [newsapi.org](https://newsapi.org/) |
| Finnhub | `FINNHUB_API_KEY` | [finnhub.io](https://finnhub.io/) |
| Twitter | `TWITTER_BEARER_TOKEN` | [developer.twitter.com](https://developer.twitter.com/) |
| FRED | `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| Financial Modeling Prep | `FMP_API_KEY` | [financialmodelingprep.com](https://site.financialmodelingprep.com/developer/docs) |
| Alpha Vantage | `ALPHA_VANTAGE_API_KEY` | [alphavantage.co](https://www.alphavantage.co/support/#api-key) |
| GNews | `GNEWS_API_KEY` | [gnews.io](https://gnews.io/) |

---

## 🧪 Testing

Run the full test suite:

```bash
# Using pytest directly
python -m pytest tests/ -v

```

Install test dependencies separately:

```bash
pip install -r requirements-test.txt
```

---

## 🔄 How It Works

### 1. Data Collection
The system fetches historical stock data from multiple sources with automatic fallback mechanisms. News, tweets, and macroeconomic data are collected and cached.

### 2. Feature Engineering
- Calculates technical indicators (SMA, RSI, MACD)
- Analyzes news and tweet sentiment using VADER
- Integrates macroeconomic indicators from FRED
- Scales and sequences data for model input

### 3. Model Training
- **Standard mode:** LSTM with attention mechanism, trained with early stopping
- **Advanced mode:** Ensemble of LSTM, GRU, CNN-LSTM, and hybrid architectures with checkpointing

### 4. Forecasting
Generates multi-step price predictions, calculates expected returns, and ranks stocks by predicted performance.

### 5. Reporting
Creates comprehensive PDF reports with cover pages, summary tables, price forecast charts, and per-stock analysis sections.

---

## 📄 Example Output

The system generates a PDF report containing:

- Cover page with analysis period
- Summary table of top-performing stocks
- Detailed analysis for each stock:
  - Current and predicted prices
  - Expected returns
  - Model performance metrics (MAE, RMSE, R²)
  - Price forecast charts

---

## 📜 License

This project is provided as-is for educational and research purposes.
