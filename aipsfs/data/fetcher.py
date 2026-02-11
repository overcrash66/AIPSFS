# data_fetcher.py
from datetime import datetime, date, timedelta
import os
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import finnhub
import tweepy
from typing import Dict, List, Optional, Tuple
from fredapi import Fred
from functools import lru_cache
import re 
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from aipsfs.config import ApiConfig
from aipsfs.utils.helpers import retry_with_backoff, cache_result

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class DataFetcher:
    """Handles all data fetching operations with caching and retry logic."""
    
    def __init__(self, api_config: ApiConfig, cache_enabled: bool = True):
        self.api_config = api_config
        self.cache_enabled = cache_enabled
        
        # Initialize API clients only if keys are available
        self.finnhub_client = finnhub.Client(api_key=api_config.finnhub_api_key) if api_config.finnhub_api_key else None
        self.fred = Fred(api_key=api_config.fred_api_key) if api_config.fred_api_key else None
        
        # Initialize VADER sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()
        
        # Configure Twitter client
        self.twitter_client = None
        if api_config.twitter_bearer_token:
            try:
                self.twitter_client = tweepy.Client(bearer_token=api_config.twitter_bearer_token, wait_on_rate_limit=False)
            except Exception as e:
                logging.warning(f"Failed to initialize Twitter client: {str(e)}")
    
    @retry_with_backoff(max_retries=3)
    def fetch_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch historical stock data with multiple fallbacks."""
        # Try yfinance first
        df = self._fetch_from_yfinance(symbol, start_date, end_date)
        if df is not None:
            return df
        
        # Fallback to FMP API if available
        if self.api_config.fmp_api_key:
            df = self._fetch_from_fmp(symbol, start_date, end_date)
            if df is not None:
                return df
        
        # Fallback to Alpha Vantage if available
        if self.api_config.alpha_vantage_api_key:
            df = self._fetch_from_alpha_vantage(symbol, start_date, end_date)
            if df is not None:
                return df
        
        return None
    
    def _fetch_from_yfinance(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch stock data from yfinance."""
        try:
            logging.info(f"Fetching {symbol} data from yfinance")
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                timeout=15,
                auto_adjust=False
            )
            
            if df.empty:
                logging.warning(f"yfinance returned empty data for {symbol}")
                return None
            
            # Debug: Print DataFrame info
            logging.info(f"yfinance DataFrame for {symbol}:")
            logging.info(f"  Shape: {df.shape}")
            logging.info(f"  Columns: {df.columns.tolist()}")
            logging.info(f"  Index type: {type(df.index)}")
            logging.info(f"  Index name: {df.index.name}")
            logging.info(f"  Head:\n{df.head()}")
            
            # Handle MultiIndex columns (yfinance returns ('Close','AAPL'))
            if isinstance(df.columns, pd.MultiIndex):
                # Drop the ticker level — keep only the metric name
                df.columns = [col[0] for col in df.columns.values]
                logging.info(f"Simplified MultiIndex columns: {df.columns.tolist()}")
            
            # Clean column names (remove spaces)
            df.columns = [col.replace(' ', '') for col in df.columns]
            
            # Normalise common column-name variations to canonical OHLCV
            canonical_map = {
                'close': 'Close', 'open': 'Open', 'high': 'High',
                'low': 'Low', 'volume': 'Volume', 'adjclose': 'AdjClose',
            }
            rename_map = {}
            for col in df.columns:
                lower = col.lower()
                if lower in canonical_map and col != canonical_map[lower]:
                    rename_map[col] = canonical_map[lower]
            if rename_map:
                df.rename(columns=rename_map, inplace=True)
                logging.info(f"Renamed columns: {rename_map}")
            
            # Ensure we have a Close column
            if 'Close' not in df.columns:
                close_candidates = [c for c in df.columns if 'close' in c.lower()]
                if close_candidates:
                    df['Close'] = df[close_candidates[0]]
                    logging.info(f"Using '{close_candidates[0]}' as 'Close' for {symbol}")
                else:
                    logging.warning(f"No Close price found for {symbol}")
                    return None
            
            # Ensure all price columns are numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logging.info(f"Successfully fetched {len(df)} rows from yfinance for {symbol}")
            return df
            
        except Exception as e:
            logging.warning(f"yfinance fetch failed for {symbol}: {str(e)}")
            return None
    
    @retry_with_backoff(max_retries=2, initial_delay=120.0, backoff_factor=3.0)
    def _fetch_from_fmp(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch stock data from FMP API as fallback."""
        try:
            logging.info(f"Fetching {symbol} data from FMP API")
            url = (
                f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
                f"?from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}"
                f"&apikey={self.api_config.fmp_api_key}"
            )
            
            response = requests.get(url, timeout=15)
            
            # Handle rate limiting
            if response.status_code == 429:
                logging.warning(f"FMP rate limit reached for {symbol}")
                # Wait much longer before retrying
                time.sleep(300)  # Wait 5 minutes
                raise requests.exceptions.HTTPError("Rate limit exceeded")
            
            response.raise_for_status()
            data = response.json()
            
            if 'historical' not in data or not data['historical']:
                logging.warning(f"FMP returned no data for {symbol}")
                return None
            
            df = pd.DataFrame(data['historical'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Rename columns to match yfinance
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'adjClose': 'Adj Close',
                'volume': 'Volume'
            }, inplace=True)
            
            # Ensure we have a Close column
            if 'Close' not in df.columns:
                if 'Adj Close' in df.columns:
                    df['Close'] = df['Adj Close']
                else:
                    logging.warning(f"No Close price found for {symbol}")
                    return None
            
            logging.info(f"Successfully fetched {len(df)} rows from FMP for {symbol}")
            return df
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                logging.warning(f"FMP rate limit for {symbol}: {str(e)}")
            else:
                logging.warning(f"FMP HTTP error for {symbol}: {str(e)}")
            return None
        except Exception as e:
            logging.warning(f"FMP fetch failed for {symbol}: {str(e)}")
            return None
    
    def _fetch_from_alpha_vantage(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch stock data from Alpha Vantage as a backup."""
        if not self.api_config.alpha_vantage_api_key:
            return None
            
        try:
            logging.info(f"Fetching {symbol} data from Alpha Vantage")
            url = (
                f"https://www.alphavantage.co/query"
                f"?function=TIME_SERIES_DAILY"
                f"&symbol={symbol}"
                f"&outputsize=full"
                f"&apikey={self.api_config.alpha_vantage_api_key}"
            )
            
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                logging.warning(f"Alpha Vantage returned no data for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
            df.index = pd.to_datetime(df.index)
            df.columns = [col.split('. ')[1] for col in df.columns]
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            if df.empty:
                logging.warning(f"No data in date range for {symbol}")
                return None
            
            # Rename columns to match yfinance
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            logging.info(f"Successfully fetched {len(df)} rows from Alpha Vantage for {symbol}")
            return df
            
        except Exception as e:
            logging.warning(f"Alpha Vantage fetch failed for {symbol}: {str(e)}")
            return None

    @cache_result(cache_dir="cache/news", expiry_days=1)
    def fetch_news_data(self, symbol: str, start_date, end_date) -> pd.DataFrame:
        """Fetch news data with multiple fallback APIs."""
        news_data = pd.DataFrame()
        today = datetime.now().date()

        # Calculate API-specific date ranges
        finnhub_min_date = today - timedelta(days=365)
        newsapi_min_date = today - timedelta(days=30)
        gnews_min_date = today - timedelta(days=30)  # GNews free tier limit

        # Ensure dates are date objects
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()

        # 1. First try: Finnhub API
        if self.api_config.finnhub_api_key and self.finnhub_client:
            try:
                finnhub_start = max(start_date, finnhub_min_date)
                if finnhub_start <= end_date:
                    logging.info(f"Fetching Finnhub news for {symbol}")
                    news = self.finnhub_client.company_news(
                        symbol,
                        _from=finnhub_start.strftime('%Y-%m-%d'),
                        to=end_date.strftime('%Y-%m-%d')
                    )
                    if news:
                        df = pd.DataFrame(news)
                        df['date'] = pd.to_datetime(df['datetime'], unit='s').dt.date
                        news_data = df[['date', 'headline', 'summary', 'url', 'source']]
                        logging.info(f"Finnhub: Found {len(news_data)} news items")
            except Exception as e:
                logging.warning(f"Finnhub news error: {str(e)}")

        # 2. Second try: NewsAPI
        if news_data.empty and self.api_config.news_api_key:
            try:
                newsapi_start = max(start_date, newsapi_min_date)
                if newsapi_start <= end_date:
                    logging.info(f"Fetching NewsAPI for {symbol}")
                    url = (f"https://newsapi.org/v2/everything?q={symbol}"
                        f"&from={newsapi_start.strftime('%Y-%m-%d')}"
                        f"&to={end_date.strftime('%Y-%m-%d')}"
                        f"&sortBy=popularity&apiKey={self.api_config.news_api_key}")
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        news = response.json().get('articles', [])
                        if news:
                            df = pd.DataFrame(news)
                            df['date'] = pd.to_datetime(df['publishedAt']).dt.date
                            news_data = df[['date', 'title', 'description', 'url', 'source']]
                            news_data.columns = ['date', 'headline', 'summary', 'url', 'source']
                            logging.info(f"NewsAPI: Found {len(news_data)} news items")
                    else:
                        logging.warning(f"NewsAPI error: {response.status_code}")
            except Exception as e:
                logging.warning(f"NewsAPI error: {str(e)}")

        # 3. Third try: GNews API (new fallback)
        if news_data.empty and self.api_config.gnews_api_key:
            try:
                gnews_start = max(start_date, gnews_min_date)
                if gnews_start <= end_date:
                    logging.info(f"Fetching GNews for {symbol}")
                    url = "https://gnews.io/api/v4/search"
                    params = {
                        'q': symbol,
                        'lang': 'en',
                        'from': gnews_start.strftime('%Y-%m-%d'),
                        'to': end_date.strftime('%Y-%m-%d'),
                        'apikey': self.api_config.gnews_api_key,
                        'max': 100  # Max results per request
                    }
                    response = requests.get(url, params=params, timeout=15)
                    if response.status_code == 200:
                        data = response.json()
                        articles = data.get('articles', [])
                        if articles:
                            rows = []
                            for article in articles:
                                rows.append({
                                    'date': pd.to_datetime(article['publishedAt']).date(),
                                    'headline': article['title'],
                                    'summary': article['description'],
                                    'url': article['url'],
                                    'source': article['source']['name']
                                })
                            news_data = pd.DataFrame(rows)
                            logging.info(f"GNews: Found {len(news_data)} news items")
                    else:
                        logging.warning(f"GNews error: {response.status_code}")
            except Exception as e:
                logging.warning(f"GNews fetch error: {str(e)}")

        # Process news if available
        if not news_data.empty:
            # (Existing processing code remains the same)
            return news_data
        else:
            logging.info(f"No news found for {symbol}")
            return pd.DataFrame()
    
    
    @cache_result(cache_dir="cache/macro", expiry_days=7)
    def fetch_macro_data(self, start_date, end_date) -> pd.DataFrame:
        """Fetches macroeconomic data from FRED API."""
        macro_data = pd.DataFrame()
        if not self.api_config.fred_api_key or not self.fred:
            logging.info("FRED API not configured. Skipping macro data fetch.")
            return macro_data

        try:
            # Ensure start_date and end_date are in the correct format
            if isinstance(start_date, datetime):
                start_date_str = start_date.strftime('%Y-%m-%d')
            else:
                start_date_str = start_date.strftime('%Y-%m-%d')
                
            if isinstance(end_date, datetime):
                end_date_str = end_date.strftime('%Y-%m-%d')
            else:
                end_date_str = end_date.strftime('%Y-%m-%d')

            indicators = {
                'GDP': 'GDPC1',
                'UNRATE': 'UNRATE',
                'CPIAUCSL': 'CPIAUCSL',
                'FEDFUNDS': 'FEDFUNDS',
                'VIX': 'VIXCLS'
            }

            dfs = []
            for name, code in indicators.items():
                try:
                    series = self.fred.get_series(
                        code,
                        observation_start=start_date_str,
                        observation_end=end_date_str
                    )
                    if not series.empty:
                        series.name = name
                        dfs.append(series)
                except Exception as e:
                    logging.warning(f"FRED error for {name} ({code}): {str(e)}")

            if dfs:
                macro_data = pd.concat(dfs, axis=1)
                macro_data = macro_data.resample('D').ffill().bfill()
                #macro_data.index = macro_data.index.date
                macro_data = macro_data.reset_index()
                macro_data.rename(columns={'index': 'date'}, inplace=True)

                # Convert to date only (without time)
                macro_data['date'] = macro_data['date'].dt.date
                logging.info(f"Fetched {len(macro_data)} macro data points")
        except Exception as e:
            logging.error(f"Macro data fetching failed: {str(e)}")

        return macro_data
    
    @cache_result(cache_dir="cache/tweets", expiry_days=1)
    def fetch_tweet_data(self, symbol: str, start_date, end_date) -> pd.DataFrame:
        """Fetch tweets for a given stock symbol using the Twitter API v2."""
        tweets = pd.DataFrame()
        if not self.api_config.twitter_bearer_token:
            logging.info("Twitter token not configured. Skipping tweet data fetch.")
            return tweets

        try:
            client = tweepy.Client(bearer_token=self.api_config.twitter_bearer_token, wait_on_rate_limit=False)
            # Use plain-text query — the $cashtag operator requires Pro/Enterprise tier
            query = f"{symbol} stock lang:en -is:retweet"
            
            # Ensure start_date and end_date are datetime objects
            if isinstance(start_date, date) and not isinstance(start_date, datetime):
                start_date = datetime.combine(start_date, datetime.min.time())
            if isinstance(end_date, date) and not isinstance(end_date, datetime):
                end_date = datetime.combine(end_date, datetime.max.time())
            
            # Twitter API v2 recent search has a 7-day limit
            effective_start_time = max(start_date, datetime.now() - timedelta(days=6))
            effective_end_time = end_date

            if effective_start_time > effective_end_time:
                logging.warning(f"Twitter search date range invalid for {symbol}. Skipping.")
                return tweets

            start_utc = effective_start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_utc = effective_end_time.strftime('%Y-%m-%dT%H:%M:%SZ')

            tweets_list = []
            next_token = None
            max_results = 100
            max_pages = 5

            logging.info(f"Fetching tweets for {symbol} from {effective_start_time.date()} to {effective_end_time.date()}")

            for _ in range(max_pages):
                try:
                    response = client.search_recent_tweets(
                        query=query,
                        max_results=max_results,
                        start_time=start_utc,
                        end_time=end_utc,
                        tweet_fields=['created_at', 'public_metrics', 'text'],
                        next_token=next_token
                    )

                    if response.data:
                        for tweet in response.data:
                            tweets_list.append({
                                'date': tweet.created_at.date(),
                                'text': tweet.text,
                                'likes': tweet.public_metrics.get('like_count', 0),
                                'retweets': tweet.public_metrics.get('retweet_count', 0)
                            })

                    if 'next_token' in response.meta:
                        next_token = response.meta['next_token']
                    else:
                        break
                except tweepy.TooManyRequests:
                    logging.warning(f"Twitter rate limit reached for {symbol}. Returning partial data.")
                    break  # Return whatever data we have instead of sleeping
                except Exception as e:
                    logging.warning(f"Twitter API error for {symbol}: {str(e)}")
                    break

            if tweets_list:
                tweets = pd.DataFrame(tweets_list)
                # Initialize VADER if not already done
                if not hasattr(self, 'vader'):
                    self.vader = SentimentIntensityAnalyzer()
                tweets['sentiment'] = tweets['text'].apply(
                    lambda x: self.vader.polarity_scores(str(x))['compound'])
                logging.info(f"Fetched {len(tweets)} tweets for {symbol}")
        except Exception as e:
            logging.error(f"Twitter fetch failed for {symbol}: {str(e)}")

        return tweets