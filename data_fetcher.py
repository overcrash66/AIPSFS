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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from fredapi import Fred
from functools import lru_cache
import re 
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from config import ApiConfig
from utils import retry_with_backoff, cache_result

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
                self.twitter_client = tweepy.Client(bearer_token=api_config.twitter_bearer_token, wait_on_rate_limit=True)
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
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip() for col in df.columns.values]
                logging.info(f"Flattened MultiIndex columns: {df.columns.tolist()}")
            
            # Ensure we have a Close column
            if 'Close' not in df.columns:
                # Try to find any column that might contain close prices
                close_candidates = [col for col in df.columns if 'close' in col.lower()]
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
            
            # Clean column names
            df.columns = [col.replace(' ', '') for col in df.columns]
            
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
        """Fetch news data from Finnhub and NewsAPI."""
        news_data = pd.DataFrame()
        today = datetime.now().date()

        # Calculate API-specific date ranges
        finnhub_min_date = today - timedelta(days=365)  # Finnhub public plan 1 year limit
        newsapi_min_date = today - timedelta(days=30)   # NewsAPI free plan 1 month limit

        # Ensure start_date and end_date are date objects
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()

        # Determine effective start dates for each API
        finnhub_start = max(start_date, finnhub_min_date)
        newsapi_start = max(start_date, newsapi_min_date)

        # Finnhub API (up to 1 year history)
        if self.api_config.finnhub_api_key and self.finnhub_client:
            try:
                if finnhub_start <= end_date:
                    logging.info(f"Fetching Finnhub news for {symbol} from {finnhub_start} to {end_date}")
                    news = self.finnhub_client.company_news(symbol,
                                                        _from=finnhub_start.strftime('%Y-%m-%d'),
                                                        to=end_date.strftime('%Y-%m-%d'))
                    if news:
                        df = pd.DataFrame(news)
                        df['date'] = pd.to_datetime(df['datetime'], unit='s').dt.date
                        df = df[['date', 'headline', 'summary', 'url', 'source']]
                        news_data = df.copy()
                        logging.info(f"Finnhub: Found {len(news_data)} news items for {symbol}")
            except Exception as e:
                logging.warning(f"Finnhub news error for {symbol}: {str(e)}")

        # NewsAPI (up to 1 month history)
        if news_data.empty and self.api_config.news_api_key:
            try:
                logging.info(f"Fetching NewsAPI for {symbol} from {newsapi_start} to {end_date}")
                url = f"https://newsapi.org/v2/everything?q={symbol}&from={newsapi_start.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&sortBy=popularity&apiKey={self.api_config.news_api_key}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    news = response.json().get('articles', [])
                    if news:
                        df = pd.DataFrame(news)
                        df['date'] = pd.to_datetime(df['publishedAt']).dt.date
                        df = df[['date', 'title', 'description', 'url', 'source']]
                        df.columns = ['date', 'headline', 'summary', 'url', 'source']
                        news_data = df.copy()
                        logging.info(f"NewsAPI: Found {len(news_data)} news items for {symbol}")
                else:
                    logging.warning(f"NewsAPI error for {symbol}: {response.status_code} - {response.text[:100]}")
            except Exception as e:
                logging.warning(f"NewsAPI error for {symbol}: {str(e)}")

        # Process news if available
        if not news_data.empty:
            news_data['clean_text'] = news_data['headline'] + " " + news_data['summary'].fillna('')
            news_data['clean_text'] = news_data['clean_text'].apply(
                lambda x: re.sub(r'[^\w\s]', '', str(x).lower())
            )

            # Event detection
            event_keywords = {
                'earnings': ['earnings', 'results', 'eps', 'profit', r'q[1-4]', 'quarterly'],
                'merger': ['acquisition', 'merge', 'takeover', 'buyout', 'acquire'],
                'regulation': ['regulator', 'doj', 'ftc', 'lawsuit', 'investigation', 'sec'],
                'product': ['launch', 'release', 'new product', 'announce', 'unveil'],
                'macro': ['fed', 'interest rate', 'inflation', 'cpi', 'unemployment']
            }

            for event, keywords in event_keywords.items():
                pattern = '|'.join(keywords)
                news_data[event] = news_data['headline'].str.contains(
                    pattern, case=False, regex=True, na=False
                ).astype(int)

            return news_data
        else:
            logging.info(f"No news found for {symbol} in the specified date range.")
            return pd.DataFrame()
    
    def _fetch_finnhub_news(self, symbol: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        """Fetch news from Finnhub API."""
        try:
            logging.info(f"Fetching Finnhub news for {symbol}")
            news = self.finnhub_client.company_news(
                symbol,
                _from=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d')
            )
            
            if not news:
                return pd.DataFrame()
            
            df = pd.DataFrame(news)
            df['date'] = pd.to_datetime(df['datetime'], unit='s').dt.date
            df = df[['date', 'headline', 'summary', 'url', 'source']]
            
            logging.info(f"Fetched {len(df)} news items from Finnhub for {symbol}")
            return df
            
        except Exception as e:
            logging.warning(f"Finnhub news fetch failed for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _fetch_newsapi_news(self, symbol: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        """Fetch news from NewsAPI."""
        try:
            logging.info(f"Fetching NewsAPI for {symbol}")
            url = (
                f"https://newsapi.org/v2/everything?q={symbol}"
                f"&from={start_date.strftime('%Y-%m-%d')}"
                f"&to={end_date.strftime('%Y-%m-%d')}"
                f"&sortBy=popularity&apiKey={self.api_config.news_api_key}"
            )
            
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                logging.warning(f"NewsAPI error for {symbol}: {response.status_code}")
                return pd.DataFrame()
            
            news = response.json().get('articles', [])
            if not news:
                return pd.DataFrame()
            
            df = pd.DataFrame(news)
            df['date'] = pd.to_datetime(df['publishedAt']).dt.date
            df = df[['date', 'title', 'description', 'url', 'source']]
            df.columns = ['date', 'headline', 'summary', 'url', 'source']
            
            logging.info(f"Fetched {len(df)} news items from NewsAPI for {symbol}")
            return df
            
        except Exception as e:
            logging.warning(f"NewsAPI fetch failed for {symbol}: {str(e)}")
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
                macro_data.index = macro_data.index.date
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
            client = tweepy.Client(bearer_token=self.api_config.twitter_bearer_token, wait_on_rate_limit=True)
            query = f"${symbol} OR #{symbol} lang:en -is:retweet"
            
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
                    logging.warning("Twitter rate limit reached. Waiting...")
                    time.sleep(60)
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