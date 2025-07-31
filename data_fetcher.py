# data_fetcher.py
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

from config import ApiConfig
from utils import retry_with_backoff, cache_result

class DataFetcher:
    """Handles all data fetching operations with caching and retry logic."""
    
    def __init__(self, api_config: ApiConfig, cache_enabled: bool = True):
        self.api_config = api_config
        self.cache_enabled = cache_enabled
        
        # Initialize API clients
        self.finnhub_client = finnhub.Client(api_key=api_config.finnhub_api_key) if api_config.finnhub_api_key else None
        self.fred = Fred(api_key=api_config.fred_api_key) if api_config.fred_api_key else None
        
        # Configure Twitter client
        self.twitter_client = None
        if api_config.twitter_bearer_token:
            try:
                self.twitter_client = tweepy.Client(bearer_token=api_config.twitter_bearer_token, wait_on_rate_limit=True)
            except Exception as e:
                logging.warning(f"Failed to initialize Twitter client: {str(e)}")
    
    @retry_with_backoff(max_retries=3)
    def fetch_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch historical stock data with fallback to FMP API."""
        # Try yfinance first
        df = self._fetch_from_yfinance(symbol, start_date, end_date)
        if df is not None:
            return df
        
        # Fallback to FMP API if available
        if self.api_config.fmp_api_key:
            return self._fetch_from_fmp(symbol, start_date, end_date)
        
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
                timeout=15
            )
            
            if df.empty:
                logging.warning(f"yfinance returned empty data for {symbol}")
                return None
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip() for col in df.columns.values]
            
            # Ensure we have a Close column
            if 'Close' not in df.columns:
                if 'Adj Close' in df.columns:
                    df['Close'] = df['Adj Close']
                else:
                    logging.warning(f"No Close price found for {symbol}")
                    return None
            
            # Clean column names
            df.columns = [col.replace(' ', '') for col in df.columns]
            
            logging.info(f"Successfully fetched {len(df)} rows from yfinance for {symbol}")
            return df
            
        except Exception as e:
            logging.warning(f"yfinance fetch failed for {symbol}: {str(e)}")
            return None
    
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
            
        except Exception as e:
            logging.warning(f"FMP fetch failed for {symbol}: {str(e)}")
            return None
    
    @cache_result(cache_dir="cache/news", expiry_days=1)
    def fetch_news_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch news data from Finnhub and NewsAPI."""
        news_data = pd.DataFrame()
        today = datetime.now().date()
        
        # Calculate API-specific date ranges
        finnhub_min_date = today - timedelta(days=365)  # Finnhub 1 year limit
        newsapi_min_date = today - timedelta(days=30)   # NewsAPI 1 month limit
        
        # Determine effective start dates
        finnhub_start = max(start_date.date(), finnhub_min_date)
        newsapi_start = max(start_date.date(), newsapi_min_date)
        
        # Try Finnhub first
        if self.api_config.finnhub_api_key and self.finnhub_client:
            news_data = self._fetch_finnhub_news(symbol, finnhub_start, end_date.date())
        
        # Fallback to NewsAPI if no results
        if news_data.empty and self.api_config.news_api_key:
            news_data = self._fetch_newsapi_news(symbol, newsapi_start, end_date.date())
        
        return news_data
    
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
    def fetch_macro_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch macroeconomic data from FRED."""
        if not self.api_config.fred_api_key or not self.fred:
            logging.info("FRED API not configured")
            return pd.DataFrame()
        
        try:
            indicators = {
                'GDP': 'GDPC1',        # Real GDP
                'UNRATE': 'UNRATE',     # Unemployment Rate
                'CPIAUCSL': 'CPIAUCSL', # CPI
                'FEDFUNDS': 'FEDFUNDS', # Fed Funds Rate
                'VIX': 'VIXCLS'         # VIX
            }
            
            dfs = []
            for name, code in indicators.items():
                try:
                    series = self.fred.get_series(
                        code,
                        observation_start=start_date.strftime('%Y-%m-%d'),
                        observation_end=end_date.strftime('%Y-%m-%d')
                    )
                    if not series.empty:
                        series.name = name
                        dfs.append(series)
                except Exception as e:
                    logging.warning(f"FRED error for {name}: {str(e)}")
            
            if not dfs:
                return pd.DataFrame()
            
            macro_data = pd.concat(dfs, axis=1)
            macro_data = macro_data.resample('D').ffill().bfill()
            macro_data.index = macro_data.index.date
            
            logging.info(f"Fetched macro data with {len(macro_data)} rows")
            return macro_data
            
        except Exception as e:
            logging.error(f"Macro data fetch failed: {str(e)}")
            return pd.DataFrame()
    
    @cache_result(cache_dir="cache/tweets", expiry_days=1)
    def fetch_tweet_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch tweets for a stock symbol."""
        if not self.api_config.twitter_bearer_token or not self.twitter_client:
            logging.info("Twitter API not configured")
            return pd.DataFrame()
        
        try:
            # Twitter API v2 recent search has a 7-day limit
            effective_start = max(start_date, datetime.now() - timedelta(days=6))
            if effective_start > end_date:
                return pd.DataFrame()
            
            query = f"${symbol} OR #{symbol} lang:en -is:retweet"
            tweets_list = []
            next_token = None
            max_pages = 5  # Limit to 500 tweets
            
            logging.info(f"Fetching tweets for {symbol}")
            
            for _ in range(max_pages):
                try:
                    response = self.twitter_client.search_recent_tweets(
                        query=query,
                        max_results=100,
                        start_time=effective_start.isoformat(),
                        end_time=end_date.isoformat(),
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
                    logging.warning("Twitter rate limit reached")
                    time.sleep(60)
                except Exception as e:
                    logging.warning(f"Twitter API error: {str(e)}")
                    break
            
            if tweets_list:
                df = pd.DataFrame(tweets_list)
                logging.info(f"Fetched {len(df)} tweets for {symbol}")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logging.error(f"Tweet fetch failed for {symbol}: {str(e)}")
            return pd.DataFrame()