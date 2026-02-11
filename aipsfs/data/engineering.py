# feature_engineering.py
import logging
import pandas as pd
import numpy as np
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

from aipsfs.utils.helpers import safe_merge

# Download NLTK data if needed
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class FeatureEngineer:
    """Handles feature engineering for stock data."""
    
    def __init__(self, news_api_history_days=400, min_data_rows_for_training=160,
                 api_config=None, data_fetcher=None):
        self.vader = SentimentIntensityAnalyzer()
        self.news_api_history_days = news_api_history_days
        self.min_data_rows_for_training = min_data_rows_for_training
        self.api_config = api_config
        self.data_fetcher = data_fetcher
    
    def process_stock_data(self, symbol: str, df: pd.DataFrame, 
                          start_date_full: datetime, end_date_full: datetime) -> pd.DataFrame:
        """Process raw stock data with all features."""
        if df.empty or 'Close' not in df.columns:
            logging.warning(f"Invalid DataFrame for {symbol}")
            return None
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Standardize date column
        df = self._standardize_dates(df)
        if df is None:
            return None
        
        # Fetch and integrate additional data
        df = self._integrate_news_data(df, symbol, start_date_full, end_date_full)
        df = self._integrate_macro_data(df, start_date_full, end_date_full)
        df = self._integrate_tweet_data(df, symbol, start_date_full, end_date_full)
        
        # Final cleaning and validation
        df = self._clean_and_validate(df)
        if df is None:
            return None
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for stock data."""
        if df.empty or 'Close' not in df.columns:
            return df
        
        try:
            # Ensure Close column is numeric
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            
            # Simple Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
            df['SMA_200'] = df['Close'].rolling(window=200, min_periods=1).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.ewm(span=14, adjust=False, min_periods=1).mean()
            avg_loss = loss.ewm(span=14, adjust=False, min_periods=1).mean()
            
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df['RSI'] = 100 - (100 / (1 + rs)).fillna(50)
            
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
            df['MACD'] = exp1 - exp2
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating technical indicators: {str(e)}")
            return df
    
    def _standardize_dates(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Standardize date column and set as index."""
        # Debug: Print current DataFrame info
        logging.info(f"DataFrame before date standardization:")
        logging.info(f"  Columns: {df.columns.tolist()}")
        logging.info(f"  Index type: {type(df.index)}")
        logging.info(f"  Index name: {df.index.name}")
        logging.info(f"  Shape: {df.shape}")
        
        # If the index is a DatetimeIndex, we need to preserve it
        if isinstance(df.index, pd.DatetimeIndex):
            logging.info("DataFrame already has DatetimeIndex")
            # Make a copy of the index as a column
            df['date'] = df.index
            # Now we can work with it
        else:
            # Look for date column
            date_col = None
            if 'Date' in df.columns:
                date_col = 'Date'
                logging.info("Found 'Date' column")
            elif 'date' in df.columns:
                date_col = 'date'
                logging.info("Found 'date' column")
            else:
                # Try to find any column that might be a date
                for col in df.columns:
                    if 'date' in col.lower():
                        date_col = col
                        logging.info(f"Found potential date column: {col}")
                        break
            
            if date_col is None:
                logging.error(f"Could not identify date column. Columns: {df.columns.tolist()}")
                return None
            
            # Rename to 'date' if needed
            if date_col != 'date':
                df = df.rename(columns={date_col: 'date'})
                logging.info(f"Renamed '{date_col}' to 'date'")
        
        try:
            # Check if already datetime
            if pd.api.types.is_datetime64_any_dtype(df['date']):
                logging.info("Date column is already datetime")
                df['date'] = df['date'].dt.date
            else:
                logging.info("Converting date column to datetime")
                df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
            
            # Check for NaT values after conversion
            if df['date'].isna().any():
                logging.warning(f"Found {df['date'].isna().sum()} NaT values after conversion")
                df = df.dropna(subset=['date'])
            
        except Exception as e:
            logging.error(f"Date conversion failed: {str(e)}")
            return None
        
        # Sort by date and set as index
        df = df.sort_values('date')
        df.set_index('date', inplace=True)
        
        logging.info(f"Date standardization completed. Final shape: {df.shape}")
        return df
    
    def _integrate_news_data(self, df: pd.DataFrame, symbol: str, 
                       start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch and integrate news data."""
        if not self.data_fetcher:
            return df
        
        news_end_date = end_date.date()
        news_start_date = max(start_date.date(), news_end_date - timedelta(days=self.news_api_history_days))
        
        news_raw = self.data_fetcher.fetch_news_data(symbol, news_start_date, news_end_date)
        if news_raw.empty:
            return df
        
        news_agg = self._aggregate_news_data(news_raw)
        if news_agg.empty:
            return df
        
        # Ensure both DataFrames have a 'date' column
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: 'date'})
        
        if 'date' not in news_agg.columns:
            logging.error("News aggregated data missing 'date' column")
            return df
        
        return safe_merge(df, news_agg, on='date')

    def _integrate_tweet_data(self, df: pd.DataFrame, symbol: str, 
                            start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch and integrate tweet data."""
        if not self.data_fetcher:
            return df
        
        news_end_date = end_date.date()
        news_start_date = max(start_date.date(), news_end_date - timedelta(days=self.news_api_history_days))
        
        tweet_raw = self.data_fetcher.fetch_tweet_data(symbol, news_start_date, news_end_date)
        if tweet_raw.empty:
            return df
        
        tweet_agg = self._aggregate_tweet_data(tweet_raw)
        if tweet_agg.empty:
            return df
        
        # Ensure both DataFrames have a 'date' column
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: 'date'})
        
        if 'date' not in tweet_agg.columns:
            logging.error("Tweet aggregated data missing 'date' column")
            return df
        
        return safe_merge(df, tweet_agg, on='date')

    def _integrate_macro_data(self, df: pd.DataFrame, 
                            start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch and integrate macroeconomic data."""
        if not self.data_fetcher:
            return df
        
        macro_data = self.data_fetcher.fetch_macro_data(start_date, end_date)
        if macro_data.empty:
            return df
        
        # Ensure both DataFrames have a 'date' column
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: 'date'})
        
        if isinstance(macro_data.index, pd.DatetimeIndex):
            macro_data = macro_data.reset_index()
            macro_data = macro_data.rename(columns={macro_data.columns[0]: 'date'})
        
        if 'date' not in macro_data.columns:
            logging.error("Macro data missing 'date' column")
            return df
        
        return safe_merge(df, macro_data, on='date')
    
    def _aggregate_news_data(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate news data by date."""
        if news_df.empty or 'date' not in news_df.columns:
            return pd.DataFrame()
        
        try:
            # Clean text
            news_df['clean_text'] = news_df['headline'] + " " + news_df['summary'].fillna('')
            news_df['clean_text'] = news_df['clean_text'].apply(
                lambda x: re.sub(r'[^\w\s]', '', str(x).lower())
            )
            
            # Calculate sentiment
            news_df['sentiment'] = news_df['clean_text'].apply(
                lambda x: self.vader.polarity_scores(x)['compound']
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
                news_df[event] = news_df['headline'].str.contains(
                    pattern, case=False, regex=True, na=False
                ).astype(int)
            
            # Aggregate by date
            daily_news = news_df.groupby('date').agg(
                news_sentiment=('sentiment', 'mean'),
                news_count=('sentiment', 'count'),
                earnings_event=('earnings', 'sum'),
                merger_event=('merger', 'sum'),
                regulation_event=('regulation', 'sum'),
                product_event=('product', 'sum'),
                macro_event=('macro', 'sum')
            ).reset_index()
            
            # Add lagged features
            daily_news['news_sentiment_lag1'] = daily_news['news_sentiment'].shift(1).fillna(0)
            daily_news['news_sentiment_lag3'] = daily_news['news_sentiment'].rolling(3).mean().shift(1).fillna(0)
            daily_news['news_count_lag1'] = daily_news['news_count'].shift(1).fillna(0)
            daily_news['news_count_lag3'] = daily_news['news_count'].rolling(3).mean().shift(1).fillna(0)
            
            return daily_news
            
        except Exception as e:
            logging.error(f"News aggregation error: {str(e)}")
            return pd.DataFrame()
    
    def _aggregate_tweet_data(self, tweet_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate tweet data by date."""
        if tweet_df.empty or 'date' not in tweet_df.columns:
            return pd.DataFrame()
        
        try:
            # Calculate sentiment
            tweet_df['sentiment'] = tweet_df['text'].apply(
                lambda x: self.vader.polarity_scores(str(x))['compound']
            )
            
            # Aggregate by date
            daily_tweets = tweet_df.groupby('date').agg(
                tweet_sentiment=('sentiment', 'mean'),
                tweet_count=('sentiment', 'count'),
                tweet_likes=('likes', 'sum'),
                retweet_count=('retweets', 'sum')
            ).reset_index()
            
            # Add lagged features
            daily_tweets['tweet_sentiment_lag1'] = daily_tweets['tweet_sentiment'].shift(1).fillna(0)
            daily_tweets['tweet_sentiment_lag3'] = daily_tweets['tweet_sentiment'].rolling(3).mean().shift(1).fillna(0)
            daily_tweets['tweet_count_lag1'] = daily_tweets['tweet_count'].shift(1).fillna(0)
            daily_tweets['tweet_count_lag3'] = daily_tweets['tweet_count'].rolling(3).mean().shift(1).fillna(0)
            
            return daily_tweets
            
        except Exception as e:
            logging.error(f"Tweet aggregation error: {str(e)}")
            return pd.DataFrame()
    
    def _clean_and_validate(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Final cleaning and validation of processed data."""
        if df.empty:
            return None
        
        # Convert columns to numeric where possible
        for col in df.columns:
            if df[col].dtype == 'object' and col not in ['url', 'source', 'headline', 'summary', 'clean_text', 'text']:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    logging.warning(f"Could not convert column {col} to numeric: {e}")
        
        # Fill missing values
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        # Ensure we have Close prices
        if 'Close' not in df.columns:
            logging.error("'Close' column missing after processing")
            return None
        
        # Drop rows with missing Close prices
        initial_rows = len(df)
        df.dropna(subset=['Close'], inplace=True)
        if len(df) < initial_rows:
            logging.warning(f"Dropped {initial_rows - len(df)} rows with missing Close prices")
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Ensure enough data for model training
        if len(df) < self.min_data_rows_for_training:
            logging.warning(f"Insufficient data after cleaning ({len(df)} rows, need at least {self.min_data_rows_for_training})")
            return None
        
        return df
    
    def prepare_multi_feature_data(self, data: pd.DataFrame, lookback: int) -> Tuple:
        """Prepare data for LSTM modeling."""
        if not isinstance(data, pd.DataFrame) or data.empty:
            return None, None, {}, []
        
        if 'Close' not in data.columns:
            logging.error(f"Missing 'Close' column. Available columns: {data.columns.tolist()}")
            return None, None, {}, []
        
        # Reset index to make date a column
        data_copy = data.reset_index().copy()
        
        # Define feature columns - start with just Close
        feature_cols = ['Close']
        
        # Add other price columns if available
        for col in ['Open', 'High', 'Low', 'Volume']:
            if col in data_copy.columns:
                feature_cols.append(col)
            else:
                logging.warning(f"Column '{col}' not found in data")
        
        # Add available optional features
        optional_features = [
            'SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal',
            'news_sentiment', 'news_count', 'earnings_event',
            'macro_event', 'merger_event', 'regulation_event', 'product_event',
            'news_sentiment_lag1', 'news_sentiment_lag3', 'news_count_lag1', 'news_count_lag3',
            'tweet_sentiment', 'tweet_count', 'tweet_likes', 'retweet_count',
            'tweet_sentiment_lag1', 'tweet_sentiment_lag3', 'tweet_count_lag1', 'tweet_count_lag3',
            'GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'VIX'
        ]
        
        # Add available optional features
        for col in optional_features:
            if col in data_copy.columns and col not in feature_cols:
                feature_cols.append(col)
        
        logging.info(f"Using features: {feature_cols}")
        
        # Select and clean data
        data_for_scaling = data_copy[feature_cols].copy()
        data_for_scaling.ffill(inplace=True)
        data_for_scaling.bfill(inplace=True)
        data_for_scaling.dropna(inplace=True)
        
        if len(data_for_scaling) < lookback + 1:
            logging.warning(f"Insufficient data after cleaning: {len(data_for_scaling)} rows, need at least {lookback + 1}")
            return None, None, {}, feature_cols
        
        # Scale features
        scalers = {}
        scaled_data = np.zeros(data_for_scaling.shape)
        
        for i, col in enumerate(feature_cols):
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_values = scaler.fit_transform(data_for_scaling[col].values.reshape(-1, 1))
            scaled_data[:, i] = scaled_values.flatten()
            scalers[col] = scaler
        
        # Create sequences
        X, y = [], []
        close_idx = feature_cols.index('Close')
        
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, :])
            y.append(scaled_data[i, close_idx])
        
        return np.array(X), np.array(y), scalers, feature_cols