# config.py
from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

@dataclass
class ModelConfig:
    lookback_days: int = 60
    forecast_steps: int = 252
    epochs: int = 100
    batch_size: int = 10
    lstm_units_layer1: int = 128
    lstm_units_layer2: int = 64
    train_split_ratio: float = 0.8
    early_stopping_patience: int = 15
    min_data_rows_for_training: int = 110  # lookback + 100

@dataclass
class ApiConfig:
    news_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    twitter_bearer_token: Optional[str] = None
    fred_api_key: Optional[str] = None
    fmp_api_key: Optional[str] = None
    alpha_vantage_api_key: Optional[str] = None
    gnews_api_key: Optional[str] = None
    
    @classmethod
    def from_env(cls):
        """Load API configuration from environment variables."""
        return cls(
            news_api_key=os.getenv('NEWS_API_KEY'),
            finnhub_api_key=os.getenv('FINNHUB_API_KEY'),
            twitter_bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
            fred_api_key=os.getenv('FRED_API_KEY'),
            fmp_api_key=os.getenv('FMP_API_KEY'),
            alpha_vantage_api_key=os.getenv('ALPHA_VANTAGE_API_KEY'),
            gnews_api_key=os.getenv('GNEWS_API_KEY')  
        )
    
    def validate(self):
        """Validate that at least one API key is configured."""
        if not any([
            self.news_api_key,
            self.finnhub_api_key,
            self.twitter_bearer_token,
            self.fred_api_key,
            self.fmp_api_key,
            self.alpha_vantage_api_key,
            self.gnews_api_key
        ]):
            logging.warning("No API keys configured. Some features may not work.")
            return False
        return True

@dataclass
class SystemConfig:
    model: ModelConfig
    api: ApiConfig
    max_processes: int = 1
    batch_processing_size: int = 10
    batch_processing_delay_min: int = 10
    batch_processing_delay_max: int = 100
    min_predicted_return_pct: int = 50
    news_api_history_days: int = 400

@dataclass
class AdvancedModelConfig:
    """Configuration for advanced models."""
    lstm_units_layer1: int = 128
    lstm_units_layer2: int = 64
    gru_units_layer1: int = 128
    gru_units_layer2: int = 64
    cnn_filters: int = 64
    epochs: int = 50
    batch_size: int = 32
    early_stopping_patience: int = 15
    forecast_steps: int = 252
    train_split_ratio: float = 0.8
    min_data_rows_for_training: int = 100   