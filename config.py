# config.py
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class ModelConfig:
    lookback_days: int = 60
    forecast_steps: int = 252
    epochs: int = 100
    batch_size: int = 32
    lstm_units_layer1: int = 128
    lstm_units_layer2: int = 64
    train_split_ratio: float = 0.8
    early_stopping_patience: int = 7
    min_data_rows_for_training: int = 160  # lookback + 100

@dataclass
class ApiConfig:
    news_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    twitter_bearer_token: Optional[str] = None
    fred_api_key: Optional[str] = None
    fmp_api_key: Optional[str] = None
    
    @classmethod
    def from_env(cls):
        return cls(
            news_api_key=os.getenv('NEWS_API_KEY'),
            finnhub_api_key=os.getenv('FINNHUB_API_KEY'),
            twitter_bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
            fred_api_key=os.getenv('FRED_API_KEY'),
            fmp_api_key=os.getenv('FMP_API_KEY')
        )

@dataclass
class SystemConfig:
    model: ModelConfig
    api: ApiConfig
    max_processes: int = 2
    batch_processing_size: int = 10
    batch_processing_delay_min: int = 30
    batch_processing_delay_max: int = 90
    min_predicted_return_pct: int = 50
    news_api_history_days: int = 400