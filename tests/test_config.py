# tests/test_config.py
"""Tests for config.py — ModelConfig, ApiConfig, SystemConfig, AdvancedModelConfig."""

import os
from unittest import mock

import pytest

from config import ModelConfig, ApiConfig, SystemConfig, AdvancedModelConfig


class TestModelConfig:
    """Tests for ModelConfig defaults."""

    def test_defaults(self):
        """All defaults should be reasonable training parameters."""
        cfg = ModelConfig()
        assert cfg.lookback_days == 60
        assert cfg.forecast_steps == 252
        assert cfg.epochs == 100
        assert cfg.batch_size == 10
        assert 0 < cfg.train_split_ratio < 1
        assert cfg.min_data_rows_for_training > cfg.lookback_days

    def test_custom_values(self):
        """Should accept custom overrides."""
        cfg = ModelConfig(lookback_days=30, epochs=50)
        assert cfg.lookback_days == 30
        assert cfg.epochs == 50


class TestApiConfig:
    """Tests for ApiConfig loading and validation."""

    def test_from_env_reads_variables(self):
        """Should pick up environment variables when set."""
        env = {
            "NEWS_API_KEY": "test_news",
            "FINNHUB_API_KEY": "test_finnhub",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = ApiConfig.from_env()
        assert cfg.news_api_key == "test_news"
        assert cfg.finnhub_api_key == "test_finnhub"

    def test_from_env_missing_returns_none(self):
        """Missing env vars should result in None values."""
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = ApiConfig.from_env()
        assert cfg.news_api_key is None
        assert cfg.fred_api_key is None

    def test_validate_returns_false_when_no_keys(self):
        """Validation should return False when no keys are set."""
        cfg = ApiConfig()
        assert cfg.validate() is False

    def test_validate_returns_true_with_one_key(self):
        """Validation should pass with at least one key."""
        cfg = ApiConfig(news_api_key="test")
        assert cfg.validate() is True


class TestSystemConfig:
    """Tests for SystemConfig."""

    def test_system_config_defaults(self):
        """SystemConfig should have sensible defaults for batch processing."""
        cfg = SystemConfig(model=ModelConfig(), api=ApiConfig())
        assert cfg.max_processes >= 1
        assert cfg.batch_processing_size > 0
        assert cfg.batch_processing_delay_min < cfg.batch_processing_delay_max


class TestAdvancedModelConfig:
    """Tests for AdvancedModelConfig."""

    def test_defaults(self):
        """Advanced config should default to reasonable hyperparameters."""
        cfg = AdvancedModelConfig()
        assert cfg.lstm_units_layer1 > 0
        assert cfg.gru_units_layer1 > 0
        assert cfg.cnn_filters > 0
        assert cfg.epochs > 0
