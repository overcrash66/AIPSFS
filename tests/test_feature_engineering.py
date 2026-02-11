# tests/test_feature_engineering.py
"""Tests for feature_engineering.py — technical indicators, data preparation."""

import numpy as np
import pandas as pd
import pytest

from feature_engineering import FeatureEngineer


@pytest.fixture
def engineer():
    """Create a FeatureEngineer with no external API dependencies."""
    return FeatureEngineer(
        news_api_history_days=30,
        min_data_rows_for_training=50,
    )


class TestCalculateTechnicalIndicators:
    """Tests for calculate_technical_indicators."""

    def test_adds_sma_columns(self, engineer, sample_stock_df):
        """Should add SMA_20, SMA_50, SMA_200 columns."""
        result = engineer.calculate_technical_indicators(sample_stock_df.copy())
        assert "SMA_20" in result.columns
        assert "SMA_50" in result.columns
        assert "SMA_200" in result.columns

    def test_adds_rsi(self, engineer, sample_stock_df):
        """RSI should be between 0 and 100."""
        result = engineer.calculate_technical_indicators(sample_stock_df.copy())
        assert "RSI" in result.columns
        assert result["RSI"].between(0, 100).all()

    def test_adds_macd_and_signal(self, engineer, sample_stock_df):
        """Should add MACD and Signal columns."""
        result = engineer.calculate_technical_indicators(sample_stock_df.copy())
        assert "MACD" in result.columns
        assert "Signal" in result.columns

    def test_empty_df_returns_unchanged(self, engineer):
        """Empty DataFrame should pass through unchanged."""
        df = pd.DataFrame()
        result = engineer.calculate_technical_indicators(df)
        assert result.empty

    def test_missing_close_returns_unchanged(self, engineer):
        """DataFrame without Close column should pass through."""
        df = pd.DataFrame({"Open": [1, 2, 3]})
        result = engineer.calculate_technical_indicators(df)
        assert "SMA_20" not in result.columns


class TestPrepareMultiFeatureData:
    """Tests for prepare_multi_feature_data."""

    def test_basic_preparation(self, engineer, sample_stock_df):
        """Should produce properly shaped X and y arrays."""
        lookback = 30
        X, y, scalers, feature_cols = engineer.prepare_multi_feature_data(
            sample_stock_df, lookback
        )
        assert X is not None
        assert y is not None
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == lookback  # timesteps
        assert "Close" in feature_cols
        assert "Close" in scalers

    def test_scaling_range(self, engineer, sample_stock_df):
        """Scaled values should be in [0, 1] range."""
        X, y, _, _ = engineer.prepare_multi_feature_data(sample_stock_df, 30)
        assert X.min() >= 0.0
        assert X.max() <= 1.0
        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_insufficient_data_returns_none(self, engineer):
        """Too little data should return None."""
        small_df = pd.DataFrame(
            {"Close": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        X, y, scalers, _ = engineer.prepare_multi_feature_data(small_df, lookback=60)
        assert X is None
        assert y is None

    def test_empty_df_returns_none(self, engineer):
        """Empty DataFrame should return None."""
        X, y, scalers, _ = engineer.prepare_multi_feature_data(
            pd.DataFrame(), lookback=30
        )
        assert X is None
        assert y is None

    def test_optional_features_included(self, engineer, sample_stock_df):
        """When tech-indicator columns exist, they should appear in feature_cols."""
        df = engineer.calculate_technical_indicators(sample_stock_df.copy())
        _, _, _, feature_cols = engineer.prepare_multi_feature_data(df, 30)
        assert "SMA_20" in feature_cols
        assert "RSI" in feature_cols
        assert "MACD" in feature_cols
