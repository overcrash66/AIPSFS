# tests/test_validation.py
"""Tests for validation.py — validate_stock_data, validate_model_input."""

import numpy as np
import pandas as pd
import pytest

from validation import validate_stock_data, validate_model_input


class TestValidateStockData:
    """Tests for validate_stock_data."""

    def test_valid_dataframe_passes(self, sample_stock_df):
        """A well-formed OHLCV DataFrame should pass validation."""
        assert validate_stock_data(sample_stock_df) is True

    def test_empty_dataframe_fails(self):
        """An empty DataFrame should fail validation."""
        assert validate_stock_data(pd.DataFrame()) is False

    def test_missing_column_fails(self, sample_stock_df):
        """Dropping a required column should fail validation."""
        df = sample_stock_df.drop(columns=["Volume"])
        assert validate_stock_data(df) is False

    def test_non_numeric_column_fails(self, sample_stock_df):
        """A non-numeric price column should fail validation."""
        df = sample_stock_df.copy()
        df["Close"] = "not_a_number"
        assert validate_stock_data(df) is False

    def test_nan_close_fails(self, sample_stock_df):
        """NaN values in Close should fail validation."""
        df = sample_stock_df.copy()
        df.loc[df.index[0], "Close"] = np.nan
        assert validate_stock_data(df) is False


class TestValidateModelInput:
    """Tests for validate_model_input."""

    def test_valid_input_passes(self):
        """Matching length arrays should pass."""
        X = np.random.rand(100, 60, 5)
        y = np.random.rand(100)
        assert validate_model_input(X, y) is True

    def test_none_input_fails(self):
        """None inputs should fail."""
        assert validate_model_input(None, np.array([1])) is False
        assert validate_model_input(np.array([1]), None) is False

    def test_empty_input_fails(self):
        """Empty arrays should fail."""
        assert validate_model_input(np.array([]), np.array([1])) is False

    def test_length_mismatch_fails(self):
        """Mismatched lengths should fail."""
        X = np.random.rand(10, 5)
        y = np.random.rand(8)
        assert validate_model_input(X, y) is False
