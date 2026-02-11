# tests/conftest.py
"""Shared fixtures for the AIPSFS test suite."""

import os
import sys
import tempfile

import pytest
import pandas as pd
import numpy as np

# Ensure project root is on the path so the aipsfs package can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def sample_stock_csv(tmp_path):
    """Create a temporary CSV file with sample stock data."""
    csv_path = tmp_path / "stocks.csv"
    csv_path.write_text("symbol,name\nAAPL,Apple Inc.\nMSFT,Microsoft Corporation\n")
    return str(csv_path)


@pytest.fixture
def empty_csv(tmp_path):
    """Create a temporary empty CSV file with headers only."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("symbol,name\n")
    return str(csv_path)


@pytest.fixture
def sample_stock_df():
    """Create a sample stock DataFrame with OHLCV data."""
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(200) * 0.5)
    df = pd.DataFrame(
        {
            "Open": close - np.random.rand(200),
            "High": close + np.random.rand(200),
            "Low": close - np.random.rand(200) - 0.5,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, 200),
        },
        index=dates,
    )
    return df
