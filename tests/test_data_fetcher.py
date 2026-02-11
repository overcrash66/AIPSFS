# tests/test_data_fetcher.py
"""Tests for DataFetcher — yfinance column normalisation and Twitter query format."""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from aipsfs.config import ApiConfig
from aipsfs.data.fetcher import DataFetcher


@pytest.fixture
def api_config_no_keys():
    """ApiConfig with no real API keys (safe for unit tests)."""
    return ApiConfig()


@pytest.fixture
def api_config_twitter():
    """ApiConfig with a fake Twitter bearer token."""
    return ApiConfig(twitter_bearer_token="fake-token")


class TestYfinanceColumnNormalization:
    """Tests for _fetch_from_yfinance column handling."""

    def test_multiindex_columns_simplified(self, api_config_no_keys):
        """MultiIndex columns like ('Close','AAPL') should become 'Close'."""
        fetcher = DataFetcher(api_config_no_keys, cache_enabled=False)

        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        arrays = [
            ["Open", "High", "Low", "Close", "Volume"],
            ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
        ]
        multi_idx = pd.MultiIndex.from_arrays(arrays)
        data = np.random.rand(5, 5)
        df = pd.DataFrame(data, index=dates, columns=multi_idx)

        with patch("yfinance.download", return_value=df):
            from datetime import datetime
            result = fetcher._fetch_from_yfinance("AAPL", datetime(2024, 1, 1), datetime(2024, 1, 8))

        assert result is not None
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns, f"Column '{col}' missing after normalisation"

    def test_lowercase_columns_renamed(self, api_config_no_keys):
        """Lowercase column names like 'close' should be renamed to 'Close'."""
        fetcher = DataFetcher(api_config_no_keys, cache_enabled=False)

        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame({
            "open": [1.0] * 5,
            "high": [2.0] * 5,
            "low": [0.5] * 5,
            "close": [1.5] * 5,
            "volume": [1000] * 5,
        }, index=dates)

        with patch("yfinance.download", return_value=df):
            from datetime import datetime
            result = fetcher._fetch_from_yfinance("AAPL", datetime(2024, 1, 1), datetime(2024, 1, 8))

        assert result is not None
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns

    def test_empty_df_returns_none(self, api_config_no_keys):
        """An empty DataFrame from yfinance should return None."""
        fetcher = DataFetcher(api_config_no_keys, cache_enabled=False)

        with patch("yfinance.download", return_value=pd.DataFrame()):
            from datetime import datetime
            result = fetcher._fetch_from_yfinance("AAPL", datetime(2024, 1, 1), datetime(2024, 1, 8))

        assert result is None


class TestTwitterQueryFormat:
    """Tests for Twitter query construction in fetch_tweet_data."""

    @patch("tweepy.Client")
    def test_query_has_no_cashtag(self, mock_client_cls, api_config_twitter):
        """The Twitter query should not use the $ cashtag operator."""
        # Mock the tweepy Client so __init__ and search don't make real calls
        mock_client_instance = MagicMock()
        mock_client_cls.return_value = mock_client_instance
        mock_response = MagicMock()
        mock_response.data = None
        mock_response.meta = {}
        mock_client_instance.search_recent_tweets.return_value = mock_response

        fetcher = DataFetcher(api_config_twitter, cache_enabled=False)

        from datetime import datetime
        fetcher.fetch_tweet_data.__wrapped__(
            fetcher, "AAPL", datetime(2024, 1, 1), datetime(2024, 1, 8)
        )

        # Inspect the query argument passed to search_recent_tweets
        call_args = mock_client_instance.search_recent_tweets.call_args
        if call_args is not None:
            query = call_args.kwargs.get("query", call_args[1].get("query", ""))
            assert "$" not in query, f"Query should not use $ cashtag, got: {query}"

    @patch("tweepy.Client")
    def test_wait_on_rate_limit_disabled(self, mock_client_cls, api_config_twitter):
        """The tweepy Client should be initialised with wait_on_rate_limit=False."""
        mock_client_cls.return_value = MagicMock()

        # Re-create the DataFetcher to trigger __init__
        DataFetcher(api_config_twitter, cache_enabled=False)

        # Check the call to tweepy.Client in __init__
        init_call = mock_client_cls.call_args
        assert init_call is not None
        assert init_call.kwargs.get("wait_on_rate_limit") is False, \
            "wait_on_rate_limit should be False to avoid blocking"
