# tests/test_utils.py
"""Tests for utils.py — load_stock_list, retry_with_backoff, cache_result, safe_merge."""

import os
import time

import pandas as pd
import pytest

from utils import load_stock_list, retry_with_backoff, cache_result, safe_merge


# ── load_stock_list ──────────────────────────────────────────────────────────


class TestLoadStockList:
    """Tests for the load_stock_list function."""

    def test_loads_valid_csv(self, sample_stock_csv):
        """Should parse a well-formed CSV into a list of dicts."""
        result = load_stock_list(sample_stock_csv)
        assert len(result) == 2
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["name"] == "Apple Inc."
        assert result[1]["symbol"] == "MSFT"

    def test_empty_csv_returns_empty(self, empty_csv):
        """A CSV with only headers should return an empty list."""
        result = load_stock_list(empty_csv)
        assert result == []

    def test_missing_file_returns_empty(self):
        """A path that doesn't exist should return an empty list (not raise)."""
        result = load_stock_list("nonexistent_file.csv")
        assert result == []

    def test_alternative_column_names(self, tmp_path):
        """Should recognise 'ticker' and 'company' as valid column names."""
        csv_path = tmp_path / "alt.csv"
        csv_path.write_text("ticker,company\nTSLA,Tesla Inc.\n")
        result = load_stock_list(str(csv_path))
        assert len(result) == 1
        assert result[0]["symbol"] == "TSLA"
        assert result[0]["name"] == "Tesla Inc."

    def test_skips_empty_symbols(self, tmp_path):
        """Rows with empty symbols should be skipped."""
        csv_path = tmp_path / "blank.csv"
        csv_path.write_text("symbol,name\nAAPL,Apple\n,\nMSFT,Microsoft\n")
        result = load_stock_list(str(csv_path))
        assert len(result) == 2

    def test_missing_name_column_uses_symbol(self, tmp_path):
        """If no name column exists, symbol should be used as name."""
        csv_path = tmp_path / "no_name.csv"
        csv_path.write_text("symbol\nAAPL\nGOOGL\n")
        result = load_stock_list(str(csv_path))
        assert len(result) == 2
        assert result[0]["name"] == "AAPL"


# ── retry_with_backoff ───────────────────────────────────────────────────────


class TestRetryWithBackoff:
    """Tests for the retry_with_backoff decorator."""

    def test_returns_on_first_success(self):
        """Function should only be called once if it succeeds immediately."""
        call_count = {"n": 0}

        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        def succeed():
            call_count["n"] += 1
            return "ok"

        assert succeed() == "ok"
        assert call_count["n"] == 1

    def test_retries_then_succeeds(self):
        """Should retry the correct number of times before succeeding."""
        call_count = {"n": 0}

        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        def fail_twice():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ValueError("not yet")
            return "finally"

        assert fail_twice() == "finally"
        assert call_count["n"] == 3

    def test_raises_after_all_retries(self):
        """Should raise the last exception after exhausting retries."""

        @retry_with_backoff(max_retries=2, initial_delay=0.01)
        def always_fail():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            always_fail()


# ── cache_result ─────────────────────────────────────────────────────────────


class TestCacheResult:
    """Tests for the cache_result decorator."""

    def test_caches_and_returns(self, tmp_path):
        """Result should be cached on disk and returned from cache on second call."""
        call_count = {"n": 0}
        cache_dir = str(tmp_path / "cache")

        @cache_result(cache_dir=cache_dir, expiry_days=1)
        def expensive(x):
            call_count["n"] += 1
            return x * 2

        assert expensive(5) == 10
        assert expensive(5) == 10  # should come from cache
        assert call_count["n"] == 1

    def test_different_args_not_cached(self, tmp_path):
        """Different arguments should produce separate cache entries."""
        call_count = {"n": 0}
        cache_dir = str(tmp_path / "cache")

        @cache_result(cache_dir=cache_dir, expiry_days=1)
        def add_one(x):
            call_count["n"] += 1
            return x + 1

        assert add_one(1) == 2
        assert add_one(2) == 3
        assert call_count["n"] == 2


# ── safe_merge ───────────────────────────────────────────────────────────────


class TestSafeMerge:
    """Tests for the safe_merge function."""

    def test_basic_merge(self):
        """Should merge two DataFrames on a common key."""
        left = pd.DataFrame({"date": [1, 2, 3], "price": [10, 20, 30]})
        right = pd.DataFrame({"date": [1, 2], "sentiment": [0.5, -0.3]})
        result = safe_merge(left, right, on="date")
        assert "sentiment" in result.columns
        assert len(result) == 3  # left-join preserves all rows

    def test_empty_right_returns_left(self):
        """If right DataFrame is empty, should return left unchanged."""
        left = pd.DataFrame({"date": [1, 2], "price": [10, 20]})
        right = pd.DataFrame()
        result = safe_merge(left, right, on="date")
        assert result.equals(left)

    def test_missing_merge_key_returns_left(self):
        """If the merge key is missing in right, should return left safely."""
        left = pd.DataFrame({"date": [1], "price": [10]})
        right = pd.DataFrame({"other": [1], "value": [5]})
        result = safe_merge(left, right, on="date")
        # Should not crash — returns left since key is missing
        assert "price" in result.columns
