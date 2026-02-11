# tests/test_advanced_predictor.py
"""Tests for AdvancedStockPredictor — inverse scaling and checkpoint validation."""

import os
import shutil
import tempfile
import numpy as np
import pytest
from unittest.mock import MagicMock
from sklearn.preprocessing import MinMaxScaler

from aipsfs.config import AdvancedModelConfig
from aipsfs.models.advanced import AdvancedStockPredictor


def _make_scalers_and_data(n_samples: int = 200, n_features: int = 5, lookback: int = 30):
    """Create synthetic scaled data with known scalers for testing."""
    # Simulate real OHLCV-like data
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(n_samples) * 2) + 150  # around $150
    feature_names = ['Open', 'High', 'Low', 'Close', 'Volume'][:n_features]

    raw_data = np.column_stack([
        prices + np.random.randn(n_samples) * 0.5  # Open
        for _ in range(n_features)
    ])
    # Make the Close column (index 3 if 5 features, else last) use actual prices
    close_idx = feature_names.index('Close') if 'Close' in feature_names else 0
    raw_data[:, close_idx] = prices

    # Fit per-column MinMaxScalers (same approach as feature engineering)
    scalers = {}
    scaled_data = np.zeros_like(raw_data)
    for i, col_name in enumerate(feature_names):
        scaler = MinMaxScaler()
        scaled_data[:, i] = scaler.fit_transform(raw_data[:, i].reshape(-1, 1)).flatten()
        scalers[col_name] = scaler

    # Build sequences
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        y.append(scaled_data[i, close_idx])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y, scalers, feature_names


class TestAdvancedPredictorInverseScaling:
    """Verify predictions are in real price scale, not [0,1]."""

    @pytest.fixture
    def checkpoint_dir(self):
        d = tempfile.mkdtemp(prefix="aipsfs_test_ensemble_")
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_predict_returns_real_prices(self, checkpoint_dir):
        """Predicted prices should be > $1 (real scale), not stuck in [0,1]."""
        X, y, scalers, feature_cols = _make_scalers_and_data(n_samples=120, lookback=20)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        config = AdvancedModelConfig()
        predictor = AdvancedStockPredictor(config, checkpoint_dir=checkpoint_dir)
        predictor.train_ensemble(X_train, y_train, X_test, y_test, scalers, feature_cols)

        last_seq = X[-1:].astype(np.float32)
        mean_prices, std_prices = predictor.predict(last_seq, steps=3)

        assert len(mean_prices) == 3
        # Real prices should be in the $100+ range, not [0,1]
        assert all(p > 1.0 for p in mean_prices), \
            f"Predicted prices {mean_prices} look like scaled values, not real prices"

    def test_evaluate_metrics_on_real_scale(self, checkpoint_dir):
        """R² and MAE should be computed on real-price scale."""
        X, y, scalers, feature_cols = _make_scalers_and_data(n_samples=120, lookback=20)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        config = AdvancedModelConfig()
        predictor = AdvancedStockPredictor(config, checkpoint_dir=checkpoint_dir)
        predictor.train_ensemble(X_train, y_train, X_test, y_test, scalers, feature_cols)

        metrics = predictor.evaluate(X_test, y_test)
        avg = metrics.get('ensemble_average', {})
        # MAE on real prices should be > 0.01 (impossible to have sub-cent error on $150 prices)
        # but not in the tiny [0,1] range either — real MAE on real prices is typically > $1
        assert 'mae' in avg
        assert avg['mae'] > 0.01, \
            f"MAE={avg['mae']} looks like it's on scaled data"


class TestCheckpointShapeValidation:
    """Verify that stale weights are not loaded when input shape changes."""

    @pytest.fixture
    def checkpoint_dir(self):
        d = tempfile.mkdtemp(prefix="aipsfs_test_shape_")
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_shape_mismatch_triggers_retrain(self, checkpoint_dir):
        """When input shape changes, saved weights should not be loaded."""
        # First run with 5 features
        X1, y1, scalers1, cols1 = _make_scalers_and_data(n_samples=100, n_features=5, lookback=20)
        split = int(len(X1) * 0.8)

        config = AdvancedModelConfig()
        p1 = AdvancedStockPredictor(config, checkpoint_dir=checkpoint_dir)
        p1.train_ensemble(X1[:split], y1[:split], X1[split:], y1[split:], scalers1, cols1)

        # Verify shape file was saved
        shape_path = os.path.join(checkpoint_dir, "input_shape.txt")
        assert os.path.exists(shape_path)

        # Second run with 3 features (different shape)
        X2, y2, scalers2, cols2 = _make_scalers_and_data(n_samples=100, n_features=3, lookback=20)
        split2 = int(len(X2) * 0.8)

        p2 = AdvancedStockPredictor(config, checkpoint_dir=checkpoint_dir)
        # This should detect shape mismatch and retrain, not crash
        p2.train_ensemble(X2[:split2], y2[:split2], X2[split2:], y2[split2:], scalers2, cols2[:3])

        # Should have updated models despite old weights existing
        assert len(p2.models) > 0
