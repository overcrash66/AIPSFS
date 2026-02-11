# advanced_model.py
import os
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf

from aipsfs.config import AdvancedModelConfig


# ----------------------------------------------------------------------
# Custom Attention Layer (kept for compatibility with original setups)
# ----------------------------------------------------------------------
class AttentionLayer(tf.keras.layers.Layer):
	"""Simple additive attention over time dimension."""

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.W = None
		self.b = None

	def build(self, input_shape):
		# input_shape = (batch, timesteps, features)
		self.W = self.add_weight(
			name="att_weight",
			shape=(int(input_shape[-1]), 1),
			initializer="glorot_uniform",
			trainable=True,
		)
		self.b = self.add_weight(
			name="att_bias",
			shape=(int(input_shape[1]), 1),
			initializer="zeros",
			trainable=True,
		)
		super().build(input_shape)

	def call(self, x):
		# x: (B, T, F)
		# scores: (B, T, 1)
		scores = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
		# weights over time
		a = tf.keras.backend.softmax(scores, axis=1)
		# context: weighted sum over time -> (B, F)
		context = tf.reduce_sum(x * a, axis=1)
		return context


# ----------------------------------------------------------------------
# Advanced Ensemble Predictor
# ----------------------------------------------------------------------
class AdvancedStockPredictor:
	"""
	Ensemble predictor using multiple deep learning architectures.
	- Does NOT depend on AdvancedModelConfig.lookback_days or feature_columns.
	- Infers input shape from X_train at train time.
	"""

	def __init__(self, config: AdvancedModelConfig, checkpoint_dir: str = "models/ensemble"):
		self.config = config
		self.checkpoint_dir = checkpoint_dir
		os.makedirs(self.checkpoint_dir, exist_ok=True)

		self.models: Dict[str, tf.keras.Model] = {}
		# Will be set during training from data (timesteps, features)
		self.input_shape: Optional[Tuple[int, int]] = None
		# Scalers and feature columns for inverse-transforming predictions
		self.scalers: Optional[Dict] = None
		self.feature_cols: Optional[List[str]] = None

	# ------------------------------------------------------------------
	# Model Builders (kept close to typical originals)
	# ------------------------------------------------------------------
	def build_lstm(self, input_shape):
		model = tf.keras.Sequential([
			tf.keras.layers.Input(shape=input_shape),
			tf.keras.layers.LSTM(64, return_sequences=True),
			tf.keras.layers.Dropout(0.2),
			tf.keras.layers.LSTM(32),
			tf.keras.layers.Dense(1),
		])
		model.compile(optimizer="adam", loss="mse", metrics=["mae"])
		return model

	def build_gru(self, input_shape):
		model = tf.keras.Sequential([
			tf.keras.layers.Input(shape=input_shape),
			tf.keras.layers.GRU(64, return_sequences=True),
			tf.keras.layers.Dropout(0.2),
			tf.keras.layers.GRU(32),
			tf.keras.layers.Dense(1),
		])
		model.compile(optimizer="adam", loss="mse", metrics=["mae"])
		return model

	def build_cnn_lstm(self, input_shape):
		model = tf.keras.Sequential([
			tf.keras.layers.Input(shape=input_shape),
			tf.keras.layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu"),
			tf.keras.layers.MaxPooling1D(pool_size=2),
			tf.keras.layers.LSTM(50),
			tf.keras.layers.Dense(1),
		])
		model.compile(optimizer="adam", loss="mse", metrics=["mae"])
		return model

	def build_lstm_attention(self, input_shape):
		inputs = tf.keras.layers.Input(shape=input_shape)
		x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
		x = AttentionLayer()(x)
		x = tf.keras.layers.Dense(32, activation="relu")(x)
		outputs = tf.keras.layers.Dense(1)(x)
		model = tf.keras.Model(inputs, outputs)
		model.compile(optimizer="adam", loss="mse", metrics=["mae"])
		return model

	def build_hybrid(self, input_shape):
		inputs = tf.keras.layers.Input(shape=input_shape)
		x1 = tf.keras.layers.LSTM(32, return_sequences=True)(inputs)
		x2 = tf.keras.layers.GRU(32, return_sequences=True)(inputs)
		x = tf.keras.layers.Concatenate()([x1, x2])
		x = AttentionLayer()(x)
		x = tf.keras.layers.Dense(32, activation="relu")(x)
		outputs = tf.keras.layers.Dense(1)(x)
		model = tf.keras.Model(inputs, outputs)
		model.compile(optimizer="adam", loss="mse", metrics=["mae"])
		return model

	# ------------------------------------------------------------------
	# Training / Loading
	# ------------------------------------------------------------------
	def train_ensemble(self, X_train, y_train, X_test, y_test, scalers, feature_cols):
		"""
		Train each model in the ensemble or load weights if present.
		Derives input shape from X_train (timesteps, features).
		Stores scalers and feature_cols for inverse-transforming predictions.
		"""
		if X_train is None or len(X_train) == 0:
			raise ValueError("X_train is empty; cannot train ensemble.")
		self.input_shape = X_train.shape[1:]  # (timesteps, features)
		self.scalers = scalers
		self.feature_cols = feature_cols

		# Save expected input shape for checkpoint validation
		shape_path = os.path.join(self.checkpoint_dir, "input_shape.txt")
		with open(shape_path, "w") as f:
			f.write(f"{self.input_shape[0]},{self.input_shape[1]}")

		architectures = {
			"lstm": self.build_lstm,
			"gru": self.build_gru,
			"cnn_lstm": self.build_cnn_lstm,
			"lstm_attention": self.build_lstm_attention,
			"hybrid": self.build_hybrid,
		}

		# Check if saved checkpoints match current input shape
		shape_match = self._check_saved_shape_match()

		for name, builder in architectures.items():
			checkpoint_path = os.path.join(self.checkpoint_dir, f"{name}_best.weights.h5")
			logging.info(f"[Ensemble] Preparing {name} (checkpoint: {checkpoint_path})")

			try:
				model = builder(self.input_shape)

				if os.path.exists(checkpoint_path) and shape_match:
					try:
						logging.info(f"[{name}] Loading weights from {checkpoint_path}")
						status = model.load_weights(checkpoint_path)
						if status is not None and hasattr(status, "expect_partial"):
							status.expect_partial()
					except Exception as load_err:
						logging.warning(f"[{name}] Failed to load weights: {load_err}. Retraining...")
						self._train_and_save(model, X_train, y_train, X_test, y_test, checkpoint_path)
				else:
					if not shape_match and os.path.exists(checkpoint_path):
						logging.warning(f"[{name}] Input shape changed — retraining instead of loading stale weights.")
					self._train_and_save(model, X_train, y_train, X_test, y_test, checkpoint_path)

				self.models[name] = model
				logging.info(f"[{name}] Ready.")

			except Exception as e:
				# Do not crash the whole run if one architecture fails
				logging.error(f"[{name}] Error during build/train/load: {e}", exc_info=True)
				continue

	def _train_and_save(self, model, X_train, y_train, X_test, y_test, checkpoint_path: str):
		"""Train a single model and save best weights to disk."""
		callbacks = [
			tf.keras.callbacks.EarlyStopping(
				patience=getattr(self.config, "early_stopping_patience", 10),
				restore_best_weights=True
			),
			tf.keras.callbacks.ModelCheckpoint(
				filepath=checkpoint_path,
				save_best_only=True,
				save_weights_only=True
			)
		]

		history = model.fit(
			X_train, y_train,
			validation_data=(X_test, y_test),
			epochs=getattr(self.config, "epochs", 50),
			batch_size=getattr(self.config, "batch_size", 32),
			verbose=0,
			callbacks=callbacks
		)

		best_val = float(np.min(history.history.get("val_loss", [np.inf])))
		logging.info(f"Finished training. Best val_loss={best_val:.6f}")
		# Ensure a final save of best weights path (harmless if ModelCheckpoint already did)
		model.save_weights(checkpoint_path)

	def _check_saved_shape_match(self) -> bool:
		"""Check if saved checkpoint input shape matches current input shape."""
		shape_path = os.path.join(self.checkpoint_dir, "input_shape.txt")
		if not os.path.exists(shape_path):
			return False
		try:
			with open(shape_path, "r") as f:
				parts = f.read().strip().split(",")
				saved_shape = (int(parts[0]), int(parts[1]))
			if saved_shape == tuple(self.input_shape):
				logging.info(f"Checkpoint shape {saved_shape} matches current input shape.")
				return True
			logging.warning(
				f"Checkpoint shape {saved_shape} != current {tuple(self.input_shape)}. Will retrain."
			)
			return False
		except Exception as e:
			logging.warning(f"Could not read saved shape: {e}")
			return False

	# ------------------------------------------------------------------
	# Evaluation (with metrics expected by main.py)
	# ------------------------------------------------------------------
	def evaluate(self, X_test, y_test) -> Dict[str, Dict[str, float]]:
		"""
		Returns per-model metrics and an 'ensemble_average' dict with:
		- mse, mae, r2, directional_accuracy
		Metrics are computed on inverse-transformed (real price) values.
		"""
		if X_test is None or len(X_test) == 0:
			raise ValueError("X_test is empty; cannot evaluate.")

		# Get the Close scaler for inverse-transforming
		close_scaler = None
		if self.scalers and 'Close' in self.scalers:
			close_scaler = self.scalers['Close']

		metrics: Dict[str, Dict[str, float]] = {}

		# Helper functions
		def r2_score(y_true, y_pred):
			y_true = y_true.astype(np.float64)
			y_pred = y_pred.astype(np.float64)
			ss_res = np.sum((y_true - y_pred) ** 2)
			ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
			return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

		def directional_accuracy(y_true, y_pred):
			if len(y_true) < 2:
				return 0.0
			dy_true = np.sign(y_true[1:] - y_true[:-1])
			dy_pred = np.sign(y_pred[1:] - y_pred[:-1])
			return float(np.mean(dy_true == dy_pred))

		# Evaluate each model
		for name, model in self.models.items():
			try:
				y_pred_raw = model.predict(X_test, verbose=0).reshape(-1)
				y_true_raw = y_test.reshape(-1)

				# Inverse transform to real prices for meaningful metrics
				if close_scaler is not None:
					y_true = close_scaler.inverse_transform(
						y_true_raw.reshape(-1, 1)
					).flatten()
					y_pred = close_scaler.inverse_transform(
						y_pred_raw.reshape(-1, 1)
					).flatten()
				else:
					logging.warning(
						f"[{name}] No Close scaler — metrics are on scaled data."
					)
					y_true = y_true_raw
					y_pred = y_pred_raw

				mse = float(np.mean((y_true - y_pred) ** 2))
				mae = float(np.mean(np.abs(y_true - y_pred)))
				r2 = float(r2_score(y_true, y_pred))
				da = float(directional_accuracy(y_true, y_pred))

				metrics[name] = {
					"mse": mse,
					"mae": mae,
					"r2": r2,
					"directional_accuracy": da,
				}

				logging.info(
					f"[{name}] mse={mse:.2f}, mae={mae:.2f}, r2={r2:.4f}, dir_acc={da:.4f}"
				)
			except Exception as e:
				logging.error(f"[{name}] Evaluation error: {e}", exc_info=True)

		# Ensemble average over available models
		if metrics:
			keys = ["mse", "mae", "r2", "directional_accuracy"]
			metrics["ensemble_average"] = {
				k: float(np.mean([m[k] for m in metrics.values() if k in m]))
				for k in keys
			}

		return metrics

	# ------------------------------------------------------------------
	# Prediction
	# ------------------------------------------------------------------
	def predict(self, last_sequence: np.ndarray, steps: int = 5):
		"""
		Predict future values using autoregressive ensemble averaging.
		Returns (mean_prices_array, std_prices_array) each length == steps.

		Performs multi-step autoregressive forecasting: each step's ensemble
		prediction is fed back as input for the next step. Predictions are
		inverse-transformed to real price scale.
		"""
		if not self.models:
			raise ValueError("No models trained. Call train_ensemble() first.")
		if last_sequence is None or last_sequence.ndim != 3:
			raise ValueError("last_sequence must be a 3D array: (1, timesteps, features).")

		# Determine Close index for autoregressive updates
		close_idx = 0
		if self.feature_cols and 'Close' in self.feature_cols:
			close_idx = self.feature_cols.index('Close')

		current_sequence = last_sequence.copy()
		mean_scaled_list = []
		std_scaled_list = []

		for step in range(int(max(1, steps))):
			per_model_preds = []
			for name, model in self.models.items():
				try:
					pred = model.predict(current_sequence, verbose=0)
					per_model_preds.append(pred.reshape(-1)[0])
				except Exception as e:
					logging.error(f"[{name}] Prediction error at step {step}: {e}")

			if not per_model_preds:
				logging.error("All ensemble members failed — stopping forecast.")
				break

			arr = np.array(per_model_preds)
			mean_val = float(np.mean(arr))
			std_val = float(np.std(arr))
			mean_scaled_list.append(mean_val)
			std_scaled_list.append(std_val)

			# Autoregressive: shift sequence and insert new prediction
			new_features = current_sequence[0, -1, :].copy()
			new_features[close_idx] = mean_val
			new_seq = np.vstack([current_sequence[0, 1:], new_features])
			current_sequence = np.array([new_seq])

		# Inverse-transform from scaled [0,1] to real prices
		mean_scaled = np.array(mean_scaled_list, dtype=np.float32)
		std_scaled = np.array(std_scaled_list, dtype=np.float32)

		if self.scalers and 'Close' in self.scalers:
			close_scaler = self.scalers['Close']
			mean_prices = close_scaler.inverse_transform(
				mean_scaled.reshape(-1, 1)
			).flatten()
			# Scale std by the scaler's range for interpretability
			scale = close_scaler.data_max_[0] - close_scaler.data_min_[0]
			std_prices = (std_scaled * scale).astype(np.float32)
		else:
			logging.warning("No Close scaler available — returning raw scaled predictions.")
			mean_prices = mean_scaled
			std_prices = std_scaled

		return mean_prices, std_prices

	# ------------------------------------------------------------------
	# Metadata (optional logging for reproducibility)
	# ------------------------------------------------------------------
	def _save_ensemble_metadata(self):
		"""Save a simple text file with info about the ensemble."""
		meta_path = os.path.join(self.checkpoint_dir, "ensemble_metadata.txt")
		try:
			with open(meta_path, "w") as f:
				for name, model in self.models.items():
					f.write(f"{name}: layers={len(model.layers)}\n")
			logging.info(f"Saved ensemble metadata to {meta_path}")
		except Exception as e:
			logging.warning(f"Failed to save ensemble metadata: {e}")
