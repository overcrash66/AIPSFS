# advanced_model.py
import os
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf

from config import AdvancedModelConfig


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
		"""
		if X_train is None or len(X_train) == 0:
			raise ValueError("X_train is empty; cannot train ensemble.")
		self.input_shape = X_train.shape[1:]  # (timesteps, features)

		architectures = {
			"lstm": self.build_lstm,
			"gru": self.build_gru,
			"cnn_lstm": self.build_cnn_lstm,
			"lstm_attention": self.build_lstm_attention,
			"hybrid": self.build_hybrid,
		}

		for name, builder in architectures.items():
			checkpoint_path = os.path.join(self.checkpoint_dir, f"{name}_best.weights.h5")
			logging.info(f"[Ensemble] Preparing {name} (checkpoint: {checkpoint_path})")

			try:
				model = builder(self.input_shape)

				if os.path.exists(checkpoint_path):
					try:
						logging.info(f"[{name}] Loading weights from {checkpoint_path}")
						status = model.load_weights(checkpoint_path)
						if status is not None and hasattr(status, "expect_partial"):
							status.expect_partial()
					except Exception as load_err:
						logging.warning(f"[{name}] Failed to load weights: {load_err}. Retraining...")
						self._train_and_save(model, X_train, y_train, X_test, y_test, checkpoint_path)
				else:
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

	# ------------------------------------------------------------------
	# Evaluation (with metrics expected by main.py)
	# ------------------------------------------------------------------
	def evaluate(self, X_test, y_test) -> Dict[str, Dict[str, float]]:
		"""
		Returns per-model metrics and an 'ensemble_average' dict with:
		- mse
		- mae
		- r2
		- directional_accuracy
		"""
		if X_test is None or len(X_test) == 0:
			raise ValueError("X_test is empty; cannot evaluate.")

		metrics: Dict[str, Dict[str, float]] = {}

		# Helper functions
		def r2_score(y_true, y_pred):
			y_true = y_true.astype(np.float64)
			y_pred = y_pred.astype(np.float64)
			ss_res = np.sum((y_true - y_pred) ** 2)
			ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
			return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

		def directional_accuracy(y_true, y_pred):
			# Compares direction of consecutive changes in y_pred vs y_true
			if len(y_true) < 2:
				return 0.0
			dy_true = np.sign(y_true[1:] - y_true[:-1])
			dy_pred = np.sign(y_pred[1:] - y_pred[:-1])
			return float(np.mean(dy_true == dy_pred))

		# Evaluate each model
		for name, model in self.models.items():
			try:
				y_pred = model.predict(X_test, verbose=0).reshape(-1)
				y_true = y_test.reshape(-1)

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
					f"[{name}] mse={mse:.6f}, mae={mae:.6f}, r2={r2:.4f}, dir_acc={da:.4f}"
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
		Predict future values using ensemble averaging.
		Returns (mean_prices_array, std_prices_array) each length == steps.

		Note: Without a full feature roll-forward pipeline, this performs a
		one-step ensemble prediction and repeats it for 'steps' so downstream
		code has a vector of length 'steps'.
		"""
		if not self.models:
			raise ValueError("No models trained. Call train_ensemble() first.")
		if last_sequence is None or last_sequence.ndim != 3:
			raise ValueError("last_sequence must be a 3D array: (1, timesteps, features).")

		per_model_preds = []
		for name, model in self.models.items():
			try:
				pred = model.predict(last_sequence, verbose=0)	# shape (1, 1)
				per_model_preds.append(pred.reshape(-1)[0])
			except Exception as e:
				logging.error(f"[{name}] Prediction error: {e}", exc_info=True)

		if not per_model_preds:
			raise RuntimeError("All ensemble members failed to predict.")

		per_model_preds = np.array(per_model_preds)	 # shape (n_models,)
		mean_one_step = float(np.mean(per_model_preds))
		std_one_step = float(np.std(per_model_preds))

		# Produce steps-length vectors to satisfy downstream consumers
		mean_series = np.full(shape=(int(max(1, steps))), fill_value=mean_one_step, dtype=np.float32)
		std_series = np.full(shape=(int(max(1, steps))), fill_value=std_one_step, dtype=np.float32)

		return mean_series, std_series

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
