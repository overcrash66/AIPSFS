# advanced_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Dense, Dropout, LayerNormalization,
    Attention, GlobalAveragePooling1D, Concatenate,
    Bidirectional, Conv1D, MaxPooling1D, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score
)
import logging
import os
import json
import joblib
from typing import Dict, List, Optional, Tuple

class AdvancedStockPredictor:
    """
    Advanced stock prediction model with an ensemble of LSTM, GRU, and CNN-LSTM networks,
    featuring attention mechanisms, robust training, and persistence capabilities.
    """

    def __init__(self, config):
        """
        Initializes the AdvancedStockPredictor.

        Args:
            config: A configuration object or dictionary with hyperparameters.
                    Expected keys: 'lstm_units_layer1', 'lstm_units_layer2',
                    'gru_units_layer1', 'gru_units_layer2', 'epochs', 'batch_size',
                    'early_stopping_patience', 'learning_rate', 'forecast_steps',
                    'target_col' (optional, defaults to 'Close').
        """
        self.config = config
        self.models = {}
        self.scalers = {}
        self.histories = {}
        self.feature_cols = []
        self.input_shape = None
        self.model_dir = "models"
        self.model_type = 'Ensemble'
        self.target_col = getattr(config, 'target_col', 'Close')
        self.best_model_name = None
        self.learning_rate = 0.001

        os.makedirs(self.model_dir, exist_ok=True)
        self._configure_gpu()

    @classmethod
    def load(cls, model_dir: str = "models"):
        """
        Loads a trained predictor instance from a directory.

        Args:
            model_dir (str): The directory containing the saved model metadata and files.

        Returns:
            An instance of AdvancedStockPredictor with loaded models and metadata.
        """
        metadata_path = os.path.join(model_dir, 'ensemble_metadata.json')
        if not os.path.exists(metadata_path):
            logging.error(f"Metadata file not found at {metadata_path}")
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Create a dummy config for initialization, real config is in metadata
        class DummyConfig:
            pass
        config = DummyConfig()
        for key, value in metadata['config'].items():
            setattr(config, key, value)

        predictor = cls(config)
        predictor.model_dir = model_dir
        predictor.feature_cols = metadata['feature_cols']
        predictor.target_col = metadata['target_col']
        predictor.best_model_name = metadata['best_model_name']
        predictor.histories = metadata.get('histories', {})

        # Load scalers
        for name, path in metadata['scalers'].items():
            predictor.scalers[name] = joblib.load(path)

        # Load models
        for name, path in metadata['models'].items():
            logging.info(f"Loading model '{name}' from {path}...")
            predictor.models[name] = load_model(path)

        logging.info("AdvancedStockPredictor loaded successfully.")
        return predictor

    def _configure_gpu(self):
        """Configure TensorFlow to use GPU if available with memory growth."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"Configured {len(gpus)} GPU(s) for memory growth.")
            except RuntimeError as e:
                logging.error(f"GPU configuration error: {e}")
        else:
            logging.info("No GPU devices found. Running on CPU.")

    def build_lstm_model(self, input_shape, name):
        """Build an enhanced LSTM model with attention."""
        inputs = Input(shape=input_shape, name=f"{name}_input")
        x = LSTM(
            self.config.lstm_units_layer1, return_sequences=True,
            recurrent_dropout=0.2, kernel_regularizer=l1_l2(0.01)
        )(inputs)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        x = LSTM(
            self.config.lstm_units_layer2, return_sequences=True,
            recurrent_dropout=0.2, kernel_regularizer=l1_l2(0.01)
        )(x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        attention = Attention()([x, x])
        attention = LayerNormalization()(attention)
        x = GlobalAveragePooling1D()(attention)
        x = Dense(64, activation='relu', kernel_regularizer=l1_l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        output = Dense(1, name=f"{name}_output")(x)
        model = Model(inputs=inputs, outputs=output, name=name)
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
        return model

    def build_gru_model(self, input_shape: Tuple[int, int], name: str = "gru") -> Model:
        """Build a GRU-based model with attention."""
        inputs = Input(shape=input_shape, name=f"{name}_input")
        x = GRU(
            self.config.gru_units_layer1, return_sequences=True,
            recurrent_dropout=0.2, kernel_regularizer=l1_l2(0.01)
        )(inputs)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        x = GRU(
            self.config.gru_units_layer2, return_sequences=True,
            recurrent_dropout=0.2, kernel_regularizer=l1_l2(0.01)
        )(x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        attention = Attention()([x, x])
        attention = LayerNormalization()(attention)
        x = GlobalAveragePooling1D()(attention)
        x = Dense(64, activation='relu', kernel_regularizer=l1_l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        output = Dense(1, name=f"{name}_output")(x)
        model = Model(inputs=inputs, outputs=output, name=name)
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
        return model

    def build_cnn_lstm_model(self, input_shape: Tuple[int, int], name: str = "cnn_lstm") -> Model:
        """Build a hybrid CNN-LSTM model with attention."""
        inputs = Input(shape=input_shape, name=f"{name}_input")
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(self.config.lstm_units_layer1, return_sequences=True))(x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        attention = Attention()([x, x])
        attention = LayerNormalization()(attention)
        x = GlobalAveragePooling1D()(attention)
        x = Dense(64, activation='relu', kernel_regularizer=l1_l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        output = Dense(1, name=f"{name}_output")(x)
        model = Model(inputs=inputs, outputs=output, name=name)
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
        return model

    def build_ensemble(self, input_shape: Tuple[int, int]):
        """Build an ensemble of different models."""
        logging.info("Building ensemble models...")
        self.models['lstm'] = self.build_lstm_model(input_shape, "lstm")
        self.models['gru'] = self.build_gru_model(input_shape, "gru")
        self.models['cnn_lstm'] = self.build_cnn_lstm_model(input_shape, "cnn_lstm")
        logging.info(f"Built {len(self.models)} models: {list(self.models.keys())}")
        return self.models

    def train_ensemble(self, X_train, y_train, X_test, y_test, scalers, feature_cols):
        """Train an ensemble of models and select the best one."""
        self.input_shape = (X_train.shape[1], X_train.shape[2])
        self.feature_cols = feature_cols
        self.scalers = scalers

        self.build_ensemble(self.input_shape)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.config.early_stopping_patience, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
        ]

        for name, model in self.models.items():
            logging.info(f"--- Training {name.upper()} model ---")
            checkpoint_path = os.path.join(self.model_dir, f"{name}_best_model.h5")
            checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
            
            tf.get_logger().setLevel('ERROR')
            history = model.fit(
                X_train, y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_data=(X_test, y_test),
                callbacks=callbacks + [checkpoint],
                verbose=1
            )
            tf.get_logger().setLevel('INFO')
            self.histories[name] = history.history
            model.load_weights(checkpoint_path)
            model.save(os.path.join(self.model_dir, f"{name}_final_model.h5"))
            logging.info(f"Completed training for {name} model.")

        self._determine_best_model(X_test, y_test)
        self._save_ensemble_metadata()
        return self.histories

    def _determine_best_model(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate all models to determine the best one based on validation loss."""
        logging.info("Determining the best model from the ensemble...")
        best_score = float('inf')
        best_model_name = None
        for name, model in self.models.items():
            loss, _, _ = model.evaluate(X_test, y_test, verbose=0)
            logging.info(f"Model '{name}' validation loss: {loss:.6f}")
            if loss < best_score:
                best_score = loss
                best_model_name = name
        
        self.best_model_name = best_model_name
        logging.info(f"The best performing model is '{self.best_model_name}' with a loss of {best_score:.6f}.")

    def _save_ensemble_metadata(self):
        """Save ensemble metadata including model paths, scalers, and configuration."""
        scalers_dir = os.path.join(self.model_dir, 'scalers')
        os.makedirs(scalers_dir, exist_ok=True)
        
        scaler_paths = {}
        for name, scaler in self.scalers.items():
            scaler_path = os.path.join(scalers_dir, f'{name}.joblib')
            joblib.dump(scaler, scaler_path)
            scaler_paths[name] = scaler_path
        
        model_paths = {name: os.path.join(self.model_dir, f"{name}_final_model.h5") for name in self.models.keys()}
        
        metadata = {
            'model_type': self.model_type,
            'feature_cols': self.feature_cols,
            'target_col': self.target_col,
            'scalers': scaler_paths,
            'models': model_paths,
            'best_model_name': self.best_model_name,
            'histories': self.histories,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else dict(self.config)
        }
        
        metadata_path = os.path.join(self.model_dir, 'ensemble_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"Ensemble metadata saved to {metadata_path}")

    def predict(self, last_sequence: np.ndarray, steps: Optional[int] = None, use_best_model: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make future predictions using the ensemble average or the single best model.

        Args:
            last_sequence (np.ndarray): The last sequence of data for starting prediction.
            steps (int, optional): Number of future steps to predict. Defaults to config.
            use_best_model (bool): If True, uses only the best model. If False, uses the ensemble average.

        Returns:
            A tuple of (mean_predictions, prediction_std). `prediction_std` is zero if using best model.
        """
        if steps is None:
            steps = self.config.forecast_steps

        if use_best_model and self.best_model_name:
            logging.info(f"Predicting using the best model: {self.best_model_name}")
            model = self.models[self.best_model_name]
            predictions = self._predict_with_model(model, last_sequence, steps)
            return predictions, np.zeros_like(predictions) # No variance with one model
        
        logging.info("Predicting using the ensemble average...")
        ensemble_predictions = []
        for name, model in self.models.items():
            try:
                predictions = self._predict_with_model(model, last_sequence, steps)
                if predictions.size > 0:
                    ensemble_predictions.append(predictions)
            except Exception as e:
                logging.error(f"Error predicting with {name}: {str(e)}")
        
        if not ensemble_predictions:
            return np.array([]), np.array([])
        
        ensemble_predictions = np.array(ensemble_predictions)
        mean_predictions = np.mean(ensemble_predictions, axis=0)
        prediction_std = np.std(ensemble_predictions, axis=0)
        
        logging.info(f"Ensemble prediction for last step: {mean_predictions[-1]:.2f} +/- {prediction_std[-1]:.4f}")
        return mean_predictions, prediction_std

    def _predict_with_model(self, model: Model, last_sequence: np.ndarray, steps: int) -> np.ndarray:
        """Helper function to predict with a single model."""
        future_prices_scaled = []
        current_sequence = last_sequence.copy()
        close_scaler = self.scalers[self.target_col]
        close_idx = self.feature_cols.index(self.target_col)
        
        for _ in range(steps):
            pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
            future_prices_scaled.append(pred_scaled)
            
            new_features_scaled = current_sequence[0, -1, :].copy()
            new_features_scaled[close_idx] = pred_scaled
            
            new_sequence = np.vstack([current_sequence[0, 1:], new_features_scaled])
            current_sequence = np.array([new_sequence])
        
        future_prices_scaled = np.array(future_prices_scaled).reshape(-1, 1)
        return close_scaler.inverse_transform(future_prices_scaled).flatten()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate all models and the ensemble on the test set."""
        all_metrics = {}
        close_scaler = self.scalers.get(self.target_col)
        if not close_scaler:
            raise ValueError(f"Scaler for target column '{self.target_col}' not found.")
        
        y_test_inv = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_test_diff = np.diff(y_test_inv)
        
        model_predictions_scaled = []
        for name, model in self.models.items():
            y_pred_scaled = model.predict(X_test, verbose=0)
            model_predictions_scaled.append(y_pred_scaled)
            y_pred_inv = close_scaler.inverse_transform(y_pred_scaled).flatten()
            
            y_pred_diff = np.diff(y_pred_inv)
            directional_accuracy = np.mean((y_test_diff > 0) == (y_pred_diff > 0)) * 100

            metrics = {
                'mse': mean_squared_error(y_test_inv, y_pred_inv),
                'mae': mean_absolute_error(y_test_inv, y_pred_inv),
                'r2': r2_score(y_test_inv, y_pred_inv),
                'directional_accuracy': directional_accuracy   
            }
            all_metrics[name] = metrics
            logging.info(f"Metrics for '{name}': MAE={metrics['mae']:.2f}, R2={metrics['r2']:.2f}, DirAcc={metrics['directional_accuracy']:.2f}%")
        
        # Evaluate ensemble performance
        if len(self.models) > 1:
            ensemble_pred_scaled = np.mean(model_predictions_scaled, axis=0)
            ensemble_pred_inv = close_scaler.inverse_transform(ensemble_pred_scaled).flatten()
            
            ensemble_pred_diff = np.diff(ensemble_pred_inv)
            ensemble_dir_acc = np.mean((y_test_diff > 0) == (ensemble_pred_diff > 0)) * 100
            
            ensemble_metrics = {
                'mse': mean_squared_error(y_test_inv, ensemble_pred_inv),
                'mae': mean_absolute_error(y_test_inv, ensemble_pred_inv),
                'r2': r2_score(y_test_inv, ensemble_pred_inv),
                'directional_accuracy': ensemble_dir_acc
            }
            all_metrics['ensemble_average'] = ensemble_metrics
            logging.info(f"Metrics for 'Ensemble Average': MAE={ensemble_metrics['mae']:.2f}, R2={ensemble_metrics['r2']:.2f}, DirAcc={ensemble_metrics['directional_accuracy']:.2f}%")
            
        return all_metrics