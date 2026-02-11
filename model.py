# model.py
import io
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, LayerNormalization, 
    Attention, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    mean_absolute_percentage_error, r2_score
)
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

class StockPredictor:
    """
    Enhanced stock prediction model with LSTM architecture and attention mechanism.
    Includes comprehensive evaluation, model persistence, and forecasting capabilities.
    """
    
    def __init__(self, config):
        """
        Initialize the StockPredictor with configuration.
        
        Args:
            config (ModelConfig): Configuration object with model parameters
        """
        self.config = config
        self.model = None
        self.scalers = {}
        self.history = None
        self.feature_cols = []
        self.input_shape = None
        
        # Configure GPU if available
        self._configure_gpu()
        
    def _configure_gpu(self):
        """Configure TensorFlow to use GPU if available with memory growth."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info("GPU configured for memory growth.")
            except RuntimeError as e:
                logging.error(f"GPU configuration error: {e}")
        else:
            logging.info("No GPU devices found. Running on CPU.")
    
    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build an enhanced LSTM model with attention mechanism.
        
        Args:
            input_shape (Tuple[int, int]): Shape of input sequences (timesteps, features)
            
        Returns:
            Model: Compiled Keras model
        """
        self.input_shape = input_shape
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # First LSTM layer with return sequences for attention
        lstm1 = LSTM(
            self.config.lstm_units_layer1, 
            return_sequences=True, 
            recurrent_dropout=0.2
        )(inputs)
        lstm1 = LayerNormalization()(lstm1)
        lstm1 = Dropout(0.3)(lstm1)
        
        # Second LSTM layer
        lstm2 = LSTM(
            self.config.lstm_units_layer2, 
            return_sequences=True, 
            recurrent_dropout=0.2
        )(lstm1)
        lstm2 = LayerNormalization()(lstm2)
        lstm2 = Dropout(0.2)(lstm2)
        
        # Attention mechanism
        attention = Attention()([lstm2, lstm2])
        attention = LayerNormalization()(attention)
        
        # Global average pooling
        avg_pool = GlobalAveragePooling1D()(attention)
        
        # Dense layers
        dense1 = Dense(64, activation='relu')(avg_pool)
        dense1 = Dropout(0.2)(dense1)
        
        dense2 = Dense(32, activation='relu')(dense1)
        dense2 = Dropout(0.1)(dense2)
        
        # Output layer
        output = Dense(1)(dense2)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        self.model = model
        logging.info(f"Model built with input shape: {input_shape}")
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, List[float]]:
        """
        Train the model with early stopping.
        
        Args:
            X_train (np.ndarray): Training input sequences
            y_train (np.ndarray): Training target values
            X_test (np.ndarray): Validation input sequences
            y_test (np.ndarray): Validation target values
            
        Returns:
            Dict[str, List[float]]: Training history
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")
            
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train model
        logging.info("Starting model training...")
        history = self.model.fit(
            X_train, y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=1
        )
        
        self.history = history.history
        logging.info("Model training completed.")
        return self.history
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, 
                      scalers: Dict, feature_cols: List[str]) -> Dict[str, float]:
        """
        Comprehensive model evaluation with multiple metrics.
        
        Args:
            X_test (np.ndarray): Test input sequences
            y_test (np.ndarray): Test target values
            scalers (Dict): Scalers used for data normalization
            feature_cols (List[str]): List of feature column names
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained.")
            
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Inverse transform predictions and actual values
        close_scaler = scalers['Close']
        y_test_inv = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_inv = close_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)
        
        # Directional accuracy (percentage of times prediction direction is correct)
        y_test_diff = np.diff(y_test_inv)
        y_pred_diff = np.diff(y_pred_inv)
        directional_accuracy = np.mean((y_test_diff > 0) == (y_pred_diff > 0)) * 100
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2),
            'directional_accuracy': float(directional_accuracy)
        }
        
        logging.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    def predict(self, last_sequence: np.ndarray, 
                scalers: Dict, feature_cols: List[str], 
                steps: Optional[int] = None) -> np.ndarray:
        """
        Forecast future prices using the trained model.
        
        Args:
            last_sequence (np.ndarray): Last sequence from historical data
            scalers (Dict): Scalers used for data normalization
            feature_cols (List[str]): List of feature column names
            steps (Optional[int]): Number of steps to forecast (uses config default if None)
            
        Returns:
            np.ndarray: Forecasted prices
        """
        if self.model is None:
            raise ValueError("Model has not been trained.")
            
        if steps is None:
            steps = self.config.forecast_steps
            
        if last_sequence is None or last_sequence.size == 0:
            raise ValueError("No valid sequence provided for forecasting.")
            
        if 'Close' not in scalers or 'Close' not in feature_cols:
            raise ValueError("Scaler or feature column for 'Close' price not found.")
            
        future_prices_scaled = []
        current_sequence = last_sequence.copy()
        close_scaler = scalers['Close']
        close_idx = feature_cols.index('Close')
        
        logging.info(f"Starting forecast for {steps} steps...")
        
        for i in range(steps):
            try:
                # Predict the next step's scaled Close price
                pred_scaled = self.model.predict(current_sequence, verbose=0)[0, 0]
                future_prices_scaled.append(pred_scaled)
                
                # Update the sequence for the next prediction
                new_features_scaled = current_sequence[0, -1, :].copy()
                new_features_scaled[close_idx] = pred_scaled
                
                # Shift the window and add the new prediction
                new_sequence = np.vstack([current_sequence[0, 1:], new_features_scaled])
                current_sequence = np.array([new_sequence])
                
            except Exception as e:
                logging.error(f"Error during forecasting step {i+1}: {str(e)}")
                break
                
        if future_prices_scaled:
            # Inverse transform the scaled predictions to original price scale
            future_prices_scaled = np.array(future_prices_scaled).reshape(-1, 1)
            return close_scaler.inverse_transform(future_prices_scaled).flatten()
            
        return np.array([])
    
    def save_model(self, filepath: str):
        """
        Save the trained model and scalers to disk.
        
        Args:
            filepath (str): Base path for saving model components
        """
        if self.model is None:
            raise ValueError("No model to save.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        model_path = f"{filepath}_model.h5"
        self.model.save(model_path)
        
        # Save scalers and metadata
        metadata = {
            'scalers': self.scalers,
            'feature_cols': self.feature_cols,
            'input_shape': self.input_shape,
            'config': self.config.__dict__,
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = f"{filepath}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
            
        logging.info(f"Model saved to {model_path}")
        logging.info(f"Metadata saved to {metadata_path}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model and associated metadata from disk.
        
        Args:
            filepath (str): Base path for loading model components
        """
        # Load model
        model_path = f"{filepath}_model.h5"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model = load_model(model_path)
        
        # Load metadata
        metadata_path = f"{filepath}_metadata.pkl"
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            
        self.scalers = metadata['scalers']
        self.feature_cols = metadata['feature_cols']
        self.input_shape = metadata['input_shape']
        self.history = metadata.get('history', None)
        
        logging.info(f"Model loaded from {model_path}")
        logging.info(f"Metadata loaded from {metadata_path}")
        logging.info(f"Model timestamp: {metadata.get('timestamp', 'Unknown')}")
    
    def get_model_summary(self) -> str:
        """
        Get a string summary of the model architecture.
        
        Returns:
            str: Model summary
        """
        if self.model is None:
            return "Model has not been built yet."
            
        # Capture model summary as string
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()