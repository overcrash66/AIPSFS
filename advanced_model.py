# advanced_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Dense, Dropout, LayerNormalization, 
    Attention, GlobalAveragePooling1D, Concatenate, Add,
    Bidirectional, Conv1D, MaxPooling1D, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    mean_absolute_percentage_error, r2_score
)
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os

class AdvancedStockPredictor:
    """Advanced stock prediction model with ensemble and attention mechanisms."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scalers = {}
        self.history = None
        self.feature_cols = []
        self.input_shape = None
        self.ensemble_models = []
        
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
    
    def build_hybrid_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build a hybrid CNN-LSTM model with attention."""
        inputs = Input(shape=input_shape)
        
        # CNN layers for feature extraction
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Bidirectional LSTM layers
        x = Bidirectional(LSTM(self.config.lstm_units_layer1, return_sequences=True))(x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Bidirectional(LSTM(self.config.lstm_units_layer2, return_sequences=True))(x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Attention mechanism
        attention = Attention()([x, x])
        attention = LayerNormalization()(attention)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(attention)
        
        # Dense layers with residual connections
        dense1 = Dense(128, activation='relu', kernel_regularizer=l1_l2(0.01))(x)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.3)(dense1)
        
        dense2 = Dense(64, activation='relu', kernel_regularizer=l1_l2(0.01))(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(0.2)(dense2)
        
        # Residual connection
        if x.shape[-1] == dense2.shape[-1]:
            dense2 = Add()([x, dense2])
        
        # Output layer
        output = Dense(1)(dense2)
        
        model = Model(inputs=inputs, outputs=output)
        
        # Learning rate scheduling
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )
        
        optimizer = Adam(learning_rate=lr_schedule)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def build_gru_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build a GRU-based model with attention."""
        inputs = Input(shape=input_shape)
        
        # GRU layers
        x = GRU(self.config.lstm_units_layer1, return_sequences=True)(inputs)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = GRU(self.config.lstm_units_layer2, return_sequences=True)(x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Attention mechanism
        attention = Attention()([x, x])
        attention = LayerNormalization()(attention)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(attention)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        output = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=output)
        
        optimizer = Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def build_ensemble_models(self, input_shape: Tuple[int, int]) -> List[Model]:
        """Build an ensemble of different models."""
        models = []
        
        # Model 1: Hybrid CNN-LSTM
        model1 = self.build_hybrid_model(input_shape)
        models.append(('hybrid_cnn_lstm', model1))
        
        # Model 2: GRU with attention
        model2 = self.build_gru_model(input_shape)
        models.append(('gru_attention', model2))
        
        # Model 3: Enhanced LSTM (original with improvements)
        model3 = self.build_enhanced_lstm(input_shape)
        models.append(('enhanced_lstm', model3))
        
        return models
    
    def build_enhanced_lstm(self, input_shape: Tuple[int, int]) -> Model:
        """Build an enhanced LSTM model with attention and regularization."""
        inputs = Input(shape=input_shape)
        
        # LSTM layers with recurrent dropout
        x = LSTM(
            self.config.lstm_units_layer1, 
            return_sequences=True,
            recurrent_dropout=0.2,
            kernel_regularizer=l1_l2(0.01)
        )(inputs)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = LSTM(
            self.config.lstm_units_layer2, 
            return_sequences=True,
            recurrent_dropout=0.2,
            kernel_regularizer=l1_l2(0.01)
        )(x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Attention mechanism
        attention = Attention()([x, x])
        attention = LayerNormalization()(attention)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(attention)
        
        # Dense layers
        x = Dense(64, activation='relu', kernel_regularizer=l1_l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        output = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=output)
        
        optimizer = Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_test: np.ndarray, y_test: np.ndarray, 
                      scalers: Dict, feature_cols: List[str]) -> Dict:
        """Train an ensemble of models."""
        self.input_shape = (X_train.shape[1], X_train.shape[2])
        self.feature_cols = feature_cols
        self.scalers = scalers
        
        # Build ensemble models
        models = self.build_ensemble_models(self.input_shape)
        self.ensemble_models = models
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath='best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train each model
        model_histories = {}
        for name, model in models:
            logging.info(f"Training {name} model...")
            
            # Suppress TensorFlow logging during training
            tf.get_logger().setLevel('ERROR')
            history = model.fit(
                X_train, y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            tf.get_logger().setLevel('INFO')
            
            model_histories[name] = history.history
            
            # Load best model
            model.load_weights('best_model.h5')
            
            logging.info(f"Completed training {name} model")
        
        self.history = model_histories
        return model_histories
    
    def predict_ensemble(self, last_sequence: np.ndarray, 
                        scalers: Dict, feature_cols: List[str], 
                        steps: int = None) -> np.ndarray:
        """Make predictions using the ensemble of models."""
        if steps is None:
            steps = self.config.forecast_steps
            
        if last_sequence is None or last_sequence.size == 0:
            logging.warning("No valid sequence provided for forecasting.")
            return np.array([])
        
        if 'Close' not in scalers or 'Close' not in feature_cols:
            logging.error("Scaler or feature column for 'Close' price not found.")
            return np.array([])
        
        # Get predictions from each model
        ensemble_predictions = []
        close_scaler = scalers['Close']
        close_idx = feature_cols.index('Close')
        
        for name, model in self.ensemble_models:
            try:
                predictions = self._predict_with_model(
                    model, last_sequence, scalers, feature_cols, steps
                )
                if predictions.size > 0:
                    ensemble_predictions.append(predictions)
                    logging.info(f"{name} prediction: {predictions[-1]:.2f}")
            except Exception as e:
                logging.error(f"Error predicting with {name}: {str(e)}")
        
        if not ensemble_predictions:
            return np.array([])
        
        # Average the predictions
        ensemble_predictions = np.array(ensemble_predictions)
        mean_predictions = np.mean(ensemble_predictions, axis=0)
        
        # Calculate prediction variance (uncertainty)
        prediction_variance = np.var(ensemble_predictions, axis=0)
        
        logging.info(f"Ensemble prediction: {mean_predictions[-1]:.2f}")
        logging.info(f"Prediction variance: {prediction_variance[-1]:.4f}")
        
        return mean_predictions
    
    def _predict_with_model(self, model: Model, last_sequence: np.ndarray,
                           scalers: Dict, feature_cols: List[str], 
                           steps: int) -> np.ndarray:
        """Predict with a single model."""
        future_prices_scaled = []
        current_sequence = last_sequence.copy()
        close_scaler = scalers['Close']
        close_idx = feature_cols.index('Close')
        
        for i in range(steps):
            try:
                pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
                future_prices_scaled.append(pred_scaled)
                
                # Update the sequence for the next prediction
                new_features_scaled = current_sequence[0, -1, :].copy()
                new_features_scaled[close_idx] = pred_scaled
                
                new_sequence = np.vstack([current_sequence[0, 1:], new_features_scaled])
                current_sequence = np.array([new_sequence])
                
            except Exception as e:
                logging.error(f"Error during prediction step {i+1}: {str(e)}")
                break
        
        if future_prices_scaled:
            future_prices_scaled = np.array(future_prices_scaled).reshape(-1, 1)
            return close_scaler.inverse_transform(future_prices_scaled).flatten()
        
        return np.array([])
    
    def evaluate_ensemble(self, X_test: np.ndarray, y_test: np.ndarray, 
                          scalers: Dict, feature_cols: List[str]) -> Dict:
        """Evaluate the ensemble models."""
        ensemble_metrics = {}
        
        for name, model in self.ensemble_models:
            # Make predictions
            y_pred = model.predict(X_test, verbose=0)
            
            # Inverse transform predictions and actual values
            close_scaler = scalers['Close']
            y_test_inv = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_inv = close_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y_test_inv, y_pred_inv),
                'mae': mean_absolute_error(y_test_inv, y_pred_inv),
                'mape': mean_absolute_percentage_error(y_test_inv, y_pred_inv),
                'r2': r2_score(y_test_inv, y_pred_inv)
            }
            
            # Directional accuracy
            y_test_diff = np.diff(y_test_inv)
            y_pred_diff = np.diff(y_pred_inv)
            metrics['directional_accuracy'] = np.mean((y_test_diff > 0) == (y_pred_diff > 0)) * 100
            
            ensemble_metrics[name] = metrics
            
            logging.info(f"{name} evaluation metrics: {metrics}")
        
        return ensemble_metrics
    
    def save_ensemble(self, filepath: str):
        """Save the ensemble models and metadata."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save each model
        for i, (name, model) in enumerate(self.ensemble_models):
            model_path = f"{filepath}_model_{i}_{name}.h5"
            model.save(model_path)
            logging.info(f"Saved {name} model to {model_path}")
        
        # Save metadata
        metadata = {
            'scalers': self.scalers,
            'feature_cols': self.feature_cols,
            'input_shape': self.input_shape,
            'config': self.config.__dict__,
            'history': self.history,
            'model_names': [name for name, _ in self.ensemble_models],
            'timestamp': datetime.now().isoformat()
        }
        
        import pickle
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logging.info(f"Ensemble metadata saved to {filepath}_metadata.pkl")
    
    def load_ensemble(self, filepath: str):
        """Load an ensemble of models."""
        # Load metadata
        import pickle
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        self.scalers = metadata['scalers']
        self.feature_cols = metadata['feature_cols']
        self.input_shape = metadata['input_shape']
        self.history = metadata.get('history', None)
        
        # Load models
        self.ensemble_models = []
        for i, name in enumerate(metadata['model_names']):
            model_path = f"{filepath}_model_{i}_{name}.h5"
            model = load_model(model_path)
            self.ensemble_models.append((name, model))
        
        logging.info(f"Loaded ensemble with {len(self.ensemble_models)} models")