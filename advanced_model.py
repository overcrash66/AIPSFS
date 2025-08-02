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
import json
import pickle
import joblib
class AdvancedStockPredictor:
    """Advanced stock prediction model with ensemble and attention mechanisms."""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.histories = {}
        self.feature_cols = []
        self.input_shape = None
        self.model_dir = "models"
        self.model_type = 'Unknown'
        self.target_col = getattr(self, 'target_col', 'Close')
        self.best_model_index = None
        
        # Create models directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Configure GPU if available
        self._configure_gpu()
    
    def _set_model_type(self):
        """Determine model type based on the models in the ensemble."""
        if not self.models:
            self.model_type = 'Unknown'
            return
            
        # Ensure we're working with a list
        if isinstance(self.models, list):
            first_model = self.models[0]
        else:
            # If it's not a list, try to get the first model
            try:
                first_model = next(iter(self.models.values()))
            except (AttributeError, StopIteration):
                self.model_type = 'Unknown'
                return
            
        # Check the type of the first model
        if 'LSTM' in str(type(first_model)):
            self.model_type = 'LSTM'
        elif 'GRU' in str(type(first_model)):
            self.model_type = 'GRU'
        elif 'CNN' in str(type(first_model)):
            self.model_type = 'CNN'
        else:
            self.model_type = 'NeuralNetwork'     
    
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
    
    def build_lstm_model(self, input_shape, name):
        """Build an enhanced LSTM model with attention."""
        inputs = Input(shape=input_shape, name=f"{name}_input")
        
        # LSTM layers with recurrent dropout
        x = LSTM(
            self.config.lstm_units_layer1, 
            return_sequences=True,
            recurrent_dropout=0.2,
            kernel_regularizer=l1_l2(0.01),
            name=f"{name}_lstm1"
        )(inputs)
        x = LayerNormalization(name=f"{name}_norm1")(x)
        x = Dropout(0.3, name=f"{name}_dropout1")(x)
        
        x = LSTM(
            self.config.lstm_units_layer2, 
            return_sequences=True,
            recurrent_dropout=0.2,
            kernel_regularizer=l1_l2(0.01),
            name=f"{name}_lstm2"
        )(x)
        x = LayerNormalization(name=f"{name}_norm2")(x)
        x = Dropout(0.3, name=f"{name}_dropout2")(x)
        
        # Attention mechanism
        attention = Attention(name=f"{name}_attention")([x, x])
        attention = LayerNormalization(name=f"{name}_attention_norm")(attention)
        
        # Global average pooling
        x = GlobalAveragePooling1D(name=f"{name}_gap")(attention)
        
        # Dense layers
        x = Dense(64, activation='relu', kernel_regularizer=l1_l2(0.01), name=f"{name}_dense1")(x)
        x = BatchNormalization(name=f"{name}_bn1")(x)
        x = Dropout(0.2, name=f"{name}_dropout3")(x)
        
        # Output layer
        output = Dense(1, name=f"{name}_output")(x)
        
        model = Model(inputs=inputs, outputs=output, name=name)

        # Learning rate scheduling
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )
        
        #optimizer = Adam(learning_rate=lr_schedule)
        optimizer = Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def build_gru_model(self, input_shape: Tuple[int, int], name: str = "gru") -> Model:
        """Build a GRU-based model with attention."""
        inputs = Input(shape=input_shape, name=f"{name}_input")
        
        # GRU layers
        x = GRU(
            self.config.gru_units_layer1,
            return_sequences=True,
            recurrent_dropout=0.2,
            kernel_regularizer=l1_l2(0.01),
            name=f"{name}_gru1"
        )(inputs)
        x = LayerNormalization(name=f"{name}_norm1")(x)
        x = Dropout(0.3, name=f"{name}_dropout1")(x)
        
        x = GRU(
            self.config.gru_units_layer2,
            return_sequences=True,
            recurrent_dropout=0.2,
            kernel_regularizer=l1_l2(0.01),
            name=f"{name}_gru2"
        )(x)
        x = LayerNormalization(name=f"{name}_norm2")(x)
        x = Dropout(0.3, name=f"{name}_dropout2")(x)
        
        # Attention mechanism
        attention = Attention(name=f"{name}_attention")([x, x])
        attention = LayerNormalization(name=f"{name}_attention_norm")(attention)
        
        # Global average pooling
        x = GlobalAveragePooling1D(name=f"{name}_gap")(attention)
        
        # Dense layers
        x = Dense(64, activation='relu', kernel_regularizer=l1_l2(0.01), name=f"{name}_dense1")(x)
        x = BatchNormalization(name=f"{name}_bn1")(x)
        x = Dropout(0.2, name=f"{name}_dropout3")(x)
        
        # Output layer
        output = Dense(1, name=f"{name}_output")(x)
        
        model = Model(inputs=inputs, outputs=output, name=name)

        optimizer = Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def build_cnn_lstm_model(self, input_shape: Tuple[int, int], name: str = "cnn_lstm") -> Model:
        """Build a hybrid CNN-LSTM model with attention."""
        inputs = Input(shape=input_shape, name=f"{name}_input")
        
        # CNN layers for feature extraction
        x = Conv1D(
            filters=64, 
            kernel_size=3, 
            activation='relu', 
            padding='same',
            name=f"{name}_conv1"
        )(inputs)
        x = MaxPooling1D(pool_size=2, name=f"{name}_pool1")(x)
        x = BatchNormalization(name=f"{name}_bn1")(x)
        x = Dropout(0.2, name=f"{name}_dropout1")(x)
        
        # Bidirectional LSTM layers
        x = Bidirectional(
            LSTM(self.config.lstm_units_layer1, return_sequences=True),
            name=f"{name}_bilstm1"
        )(x)
        x = LayerNormalization(name=f"{name}_norm1")(x)
        x = Dropout(0.3, name=f"{name}_dropout2")(x)
        
        x = Bidirectional(
            LSTM(self.config.lstm_units_layer2, return_sequences=True),
            name=f"{name}_bilstm2"
        )(x)
        x = LayerNormalization(name=f"{name}_norm2")(x)
        x = Dropout(0.3, name=f"{name}_dropout3")(x)
        
        # Attention mechanism
        attention = Attention(name=f"{name}_attention")([x, x])
        attention = LayerNormalization(name=f"{name}_attention_norm")(attention)
        
        # Global average pooling
        x = GlobalAveragePooling1D(name=f"{name}_gap")(attention)
        
        # Dense layers
        x = Dense(64, activation='relu', kernel_regularizer=l1_l2(0.01), name=f"{name}_dense1")(x)
        x = BatchNormalization(name=f"{name}_bn2")(x)
        x = Dropout(0.2, name=f"{name}_dropout4")(x)
        
        # Output layer
        output = Dense(1, name=f"{name}_output")(x)
        
        model = Model(inputs=inputs, outputs=output, name=name)
        
        # Learning rate scheduling
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )
        
        #optimizer = Adam(learning_rate=lr_schedule)
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def build_ensemble(self, input_shape: Tuple[int, int]):
        """Build an ensemble of different models."""
        logging.info("Building ensemble models...")
        
        # Build different model architectures
        self.models['lstm'] = self.build_lstm_model(input_shape, "lstm")
        self.models['gru'] = self.build_gru_model(input_shape, "gru")
        self.models['cnn_lstm'] = self.build_cnn_lstm_model(input_shape, "cnn_lstm")
        self._set_model_type()
        logging.info(f"Built {len(self.models)} models: {list(self.models.keys())}")
        
        return self.models
    
    def train_ensemble(self, X_train, y_train, X_test, y_test, scalers, feature_cols):
        """Train an ensemble of models."""
        self.input_shape = (X_train.shape[1], X_train.shape[2])
        self.feature_cols = feature_cols
        self.scalers = scalers
        
        # Build ensemble models
        self.build_ensemble(self.input_shape)
        
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
            )
        ]
        
        # Train each model
        model_histories = {}
        for name, model in self.models.items():
            logging.info(f"Training {name} model...")
            
            # Model-specific checkpoint
            checkpoint_path = os.path.join(self.model_dir, f"{name}_best_model.h5")
            checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            
            model_callbacks = callbacks + [checkpoint]
            
            # Suppress TensorFlow logging during training
            tf.get_logger().setLevel('ERROR')
            history = model.fit(
                X_train, y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_data=(X_test, y_test),
                callbacks=model_callbacks,
                verbose=1
            )
            tf.get_logger().setLevel('INFO')
            
            model_histories[name] = history.history
            
            # Load best model
            model.load_weights(checkpoint_path)

            # Save the complete model
            model.save(os.path.join(self.model_dir, f"{name}_final_model.h5"))
            
            logging.info(f"Completed training {name} model")
        
        self._set_model_type()
        
        
        self.histories = model_histories
        
         # After training all models, determine the best one
        best_score = float('inf')
        best_index = 0
        
        for model_name, model in self.models.items():
            # Evaluate the model
            evaluation = model.evaluate(X_test, y_test, verbose=0)
            
            # Handle different return types
            if isinstance(evaluation, list):
                loss = evaluation[0]  # First element is typically the loss
            else:
                loss = evaluation  # If it's a single value
            
            # Update best model if this one has lower loss
            if loss < best_score:
                best_score = loss
                best_model_key = model_name
        
        # Set the best model index
        self.best_model_index = best_index

        # Save ensemble metadata
        self._save_ensemble_metadata()
        
        return model_histories
    
    def _save_ensemble_metadata(self):
        """Save ensemble metadata including model information and scalers."""
        # Create a directory for scalers if it doesn't exist
        scalers_dir = os.path.join(self.model_dir, 'scalers')
        os.makedirs(scalers_dir, exist_ok=True)
        
        # Save each scaler separately
        scaler_paths = {}
        for name, scaler in self.scalers.items():
            scaler_path = os.path.join(scalers_dir, f'{name}.joblib')
            joblib.dump(scaler, scaler_path)
            scaler_paths[name] = scaler_path
        
        # Prepare metadata without scaler objects
        metadata = {
            'model_type': self.model_type,
            'feature_cols': self.feature_cols,
            'target_col': self.target_col, 
            'scalers': scaler_paths,  # Store paths instead of objects
            'models': self.models,
            'best_model_index': self.best_model_index,
            'history': self.history
        }
        
        # Save metadata to JSON
        metadata_path = os.path.join(self.model_dir, 'ensemble_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_ensemble_metadata(self):
        """Load ensemble metadata including scalers from files."""
        metadata_path = os.path.join(self.model_dir, 'ensemble_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load scalers from files
        scalers = {}
        for name, path in metadata['scalers'].items():
            scalers[name] = joblib.load(path)
        
        # Update metadata with loaded scalers
        metadata['scalers'] = scalers
        
        # Set instance attributes
        self.model_type = metadata['model_type']
        self.feature_cols = metadata['feature_cols']
        self.target_col = metadata['target_col']
        self.scalers = metadata['scalers']
        self.models = metadata['models']
        self.best_model_index = metadata['best_model_index']
        self.history = metadata['history']
        
        return metadata
    
    def predict_ensemble(self, last_sequence: np.ndarray, 
                        scalers: Dict, feature_cols: List[str], 
                        steps: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the ensemble of models."""
        if steps is None:
            steps = self.config.forecast_steps
            
        if last_sequence is None or last_sequence.size == 0:
            logging.warning("No valid sequence provided for forecasting.")
            return np.array([]), np.array([])
        
        if 'Close' not in scalers or 'Close' not in feature_cols:
            logging.error("Scaler or feature column for 'Close' price not found.")
            return np.array([]), np.array([])
        
        # Get predictions from each model
        ensemble_predictions = []
        close_scaler = scalers['Close']
        close_idx = feature_cols.index('Close')
        
        for name, model in self.models.items():
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
            return np.array([]), np.array([])
        
        # Convert to numpy array
        ensemble_predictions = np.array(ensemble_predictions)
        
        # Calculate mean predictions (ensemble prediction)
        mean_predictions = np.mean(ensemble_predictions, axis=0)
        
        # Calculate prediction variance (uncertainty)
        prediction_variance = np.var(ensemble_predictions, axis=0)
        prediction_std = np.sqrt(prediction_variance)
        
        logging.info(f"Ensemble prediction: {mean_predictions[-1]:.2f}")
        logging.info(f"Prediction std dev: {prediction_std[-1]:.4f}")
        
        return mean_predictions, prediction_std
    
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
        
        # Handle None scalers by creating dummy scalers
        if scalers.get('Close') is None:
            logging.warning("Close scaler is None, creating dummy scaler")
            from sklearn.preprocessing import MinMaxScaler
            close_scaler = MinMaxScaler(feature_range=(0, 1))
            close_scaler.fit(y_test.reshape(-1, 1))
            scalers['Close'] = close_scaler
        
        for name, model in self.models.items():
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
        
        # Calculate ensemble metrics
        if len(self.models) > 1:
            # Get ensemble predictions
            ensemble_pred, _ = self.predict_ensemble(X_test, scalers, feature_cols, len(y_test))
            
            if ensemble_pred.size > 0:
                ensemble_metrics['ensemble'] = {
                    'mse': mean_squared_error(y_test_inv, ensemble_pred),
                    'mae': mean_absolute_error(y_test_inv, ensemble_pred),
                    'mape': mean_absolute_percentage_error(y_test_inv, ensemble_pred),
                    'r2': r2_score(y_test_inv, ensemble_pred),
                    'directional_accuracy': np.mean((y_test_diff > 0) == (np.diff(ensemble_pred) > 0)) * 100
                }
                
                logging.info(f"Ensemble evaluation metrics: {ensemble_metrics['ensemble']}")
        
        return ensemble_metrics