# visualization.py
import logging

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def plot_prediction_vs_actual(historical_dates, historical_prices, 
                             forecast_dates, forecast_prices, 
                             symbol, save_path=None):
    """Plot historical prices and forecasted prices"""
    plt.figure(figsize=(14, 7))
    
    # Plot historical prices
    plt.plot(historical_dates, historical_prices, label='Historical Prices', color='blue')
    
    # Plot forecasted prices
    plt.plot(forecast_dates, forecast_prices, label='Forecasted Prices', color='red', linestyle='--')
    
    # Add vertical line at forecast start
    plt.axvline(x=historical_dates[-1], color='gray', linestyle=':', alpha=0.7)
    
    # Add labels and title
    plt.title(f'Stock Price Forecast for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gcf().autofmt_xdate()
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Prediction plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_model_training_history(history, save_path=None):
    """Plot model training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot training & validation loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot training & validation MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model Mean Absolute Error')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()