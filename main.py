# main.py
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
from dotenv import load_dotenv

from config import ModelConfig, ApiConfig, SystemConfig
from data_fetcher import DataFetcher
from feature_engineering import FeatureEngineer
from model import StockPredictor
from reporter import ReportGenerator
from utils import setup_logging, load_stock_list
import numpy as np

# Load environment variables
load_dotenv()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AI-Powered Stock Forecasting System')
    
    parser.add_argument('--stocks', type=str, default='us_canada_stocks.xlsx',
                       help='Path to CSV file containing stock list')
    parser.add_argument('--start-date', type=str, 
                       help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, 
                       help='End date for analysis (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='reports',
                       help='Output directory for reports')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top stocks to include in report')
    parser.add_argument('--min-return', type=float, default=50.0,
                       help='Minimum predicted return percentage to include in report')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching of API responses')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    return parser.parse_args()

def main():
    """Main application entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(debug=args.debug)
    logging.info("Starting AI Stock Forecasting System")
    
    try:
        # Load configuration from environment variables
        api_config = ApiConfig.from_env()
        model_config = ModelConfig()
        system_config = SystemConfig(model=model_config, api=api_config)
        
        # Validate API configuration
        api_config.validate()
        
        # Determine date range
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else datetime.now()
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else end_date - timedelta(days=365*2)
        
        logging.info(f"Analysis period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Load stock list
        stock_list = load_stock_list(args.stocks)
        if not stock_list:
            logging.error("No stocks to analyze. Exiting.")
            sys.exit(1)
        
        logging.info(f"Loaded {len(stock_list)} stocks for analysis")
        
        # Initialize components
        data_fetcher = DataFetcher(api_config, cache_enabled=not args.no_cache)
        
        # Initialize feature engineer with API config
        feature_engineer = FeatureEngineer(
            news_api_history_days=system_config.news_api_history_days,
            min_data_rows_for_training=system_config.model.min_data_rows_for_training,
            api_config=api_config  # Pass the API config here
        )
        
        report_generator = ReportGenerator(args.output)
        
        # Process stocks
        results = process_stocks(
            stock_list=stock_list,
            start_date=start_date,
            end_date=end_date,
            data_fetcher=data_fetcher,
            feature_engineer=feature_engineer,
            system_config=system_config
        )
        
        if not results:
            logging.error("No stocks were successfully analyzed. Exiting.")
            sys.exit(1)
        
        logging.info(f"Successfully analyzed {len(results)} stocks")
        
        # Filter top performing stocks
        top_stocks = filter_top_stocks(
            results=results,
            min_return=args.min_return,
            top_n=args.top_n
        )
        
        if not top_stocks:
            logging.warning("No stocks meet the minimum return threshold")
            top_stocks = results[:args.top_n]  # Fall back to top N by return
        
        logging.info(f"Selected {len(top_stocks)} top stocks for report")
        
        # Generate report
        report_path = report_generator.generate_report(
            top_stocks=top_stocks,
            analysis_period=(start_date, end_date)
        )
        
        logging.info(f"Report generated successfully: {report_path}")
        
    except Exception as e:
        logging.error(f"Application error: {str(e)}", exc_info=True)
        sys.exit(1)

def process_stocks(stock_list: List[Dict], start_date: datetime, end_date: datetime,
                  data_fetcher: DataFetcher, feature_engineer: FeatureEngineer,
                  system_config: SystemConfig) -> List[Dict]:
    """Process stocks in parallel batches."""
    from multiprocessing import Pool, cpu_count
    import time
    import random
    from tqdm import tqdm
    
    results = []
    
    # Process in batches to avoid API rate limits
    batch_size = system_config.batch_processing_size
    max_processes = min(system_config.max_processes, cpu_count())
    
    logging.info(f"Processing stocks in batches of {batch_size} using {max_processes} processes")
    
    for i in tqdm(range(0, len(stock_list), batch_size), desc="Processing stocks"):
        batch = stock_list[i:i + batch_size]
        
        # Prepare arguments for each task
        tasks = [
            (stock, start_date, end_date, data_fetcher, 
             feature_engineer,  # Pass the feature_engineer instance
             system_config.model)
            for stock in batch
        ]
        
        # Process batch in parallel
        with Pool(processes=max_processes) as pool:
            batch_results = pool.map(analyze_stock_task, tasks)
            results.extend([r for r in batch_results if r is not None])
        
        # Random delay between batches to avoid rate limits
        if i + batch_size < len(stock_list):
            delay = random.uniform(
                system_config.batch_processing_delay_min, 
                system_config.batch_processing_delay_max
            )
            logging.info(f"Waiting {delay:.1f}s before next batch...")
            time.sleep(delay)
    
    return results

def analyze_stock_task(args: tuple) -> Optional[Dict]:
    """Worker function for parallel stock analysis."""
    import tensorflow as tf
    from model import StockPredictor
    
    # Unpack arguments
    stock, start_date, end_date, data_fetcher, feature_engineer, model_config = args
    
    # Clear TensorFlow session for each process
    tf.keras.backend.clear_session()
    
    try:
        symbol = stock['symbol']
        name = stock['name']
        
        logging.info(f"Starting analysis for {symbol} - {name}")
        
        # 1. Download historical data
        df = data_fetcher.fetch_stock_data(symbol, start_date, end_date)
        if df is None:
            logging.warning(f"Skipping {symbol} due to data download failure")
            return None
        
        # 2. Process and integrate all data sources
        df = feature_engineer.process_stock_data(symbol, df, start_date, end_date)
        if df is None:
            logging.warning(f"Skipping {symbol} due to data processing failure")
            return None
        
        # 3. Prepare data for modeling
        X, y, scalers, feature_cols = feature_engineer.prepare_multi_feature_data(
            df, model_config.lookback_days
        )
        if X is None or len(X) == 0:
            logging.warning(f"Skipping {symbol}: Insufficient data for model preparation")
            return None
        
        # Split data into training and testing sets
        split = int(len(X) * model_config.train_split_ratio)
        if split < model_config.lookback_days or len(X) - split < 10:
            logging.warning(f"Skipping {symbol}: Not enough data for robust train/test split")
            return None
        
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # 4. Build and train model
        predictor = StockPredictor(model_config)
        predictor.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Suppress TensorFlow logging during training
        tf.get_logger().setLevel('ERROR')
        history = predictor.train(X_train, y_train, X_test, y_test)
        tf.get_logger().setLevel('INFO')
        
        if not history['val_loss'] or np.isnan(history['val_loss'][-1]):
            logging.warning(f"Model training for {symbol} resulted in NaN validation loss")
            return None
        
        # 5. Evaluate model
        metrics = predictor.evaluate_model(X_test, y_test, scalers, feature_cols)
        
        # 6. Forecast future prices
        last_sequence = X[-1:]
        forecast_prices = predictor.predict(last_sequence, scalers, feature_cols)
        
        if forecast_prices.size == 0:
            logging.warning(f"Forecasting failed for {symbol}")
            return None
        
        current_price = df['Close'].iloc[-1]
        predicted_price = forecast_prices[-1]
        return_pct = ((predicted_price - current_price) / current_price) * 100
        
        logging.info(f"Completed {symbol}: Current=${current_price:.2f}, Predicted=${predicted_price:.2f}, Return={return_pct:.2f}%")
        
        # Prepare forecast dates
        last_historical_date = df.index[-1]
        forecast_dates = pd.date_range(last_historical_date + timedelta(days=1), periods=model_config.forecast_steps)
        
        return {
            'symbol': symbol,
            'name': name,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'return_pct': return_pct,
            'model_metrics': metrics,
            'historical_dates': df.index.tolist(),
            'historical_prices': df['Close'].values.tolist(),
            'forecast_dates': forecast_dates.tolist(),
            'forecast_prices': forecast_prices.tolist()
        }
        
    except Exception as e:
        logging.error(f"Critical error during analysis for {symbol}: {str(e)}", exc_info=True)
        return None

def filter_top_stocks(results: List[Dict], min_return: float, top_n: int) -> List[Dict]:
    """Filter stocks by minimum return and select top N."""
    # Filter by minimum return
    filtered = [r for r in results if r['return_pct'] >= min_return]
    
    # Sort by return percentage (descending)
    filtered.sort(key=lambda x: x['return_pct'], reverse=True)
    
    # Return top N
    return filtered[:top_n]

if __name__ == "__main__":
    main()