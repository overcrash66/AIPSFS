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

from advanced_model import AdvancedStockPredictor
from config import AdvancedModelConfig

# Load environment variables
load_dotenv()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AI-Powered Stock Forecasting System')
    
    parser.add_argument('--stocks', type=str, default='stocks.csv',
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
    parser.add_argument('--use-advanced', action='store_true',
                       help='Use advanced ensemble models')
    
    return parser.parse_args()

def main():
    """Main application entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(debug=args.debug)
    logging.info("Starting AI Stock Forecasting System")
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
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
        logging.info(f"Loading stock list from: {args.stocks}")
        stock_list = load_stock_list(args.stocks)
        
        if not stock_list:
            logging.error("No stocks to analyze. Exiting.")
            sys.exit(1)
        
        logging.info(f"Loaded {len(stock_list)} stocks for analysis")
        
        # Log all stocks for verification
        logging.info("Stocks to analyze:")
        for i, stock in enumerate(stock_list):
            logging.info(f"  {i+1}. {stock['symbol']} - {stock['name']}")
        
        # Initialize components
        data_fetcher = DataFetcher(api_config, cache_enabled=not args.no_cache)
        
        # Initialize feature engineer with API config
        feature_engineer = FeatureEngineer(
            news_api_history_days=system_config.news_api_history_days,
            min_data_rows_for_training=system_config.model.min_data_rows_for_training,
            api_config=api_config
        )
        
        report_generator = ReportGenerator(args.output)
        
        # Process stocks
        results = process_stocks(
            stock_list=stock_list,
            start_date=start_date,
            end_date=end_date,
            data_fetcher=data_fetcher,
            feature_engineer=feature_engineer,
            system_config=system_config,
            use_advanced=args.use_advanced
        )
        
        if not results:
            logging.error("No stocks were successfully analyzed. Exiting.")
            sys.exit(1)
        
        logging.info(f"Successfully analyzed {len(results)} stocks")
        
        # Log results for each stock
        for result in results:
            logging.info(f"  {result['symbol']}: {result['return_pct']:.2f}% return")
        
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
                  system_config: SystemConfig, use_advanced: bool) -> List[Dict]:
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
    logging.info(f"Total stocks to process: {len(stock_list)}")
    
    total_batches = (len(stock_list) + batch_size - 1) // batch_size
    logging.info(f"Total batches to process: {total_batches}")
    
    for batch_idx, i in enumerate(tqdm(range(0, len(stock_list), batch_size), desc="Processing stocks", total=total_batches)):
        batch = stock_list[i:i + batch_size]
        
        logging.info(f"Processing batch {batch_idx + 1}/{total_batches} with {len(batch)} stocks:")
        for stock in batch:
            logging.info(f"  - {stock['symbol']}: {stock['name']}")
        
        # Prepare arguments for each task
        tasks = [
            (stock, start_date, end_date, data_fetcher, 
             feature_engineer, system_config.model, use_advanced)
            for stock in batch
        ]
        
        # Process batch in parallel
        with Pool(processes=max_processes) as pool:
            batch_results = pool.map(analyze_stock_task, tasks)
            successful_results = [r for r in batch_results if r is not None]
            results.extend(successful_results)
        
        logging.info(f"Batch {batch_idx + 1} completed: {len(successful_results)}/{len(batch)} stocks successful")
        
        # Random delay between batches to avoid rate limits
        if i + batch_size < len(stock_list):
            delay = random.uniform(
                system_config.batch_processing_delay_min, 
                system_config.batch_processing_delay_max
            )
            logging.info(f"Waiting {delay:.1f}s before next batch...")
            time.sleep(delay)
    
    logging.info(f"All batches completed. Total successful: {len(results)}/{len(stock_list)}")
    return results

def analyze_stock_task(args: tuple) -> Optional[Dict]:
    """Worker function for parallel stock analysis with advanced models."""
    import tensorflow as tf
    from advanced_model import AdvancedStockPredictor
    from config import AdvancedModelConfig
    
    # Unpack arguments
    stock, start_date, end_date, data_fetcher, feature_engineer, model_config, use_advanced = args
    
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
        if use_advanced:
            logging.info(f"Using advanced ensemble model for {symbol}")
            advanced_config = AdvancedModelConfig()
            predictor = AdvancedStockPredictor(advanced_config)
            
            # Train ensemble
            history = predictor.train_ensemble(X_train, y_train, X_test, y_test, scalers, feature_cols)
            
            # Evaluate ensemble
            metrics = predictor.evaluate(X_test, y_test)
            
            # Get ensemble metrics
            if 'ensemble_average' in metrics:
                avg_metrics = metrics['ensemble_average']
            else:
                # Calculate average of individual model metrics
                avg_metrics = {}
                for metric in ['mse', 'mae', 'r2', 'directional_accuracy']:
                    avg_metrics[metric] = np.mean([m.get(metric, 0) for m in metrics.values()])
            
            # Forecast future prices with uncertainty
            last_sequence = X[-1:]
            # Ensure proper shape and type
            if last_sequence.ndim == 2:
                last_sequence = np.expand_dims(last_sequence, axis=0)
            last_sequence = last_sequence.astype(np.float32)
            
            forecast_prices, forecast_std = predictor.predict(
                last_sequence=last_sequence,
                steps=model_config.forecast_steps
            )
            
            # Save ensemble
            predictor._save_ensemble_metadata()
            
        else:
            logging.info(f"Using standard model for {symbol}")
            from model import StockPredictor
            predictor = StockPredictor(model_config)
            predictor.build_model((X_train.shape[1], X_train.shape[2]))
            
            # Train model
            history = predictor.train(X_train, y_train, X_test, y_test)
            
            # Evaluate model
            metrics = predictor.evaluate_model(X_test, y_test, scalers, feature_cols)
            avg_metrics = metrics
            
            # Forecast future prices
            last_sequence = X[-1:]
            # Ensure proper shape and type
            if last_sequence.ndim == 2:
                last_sequence = np.expand_dims(last_sequence, axis=0)
            last_sequence = last_sequence.astype(np.float32)
            
            forecast_prices = predictor.predict(last_sequence, scalers, feature_cols)
            forecast_std = np.zeros_like(forecast_prices)  # No uncertainty for single model
        
        # Convert to numpy if needed
        if isinstance(forecast_prices, tf.Tensor):
            forecast_prices = forecast_prices.numpy()
        if isinstance(forecast_std, tf.Tensor):
            forecast_std = forecast_std.numpy()
            
        # Check for empty predictions
        if forecast_prices.size == 0:
            logging.warning(f"Forecasting failed for {symbol}")
            return None
            
        # Extract scalar values
        predicted_price = float(np.ravel(forecast_prices)[-1])
        std_value = float(np.ravel(forecast_std)[-1]) if forecast_std.size > 0 else 0.0
        
        current_price = df['Close'].iloc[-1]
        return_pct = ((predicted_price - current_price) / current_price) * 100
        
        logging.info(f"Completed {symbol}: Current=${current_price:.2f}, "
                     f"Predicted=${predicted_price:.2f}, Return={return_pct:.2f}%")
        
        # Prepare forecast dates
        last_historical_date = df.index[-1]
        forecast_dates = pd.date_range(
            last_historical_date + timedelta(days=1), 
            periods=model_config.forecast_steps
        )
        
        return {
            'symbol': symbol,
            'name': name,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'return_pct': return_pct,
            'model_metrics': avg_metrics,
            'forecast_std': std_value,
            'historical_dates': df.index.tolist(),
            'historical_prices': df['Close'].values.tolist(),
            'forecast_dates': forecast_dates.tolist(),
            'forecast_prices': forecast_prices.flatten().tolist(),
            'use_advanced': use_advanced
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