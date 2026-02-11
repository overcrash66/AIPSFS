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
    args = parse_arguments()

    setup_logging(debug=args.debug)
    logging.info("Starting AI Stock Forecasting System")

    try:
        os.makedirs('models', exist_ok=True)

        api_config = ApiConfig.from_env()
        model_config = ModelConfig()
        system_config = SystemConfig(model=model_config, api=api_config)

        api_config.validate()

        end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else datetime.now()
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else end_date - timedelta(days=365 * 2)

        logging.info(f"Analysis period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        stock_list = load_stock_list(args.stocks)
        if not stock_list:
            logging.error("No stocks to analyze. Exiting.")
            sys.exit(1)

        logging.info(f"Loaded {len(stock_list)} stocks for analysis")
        for i, stock in enumerate(stock_list):
            logging.info(f"  {i + 1}. {stock['symbol']} - {stock['name']}")

        data_fetcher = DataFetcher(api_config, cache_enabled=not args.no_cache)
        feature_engineer = FeatureEngineer(
            news_api_history_days=system_config.news_api_history_days,
            min_data_rows_for_training=system_config.model.min_data_rows_for_training,
            api_config=api_config,
            data_fetcher=data_fetcher
        )
        report_generator = ReportGenerator(args.output)

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
        for result in results:
            logging.info(f"  {result['symbol']}: {result['return_pct']:.2f}% return")

        top_stocks = filter_top_stocks(
            results=results,
            min_return=args.min_return,
            top_n=args.top_n
        )

        if not top_stocks:
            logging.warning("No stocks meet the minimum return threshold")
            top_stocks = results[:args.top_n]

        logging.info(f"Selected {len(top_stocks)} top stocks for report")

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
    """Process stocks in parallel batches with safer multiprocessing and adaptive batch sizing."""
    import multiprocessing as mp
    import time
    import random
    from tqdm import tqdm
    import traceback

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    results = []
    batch_size = system_config.batch_processing_size
    max_processes = min(system_config.max_processes, mp.cpu_count())

    logging.info(f"Processing stocks in batches (initial size={batch_size}) using {max_processes} processes")
    batch_idx = 0
    i = 0

    while i < len(stock_list):
        batch_idx += 1
        batch = stock_list[i:i + batch_size]

        logging.info(f"Processing batch {batch_idx} with {len(batch)} stocks:")
        for stock in batch:
            logging.info(f"  - {stock['symbol']}: {stock['name']}")

        tasks = [
            (stock, start_date, end_date, data_fetcher,
             feature_engineer, system_config.model, use_advanced)
            for stock in batch
        ]

        try:
            with mp.get_context("spawn").Pool(processes=max_processes) as pool:
                batch_results = pool.map(analyze_stock_task, tasks)
                successful_results = [r for r in batch_results if r is not None]
                results.extend(successful_results)

            logging.info(f"Batch {batch_idx} completed: {len(successful_results)}/{len(batch)} stocks successful")
            i += batch_size  # Move to next batch

        except Exception as e:
            err_str = str(e).lower()
            if "cuda" in err_str or "resource exhausted" in err_str or "oom" in err_str:
                old_size = batch_size
                batch_size = max(1, batch_size // 2)
                logging.warning(f"⚠ GPU OOM detected! Reducing batch size from {old_size} to {batch_size} and retrying...")
                if batch_size == 1 and old_size == 1:
                    logging.error("Cannot reduce batch size further — aborting remaining jobs.")
                    break
            else:
                logging.error(f"Batch {batch_idx} failed: {e}")
                logging.debug(traceback.format_exc())
                i += batch_size  # Skip to next batch

        if i < len(stock_list):
            delay = random.uniform(
                system_config.batch_processing_delay_min,
                system_config.batch_processing_delay_max
            )
            logging.info(f"Waiting {delay:.1f}s before next batch...")
            time.sleep(delay)

    logging.info(f"All batches completed. Total successful: {len(results)}/{len(stock_list)}")
    return results


def analyze_stock_task(args: tuple) -> Optional[Dict]:
    """Worker function for parallel stock analysis with shape checks and memory safety."""
    import tensorflow as tf
    import gc
    from advanced_model import AdvancedStockPredictor
    from config import AdvancedModelConfig
    from model import StockPredictor

    stock, start_date, end_date, data_fetcher, feature_engineer, model_config, use_advanced = args
    symbol = stock.get('symbol', '').strip()
    name = stock.get('name', '').strip()

    try:
        tf.keras.backend.clear_session()
    except Exception as e:
        logging.warning(f"TF session clear failed for {symbol}: {e}")

    logging.info(f"--- Starting analysis for {symbol} ({name}) ---")

    try:
        df = data_fetcher.fetch_stock_data(symbol, start_date, end_date)
        if df is None or df.empty:
            logging.warning(f"{symbol} skipped: no historical data.")
            return None

        df = feature_engineer.process_stock_data(symbol, df, start_date, end_date)
        if df is None or df.empty:
            logging.warning(f"{symbol} skipped: feature engineering failed.")
            return None

        X, y, scalers, feature_cols = feature_engineer.prepare_multi_feature_data(
            df, model_config.lookback_days
        )
        if X is None or len(X) == 0:
            logging.warning(f"{symbol} skipped: insufficient data for modeling.")
            return None

        split = int(len(X) * model_config.train_split_ratio)
        if split < model_config.lookback_days or (len(X) - split) < 10:
            logging.warning(f"{symbol} skipped: not enough data for a valid train/test split.")
            return None

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        forecast_prices, forecast_std, avg_metrics = None, None, {}

        if use_advanced:
            logging.info(f"{symbol}: Using advanced ensemble model.")
            advanced_config = AdvancedModelConfig()
            predictor = AdvancedStockPredictor(advanced_config)
            predictor.train_ensemble(X_train, y_train, X_test, y_test, scalers, feature_cols)
            metrics = predictor.evaluate(X_test, y_test)

            avg_metrics = metrics.get('ensemble_average', {
                k: float(np.mean([m.get(k, 0) for m in metrics.values()]))
                for k in ['mse', 'mae', 'r2', 'directional_accuracy']
            })

            last_sequence = X[-1:].astype(np.float32)
            if last_sequence.shape != (1, predictor.input_shape[0], predictor.input_shape[1]):
                logging.warning(f"{symbol}: Reshaping prediction input {last_sequence.shape} "
                                f"-> {(1, predictor.input_shape[0], predictor.input_shape[1])}")
                last_sequence = last_sequence[:, -predictor.input_shape[0]:, :predictor.input_shape[1]]

            forecast_prices, forecast_std = predictor.predict(
                last_sequence=last_sequence,
                steps=int(model_config.forecast_steps)
            )

            predictor._save_ensemble_metadata()

        else:
            logging.info(f"{symbol}: Using standard LSTM model.")
            predictor = StockPredictor(model_config)
            predictor.build_model((X_train.shape[1], X_train.shape[2]))
            predictor.train(X_train, y_train, X_test, y_test)
            avg_metrics = predictor.evaluate_model(X_test, y_test, scalers, feature_cols)

            last_sequence = X[-1:].astype(np.float32)
            forecast_prices = predictor.predict(last_sequence, scalers, feature_cols)
            forecast_std = np.zeros_like(forecast_prices)

        if forecast_prices is None or np.size(forecast_prices) == 0:
            logging.warning(f"{symbol}: prediction failed or empty.")
            return None

        if isinstance(forecast_prices, tf.Tensor):
            forecast_prices = forecast_prices.numpy()
        if isinstance(forecast_std, tf.Tensor):
            forecast_std = forecast_std.numpy()

        predicted_price = float(np.ravel(forecast_prices)[-1])
        std_value = float(np.ravel(forecast_std)[-1]) if forecast_std.size > 0 else 0.0
        current_price = float(df['Close'].iloc[-1])
        return_pct = ((predicted_price - current_price) / current_price) * 100

        forecast_dates = pd.date_range(
            df.index[-1] + timedelta(days=1),
            periods=model_config.forecast_steps
        )

        logging.info(
            f"{symbol}: Current=${current_price:.2f} "
            f"Predicted=${predicted_price:.2f} "
            f"Return={return_pct:.2f}%"
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
        logging.error(f"{symbol}: Critical error - {e}", exc_info=True)
        return None

    finally:
        tf.keras.backend.clear_session()
        gc.collect()


def filter_top_stocks(results: List[Dict], min_return: float, top_n: int) -> List[Dict]:
    """Filter stocks by minimum return and select top N."""
    filtered = [r for r in results if r['return_pct'] >= min_return]
    filtered.sort(key=lambda x: x['return_pct'], reverse=True)
    return filtered[:top_n]


if __name__ == "__main__":
    main()
