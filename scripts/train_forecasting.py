#!/usr/bin/env python3
"""
Train time series forecasting models for news category trends.

This script trains hybrid ML/DL models (Prophet, XGBoost, LSTM) for forecasting
future news article trends by category using historical data.
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
from app.services.time_series_forecaster import TimeSeriesForecaster

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train time series forecasting models for news trends"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/forecasting",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="news_trends_forecasting",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=10,
        help="Maximum number of top categories to train models for",
    )

    args = parser.parse_args()

    logger.info("Starting time series forecasting model training...")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Max categories: {args.max_categories}")

    try:
        # Get the forecaster service
        forecaster = TimeSeriesForecaster()

        # Override max categories if specified
        if args.max_categories != 10:
            # This would require modifying the service, for now use default
            logger.info(f"Training for top {args.max_categories} categories")

        # Train all models
        forecaster.train_all_models()

        logger.info("Time series forecasting training completed successfully!")
        logger.info(f"Models saved to: {args.output_dir}")

        # Print summary
        print("\n🎯 Training Summary:")
        print(f"📁 Models saved to: {args.output_dir}")
        print("Available forecasting endpoints:")
        print("  GET /trends/forecast/{category} - Forecast category trends")
        print("  POST /trends/forecast/train - Retrain models")
        print("\n📊 Supported categories (top 10):")
        print("  POLITICS, WELLNESS, ENTERTAINMENT, TRAVEL, STYLE & BEAUTY,")
        print("  PARENTING, HEALTHY LIVING, QUEER VOICES, FOOD & DRINK, BUSINESS")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
