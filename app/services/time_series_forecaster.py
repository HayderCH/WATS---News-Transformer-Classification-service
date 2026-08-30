import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import joblib
import mlflow
import mlflow.sklearn
import mlflow.pytorch

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import xgboost as xgb
from prophet import Prophet

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting"""

    def __init__(self, data: np.ndarray, seq_length: int):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class LSTMForecaster(nn.Module):
    """LSTM-based forecaster for time series"""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class TimeSeriesForecaster:
    """Hybrid ML/DL time series forecasting service"""

    def __init__(self):
        self.prophet_models = {}
        self.xgb_models = {}
        self.lstm_models = {}
        self.scalers = {}
        self.category_mapping = {}
        self.seq_length = 30  # 30 days lookback
        self.data_path = Path("data/processed/huffpost_with_dates.csv")

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the time series data"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        df = pd.read_csv(self.data_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # Create category mapping
        self.category_mapping = {
            cat: i for i, cat in enumerate(df["category"].unique())
        }

        return df

    def prepare_time_series(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """Prepare time series data for a specific category"""
        # Filter by category
        cat_data = df[df["category"] == category].copy()

        # Aggregate by date
        daily_counts = cat_data.groupby("date").size().reset_index(name="count")

        # Fill missing dates with 0
        date_range = pd.date_range(
            start=daily_counts["date"].min(), end=daily_counts["date"].max(), freq="D"
        )
        daily_counts = (
            daily_counts.set_index("date")
            .reindex(date_range, fill_value=0)
            .reset_index()
        )
        daily_counts.columns = ["date", "count"]

        return daily_counts

    def train_prophet(self, ts_data: pd.DataFrame, category: str) -> Prophet:
        """Train Prophet model for a category"""
        # Prepare data for Prophet
        prophet_data = ts_data[["date", "count"]].copy()
        prophet_data.columns = ["ds", "y"]

        # Add seasonality
        # Ensure non-negative
        prophet_data["y"] = prophet_data["y"].clip(lower=0)

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
        )

        with mlflow.start_run(run_name=f"prophet_{category}"):
            model.fit(prophet_data)

            # Log parameters
            mlflow.log_param("model_type", "prophet")
            mlflow.log_param("category", category)
            mlflow.log_param("changepoint_prior_scale", 0.05)

            # Log model
            mlflow.prophet.log_model(model, f"prophet_{category}")

        return model

    def train_xgboost(self, ts_data: pd.DataFrame, category: str) -> xgb.XGBRegressor:
        """Train XGBoost model for a category"""
        # Create features
        df = ts_data.copy()
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear

        # Lag features
        for lag in [1, 7, 14, 30]:
            df[f"lag_{lag}"] = df["count"].shift(lag)

        # Rolling statistics
        df["rolling_mean_7"] = df["count"].rolling(7).mean()
        df["rolling_std_7"] = df["count"].rolling(7).std()

        df = df.dropna()

        features = [
            "day_of_week",
            "month",
            "day_of_year",
            "lag_1",
            "lag_7",
            "lag_14",
            "lag_30",
            "rolling_mean_7",
            "rolling_std_7",
        ]

        X = df[features]
        y = df["count"]

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        with mlflow.start_run(run_name=f"xgboost_{category}"):
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)

            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Log parameters and metrics
            mlflow.log_param("model_type", "xgboost")
            mlflow.log_param("category", category)
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 6)

            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)

            # Log feature importance
            importance_df = pd.DataFrame(
                {"feature": features, "importance": model.feature_importances_}
            )
            csv_path = f"feature_importance_{category}.csv"
            importance_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)

            # Log model
            mlflow.xgboost.log_model(model, f"xgboost_{category}")

        return model

    def train_lstm(
        self, ts_data: pd.DataFrame, category: str
    ) -> tuple[LSTMForecaster, StandardScaler]:
        """Train LSTM model for a category"""
        # Detect GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training LSTM on device: {device}")

        # Prepare data
        values = ts_data["count"].values.reshape(-1, 1)
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(values)

        # Create sequences
        dataset = TimeSeriesDataset(scaled_values, self.seq_length)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Model
        model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2)
        model.to(device)  # Move model to GPU if available
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training
        model.train()
        epochs = 50

        with mlflow.start_run(run_name=f"lstm_{category}"):
            for epoch in range(epochs):
                total_loss = 0
                for x_batch, y_batch in dataloader:
                    # Move data to device
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    optimizer.zero_grad()
                    output = model(x_batch)
                    loss = criterion(output, y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(dataloader)
                mlflow.log_metric("train_loss", avg_loss, step=epoch)

            # Log parameters
            mlflow.log_param("model_type", "lstm")
            mlflow.log_param("category", category)
            mlflow.log_param("seq_length", self.seq_length)
            mlflow.log_param("hidden_size", 64)
            mlflow.log_param("num_layers", 2)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("device", str(device))

            # Log model
            mlflow.pytorch.log_model(model, f"lstm_{category}")

        return model, scaler

    def train_all_models(self):
        """Train all forecasting models for all categories"""
        logger.info("Starting time series forecasting model training...")

        df = self.load_data()
        # Top 10 categories
        categories = df["category"].value_counts().head(10).index.tolist()

        for category in categories:
            logger.info(f"Training models for category: {category}")

            ts_data = self.prepare_time_series(df, category)

            if len(ts_data) < 100:  # Skip if not enough data
                msg = f"Skipping {category}: insufficient data ({len(ts_data)} points)"
                logger.warning(msg)
                continue

            try:
                # Train Prophet
                prophet_model = self.train_prophet(ts_data, category)
                self.prophet_models[category] = prophet_model

                # Train XGBoost
                xgb_model = self.train_xgboost(ts_data, category)
                self.xgb_models[category] = xgb_model

                # Train LSTM
                lstm_model, scaler = self.train_lstm(ts_data, category)
                self.lstm_models[category] = lstm_model
                self.scalers[category] = scaler

                logger.info(f"Successfully trained all models for {category}")

            except Exception as e:
                logger.error(f"Failed to train models for {category}: {e}")
                continue

        # Save models
        self.save_models()
        logger.info("Time series forecasting training completed!")

    def forecast(self, category: str, days_ahead: int = 7) -> Dict:
        """Generate forecasts for a category using ensemble of models"""
        if category not in self.prophet_models:
            raise ValueError(f"No models trained for category: {category}")

        # Load models if not in memory
        if not self.prophet_models:
            self.load_models()

        # Generate future dates
        last_date = pd.Timestamp.now().normalize()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq="D"
        )

        # Prophet forecast
        prophet_future = pd.DataFrame({"ds": future_dates})
        prophet_forecast = self.prophet_models[category].predict(prophet_future)

        # XGBoost forecast (simplified - would need proper feature engineering)
        # For now, use Prophet as primary
        prophet_values = prophet_forecast["yhat"].values

        # LSTM forecast
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if category in self.lstm_models:
            lstm_model = self.lstm_models[category]
            scaler = self.scalers[category]
            lstm_model.to(device)
            lstm_model.eval()

            # Get recent data for LSTM input
            # This is simplified - in production you'd use actual recent data
            recent_values = np.array([prophet_values[0]] * self.seq_length)
            recent_values = recent_values.reshape(-1, 1)
            scaled_recent = scaler.transform(recent_values)

            with torch.no_grad():
                input_seq = torch.FloatTensor(scaled_recent)
                input_seq = input_seq.unsqueeze(0).to(device)
                lstm_pred = lstm_model(input_seq).cpu().numpy()
                lstm_pred_reshaped = lstm_pred.reshape(1, -1)
                lstm_value = scaler.inverse_transform(lstm_pred_reshaped)[0][0]
        else:
            lstm_value = prophet_values[0]  # Fallback

        # Ensemble: weighted average
        prophet_weight = 0.5
        lstm_weight = 0.3
        final_forecast = prophet_values * prophet_weight + lstm_value * lstm_weight

        # Ensure non-negative
        final_forecast = np.maximum(final_forecast, 0)

        return {
            "category": category,
            "dates": future_dates.strftime("%Y-%m-%d").tolist(),
            "forecast": final_forecast.tolist(),
            "confidence_lower": (final_forecast * 0.8).tolist(),  # Simplified
            "confidence_upper": (final_forecast * 1.2).tolist(),  # Simplified
            "model_info": {
                "prophet_weight": 0.7,
                "xgb_weight": 0.3,
                "method": "ensemble",
            },
        }

    def save_models(self):
        """Save trained models to disk"""
        model_dir = Path("models/forecasting")
        model_dir.mkdir(exist_ok=True)

        joblib.dump(
            {
                "prophet_models": self.prophet_models,
                "xgb_models": self.xgb_models,
                "lstm_models": self.lstm_models,
                "scalers": self.scalers,
                "category_mapping": self.category_mapping,
            },
            model_dir / "forecasting_models.pkl",
        )

        logger.info(f"Models saved to {model_dir}")

    def load_models(self):
        """Load trained models from disk"""
        model_dir = Path("models/forecasting")
        model_path = model_dir / "forecasting_models.pkl"

        if model_path.exists():
            models = joblib.load(model_path)
            self.prophet_models = models["prophet_models"]
            self.xgb_models = models["xgb_models"]
            self.lstm_models = models["lstm_models"]
            self.scalers = models["scalers"]
            self.category_mapping = models["category_mapping"]
            logger.info("Models loaded successfully")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")


# Global instance
_forecaster = None


def get_time_series_forecaster() -> TimeSeriesForecaster:
    """Get or create the global forecaster instance"""
    global _forecaster
    if _forecaster is None:
        _forecaster = TimeSeriesForecaster()
        try:
            _forecaster.load_models()
            logger.info("Forecasting models loaded successfully")
        except FileNotFoundError:
            logger.warning(
                "No trained forecasting models found. " "Train models first."
            )
    return _forecaster
