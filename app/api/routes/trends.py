from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.schemas import TrendsResponse
from app.services.trends import compute_trend_snapshot
from app.services.time_series_forecaster import get_time_series_forecaster

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/trends", response_model=TrendsResponse)
def read_trends(
    window: int = Query(7, ge=1, le=90),
    db: Session = Depends(get_db),
):
    snapshot = compute_trend_snapshot(db, window_days=window)
    return {
        "window_days": window,
        "generated_at": datetime.now(timezone.utc),
        **snapshot,
    }


@router.get("/forecast/{category}")
def forecast_category_trends(
    category: str,
    days_ahead: int = Query(7, ge=1, le=30, description="Days to forecast ahead"),
):
    """
    Forecast future trends for a specific news category using hybrid ML/DL models.

    Returns ensemble predictions from Prophet, XGBoost, and LSTM models.
    """
    try:
        forecaster = get_time_series_forecaster()
        forecast = forecaster.forecast(category, days_ahead)
        return forecast
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")


@router.post("/forecast/train")
def train_forecasting_models():
    """
    Train time series forecasting models for all categories.

    This is a long-running operation that trains Prophet, XGBoost, and LSTM models
    for the top news categories using historical data.
    """
    try:
        forecaster = get_time_series_forecaster()
        forecaster.train_all_models()
        return {"message": "Forecasting models training completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")
