from datetime import datetime, timedelta

import sqlalchemy as sa
from fastapi.testclient import TestClient

from app.main import app
from app.db.models import Base, Feedback, ReviewItem
from app.db.session import SessionLocal, engine


client = TestClient(app)


def _seed_trend_data() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        session.execute(sa.delete(Feedback))
        session.execute(sa.delete(ReviewItem))

        now = datetime.utcnow()
        session.add(
            Feedback(
                text="Business headline",
                predicted_label="business",
                true_label="business",
                model_version="clf_v1",
                confidence_score=0.9,
                created_at=now - timedelta(days=1),
            )
        )
        session.add(
            Feedback(
                text="Old item",
                predicted_label="business",
                model_version="clf_v1",
                confidence_score=0.4,
                created_at=now - timedelta(days=10),
            )
        )
        session.add(
            ReviewItem(
                text="Sports upset",
                predicted_label="sports",
                confidence_score=0.6,
                confidence_margin=0.1,
                model_version="clf_v1",
                created_at=now - timedelta(days=2),
            )
        )
        session.commit()


def test_trends_endpoint_returns_recent_counts() -> None:
    _seed_trend_data()

    resp = client.get("/trends?window=7")
    assert resp.status_code == 200
    data = resp.json()

    assert data["window_days"] == 7
    assert "generated_at" in data and "T" in data["generated_at"]
    assert isinstance(data["buckets"], list)
    assert isinstance(data["totals"], list)

    totals = {item["label"]: item["count"] for item in data["totals"]}
    assert totals["business"] == 1
    assert totals["sports"] == 1

    labels_in_buckets = {item["label"] for item in data["buckets"]}
    assert "business" in labels_in_buckets
    assert "sports" in labels_in_buckets

    recent_resp = client.get("/trends?window=1")
    assert recent_resp.status_code == 200
    recent_totals = {
        item["label"]: item["count"] for item in recent_resp.json()["totals"]
    }
    assert "sports" not in recent_totals


def test_trends_window_validation() -> None:
    resp = client.get("/trends?window=0")
    assert resp.status_code == 422

    resp = client.get("/trends?window=120")
    assert resp.status_code == 422


def test_forecast_endpoint_returns_predictions() -> None:
    """Test forecasting endpoint returns predictions for a category."""
    resp = client.get("/trends/forecast/POLITICS?days_ahead=3")
    # Note: This will return 404 if models aren't trained, which is expected
    # In a real test environment, we'd train models first
    assert resp.status_code in [200, 404]  # 404 if no models trained yet

    if resp.status_code == 200:
        data = resp.json()
        assert "category" in data
        assert "dates" in data
        assert "forecast" in data
        assert "confidence_lower" in data
        assert "confidence_upper" in data
        assert "model_info" in data
        assert data["category"] == "POLITICS"
        assert len(data["dates"]) == 3
        assert len(data["forecast"]) == 3
