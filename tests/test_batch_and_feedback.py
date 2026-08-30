from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)
client.headers.update({"x-api-key": "test-key"})


def test_classify_news_batch():
    payload = {
        "items": [
            {"title": "NASA update", "text": "New mission to Mars announced."},
            {"text": "The stock market rallied after the tech earnings beat."},
        ],
        "top_k": 3,
    }
    resp = client.post("/classify_news_batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data and isinstance(data["results"], list)
    assert len(data["results"]) == 2
    for r in data["results"]:
        assert "top_category" in r and "categories" in r


def test_feedback_flow():
    feedback = {
        "text": "A short sample.",
        "predicted_label": "TECH",
        "true_label": None,
        "model_version": "test_version",
        "confidence_score": 0.9,
        "confidence_margin": 0.5,
    }
    resp = client.post("/feedback", json=feedback)
    assert resp.status_code == 200
    ack = resp.json()
    assert ack["status"] == "ok" and ack["id"] > 0

    stats = client.get("/feedback/stats")
    assert stats.status_code == 200
    data = stats.json()
    assert "by_predicted" in data and isinstance(data["by_predicted"], dict)
