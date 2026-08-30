from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
client.headers.update({"x-api-key": "test-key"})


def test_summarize_batch_happy_and_empty(monkeypatch):
    # Patch summarizer to avoid heavy model invocation.
    from app.api.routes import summarize as summarize_route

    def fake_summarize(
        text: str, max_len: int | None = None, min_len: int | None = None
    ):
        return {
            "summary": f"summary:{text[:8]}",
            "model_version": "fake-sum",
            "latency_ms": 1.0,
            "cached": False,
        }

    monkeypatch.setattr(summarize_route, "summarize_text", fake_summarize)

    payload = {
        "items": [
            {"text": "Article number one", "max_len": 60, "min_len": 10},
            {"text": "Article number two"},
        ]
    }
    resp = client.post("/summarize_batch", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 2
    assert all(r["model_version"] == "fake-sum" for r in body["results"])

    empty_resp = client.post("/summarize_batch", json={"items": []})
    assert empty_resp.status_code == 200
    assert empty_resp.json()["results"] == []


def test_export_dataset_filters(monkeypatch):
    # Ensure a clean window for queries
    window_start = (datetime.utcnow() - timedelta(seconds=1)).isoformat()

    # Enqueue and label a review item
    review_payload = {
        "text": "dataset review example text",
        "predicted_label": "TECH",
        "confidence_score": 0.3,
        "confidence_margin": 0.02,
        "model_version": "test-ds",
    }
    enqueue_resp = client.post("/review/enqueue", json=review_payload)
    assert enqueue_resp.status_code == 200
    review_id = enqueue_resp.json()["id"]
    label_resp = client.post(
        "/review/label",
        json={"item_id": review_id, "true_label": "SPORTS"},
    )
    assert label_resp.status_code == 200

    # Submit feedback with ground truth
    feedback_payload = {
        "text": "dataset feedback example text",
        "predicted_label": "BUSINESS",
        "true_label": "BUSINESS",
        "model_version": "test-ds",
        "confidence_score": 0.4,
    }
    fb_resp = client.post("/feedback", json=feedback_payload)
    assert fb_resp.status_code == 200

    base_params = {"date_from": window_start}
    data_all = client.get("/export/dataset", params=base_params)
    assert data_all.status_code == 200
    items = data_all.json()["items"]
    assert any(
        it["source"] == "review" and it["text"] == review_payload["text"]
        for it in items
    )
    assert any(
        it["source"] == "feedback" and it["text"] == feedback_payload["text"]
        for it in items
    )

    review_only = client.get(
        "/export/dataset",
        params={"source": "review", "date_from": window_start},
    )
    assert review_only.status_code == 200
    review_items = review_only.json()["items"]
    assert review_items
    assert all(it["source"] == "review" for it in review_items)

    # Future window should yield empty set
    future_start = (datetime.utcnow() + timedelta(seconds=5)).isoformat()
    future_resp = client.get(
        "/export/dataset",
        params={"date_from": future_start},
    )
    assert future_resp.status_code == 200
    assert future_resp.json()["count"] == 0


def test_classify_auto_enqueues(monkeypatch):
    from app.api.routes import classify as classify_route

    def fake_classify_text(text: str):
        return {
            "top_category": "TECH",
            "categories": [
                {"name": "TECH", "prob": 0.55},
                {"name": "SPORTS", "prob": 0.45},
            ],
            "confidence_level": "LOW",
            "confidence_score": 0.55,
            "confidence_margin": 0.1,
            "model_version": "stub",
            "latency_ms": 1.0,
        }

    monkeypatch.setattr(classify_route, "classify_text", fake_classify_text)

    stats_before = client.get("/review/stats").json()["unlabeled"]
    classify_resp = client.post(
        "/classify_news",
        json={"text": "Short ambiguous sample that should trigger review."},
    )
    assert classify_resp.status_code == 200
    stats_after = client.get("/review/stats").json()["unlabeled"]
    assert stats_after >= stats_before + 1
