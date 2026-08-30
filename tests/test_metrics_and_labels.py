from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_metrics_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    for key in ["backend", "model_version", "device", "label_count"]:
        assert key in data
    assert isinstance(data["label_count"], int)


def test_labels_endpoint():
    resp = client.get("/labels")
    assert resp.status_code == 200
    data = resp.json()
    assert "count" in data and "labels" in data
    assert isinstance(data["labels"], list)
    assert data["count"] == len(data["labels"]) >= 4


def test_metrics_reset_requires_api_key():
    unauth = TestClient(app)
    resp = unauth.post("/metrics/reset")
    assert resp.status_code == 401

    auth_client = TestClient(app)
    auth_client.headers.update({"x-api-key": "test-key"})
    resp_ok = auth_client.post("/metrics/reset")
    assert resp_ok.status_code == 200
    assert resp_ok.json()["status"] == "reset"
