from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_summarize():
    payload = {
        "text": (
            "The European Space Agency launched a new orbital platform to "
            "study exoplanet atmospheres and gather high-resolution "
            "spectral data for future climate modeling synergies."
        )
    }
    resp = client.post("/summarize", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert len(data["summary"].split()) > 3
