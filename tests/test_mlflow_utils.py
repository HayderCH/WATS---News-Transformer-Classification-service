"""Tests for the MLflow utility helpers."""

import mlflow

from app.core.config import get_settings
from app.services.mlflow_utils import mlflow_run


def test_mlflow_run_disabled(monkeypatch):
    monkeypatch.delenv("MLFLOW_ENABLED", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    get_settings.cache_clear()

    with mlflow_run("disabled-test") as run_ctx:
        assert run_ctx is None

    get_settings.cache_clear()


def test_mlflow_run_records_metrics(monkeypatch, tmp_path):
    tracking_dir = tmp_path / "mlruns"
    tracking_dir.mkdir()
    monkeypatch.setenv("MLFLOW_ENABLED", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_dir.as_uri())
    monkeypatch.setenv("MLFLOW_EXPERIMENT", "unit-test-exp")
    monkeypatch.setenv("MLFLOW_TAGS", "env=test;component=training")
    get_settings.cache_clear()

    run_id = None

    with mlflow_run("baseline", tags={"custom": "tag"}) as run_ctx:
        assert run_ctx is not None
        run_ctx.log_metric("accuracy", 0.9)
        run_ctx.log_param("limit", "100")
        active_run = run_ctx.active_run()
        assert active_run is not None
        run_id = active_run.info.run_id

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_dir.as_uri())
    experiment = client.get_experiment_by_name("unit-test-exp")
    assert experiment is not None

    runs = client.search_runs([experiment.experiment_id])
    assert runs

    saved = next(run for run in runs if run.info.run_id == run_id)
    assert saved.data.metrics["accuracy"] == 0.9
    assert saved.data.params["limit"] == "100"
    # Tags should include defaults from settings and extra overrides
    assert saved.data.tags["component"] == "training"
    assert saved.data.tags["custom"] == "tag"

    get_settings.cache_clear()
    # Clean up environment overrides
    monkeypatch.delenv("MLFLOW_ENABLED", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT", raising=False)
    monkeypatch.delenv("MLFLOW_TAGS", raising=False)
