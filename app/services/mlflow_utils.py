from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, Optional

import mlflow

from app.core.config import get_settings


def _combine_tags(extra: Optional[Dict[str, str]]) -> Dict[str, str]:
    settings = get_settings()
    default_tags = settings.mlflow_tags()
    if not extra:
        return default_tags
    merged = dict(default_tags)
    merged.update({k: str(v) for k, v in extra.items()})
    return merged


@contextmanager
def mlflow_run(
    prefix: str,
    *,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Iterator[Optional[Any]]:
    """Context manager that activates an MLflow run if enabled in settings."""

    settings = get_settings()
    if not settings.mlflow_enabled:
        yield None
        return

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    template = settings.mlflow_run_template or "{prefix}-{timestamp}"
    resolved_name = run_name or template.format(
        prefix=prefix,
        timestamp=timestamp,
    )

    with mlflow.start_run(run_name=resolved_name):
        combined_tags = _combine_tags(tags)
        if combined_tags:
            mlflow.set_tags(combined_tags)
        yield mlflow


__all__ = [
    "mlflow_run",
]
