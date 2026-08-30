from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from app.db.session import SessionLocal
from app.db.models import ReviewItem, Feedback
from app.services.summarizer import _SummarizerHolder
import torch

from app.core.config import get_settings
from app.services.classifier import _classifier_holder
from app.core.metrics import METRICS_STATE
from app.core.security import require_api_key


router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/metrics")
def metrics(db: Session = Depends(get_db)) -> Dict[str, Any]:
    settings = get_settings()
    # Ensure model is loaded to expose device and labels
    _classifier_holder.load()

    info: Dict[str, Any] = {
        "backend": _classifier_holder.backend,
        "model_version": settings.classifier_version,
        "device": str(_classifier_holder.device),
        "label_count": len(getattr(_classifier_holder, "label_names", {})),
        "summarizer_device": str(_SummarizerHolder.device),
    }

    # Include label samples (first 10) for visibility
    labels = getattr(_classifier_holder, "label_names", {})
    if isinstance(labels, dict) and labels:
        try:
            ordered = [labels[i] for i in sorted(labels)]
            info["labels_preview"] = ordered[:10]
        except (KeyError, TypeError):
            # Fallback to raw dict keys
            info["labels_preview"] = list(labels.values())[:10]

    # Try to read saved eval metrics from transformer directory
    metrics_path = Path(settings.transformer_model_dir) / "metrics.txt"
    if metrics_path.exists():
        try:
            eval_metrics: Dict[str, float] = {}
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    try:
                        eval_metrics[k] = float(v)
                    except ValueError:
                        # ignore non-numeric values
                        continue
            if eval_metrics:
                info["eval_metrics"] = eval_metrics
        except OSError:
            # ignore file read errors
            pass

    # Live counters from DB
    try:
        unlabeled_reviews = db.execute(
            select(func.count())
            .select_from(ReviewItem)
            .where(ReviewItem.labeled == 0)
        ).scalar_one()
        feedback_total = db.execute(
            select(func.count()).select_from(Feedback)
        ).scalar_one()
        info["review_queue_unlabeled"] = int(unlabeled_reviews)
        info["feedback_total"] = int(feedback_total)
    except Exception:
        # If DB not available for any reason, skip live counters
        pass

    # CUDA diagnostics
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            # Some builds expose bf16 capability check
            is_bf16 = getattr(torch.cuda, "is_bf16_supported", None)
            if callable(is_bf16):
                info["cuda_bf16_supported"] = bool(is_bf16())
    except Exception:
        pass

    # In-memory request metrics (populated by middleware)
    info["request_counters"] = METRICS_STATE.get_counters()
    info["latency_ms"] = METRICS_STATE.get_latencies()

    return info


@router.post("/metrics/reset", dependencies=[Depends(require_api_key)])
def metrics_reset() -> Dict[str, str]:
    METRICS_STATE.reset()
    return {"status": "reset"}
