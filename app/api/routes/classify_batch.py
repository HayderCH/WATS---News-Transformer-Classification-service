from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from app.models.schemas import (
    BatchClassificationRequest,
    BatchClassificationResponse,
    ClassificationResponse,
)
from app.services.classifier import classify_batch
from app.core.config import get_settings
from app.core.security import require_api_key
from app.db.session import SessionLocal
from app.db.models import ReviewItem


router = APIRouter()
_settings = get_settings()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post(
    "/classify_news_batch",
    response_model=BatchClassificationResponse,
)
def classify_batch_route(
    req: BatchClassificationRequest,
    db: Session = Depends(get_db),
    _: None = Depends(require_api_key),
):
    items = [item.model_dump() for item in req.items]
    results = classify_batch(items, top_k=req.top_k)
    # Auto-enqueue any low-confidence predictions
    try:
        for src, res in zip(items, results):
            if (
                res.get("confidence_score", 1.0)
                < _settings.review_conf_threshold
                or res.get("confidence_margin", 1.0)
                < _settings.review_margin_threshold
            ):
                text = (
                    f"{src.get('title')}. {src.get('text')}"
                    if src.get("title")
                    else src.get("text", "")
                )
                rec = ReviewItem(
                    text=text,
                    predicted_label=res.get("top_category", ""),
                    confidence_score=float(res.get("confidence_score", 0.0)),
                    confidence_margin=float(res.get("confidence_margin", 0.0)),
                    model_version=res.get("model_version"),
                )
                db.add(rec)
        db.commit()
    except SQLAlchemyError:
        pass
    typed = [ClassificationResponse(**r) for r in results]
    return BatchClassificationResponse(results=typed)
