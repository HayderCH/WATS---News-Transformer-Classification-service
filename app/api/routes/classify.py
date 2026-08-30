from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.models.schemas import ClassificationRequest, ClassificationResponse
from sqlalchemy.exc import SQLAlchemyError
from app.services.multimodal_classifier import classify_multimodal_news
from app.core.config import get_settings
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


@router.post("/classify_news", response_model=ClassificationResponse)
def classify(req: ClassificationRequest, db: Session = Depends(get_db)):
    result = classify_multimodal_news(
        title=req.title,
        text=req.text,
        image_url=req.image_url,
        image_base64=req.image_base64,
    )
    # Auto-enqueue if below thresholds
    try:
        confidence_score = float(result.get("confidence_score", 1.0))
        confidence_margin = float(result.get("confidence_margin", 1.0))
        needs_review = (
            confidence_score < _settings.review_conf_threshold
            or confidence_margin < _settings.review_margin_threshold
        )
        if needs_review:
            text = req.text if req.title is None else f"{req.title}. {req.text}"
            rec = ReviewItem(
                text=text,
                predicted_label=result.get("top_category", ""),
                confidence_score=confidence_score,
                confidence_margin=confidence_margin,
                model_version=result.get("model_version"),
                top_labels=result.get("categories"),
            )
            db.add(rec)
            db.commit()
    except SQLAlchemyError:
        # Do not fail the request if queuing encounters an error
        pass
    return ClassificationResponse(**result)
