from collections import Counter
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.schemas import FeedbackIn, FeedbackAck, FeedbackStats
from app.db.session import SessionLocal, engine
from app.db.models import Base, Feedback
from app.core.security import require_api_key


router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Ensure tables exist on first import
Base.metadata.create_all(bind=engine)


@router.post("/feedback", response_model=FeedbackAck)
def submit_feedback(
    item: FeedbackIn,
    db: Session = Depends(get_db),
    _: None = Depends(require_api_key),
):
    rec = Feedback(
        text=item.text,
        predicted_label=item.predicted_label,
        true_label=item.true_label,
        model_version=item.model_version,
        confidence_score=item.confidence_score,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return FeedbackAck(id=rec.id)


@router.get("/feedback/stats", response_model=FeedbackStats)
def feedback_stats(db: Session = Depends(get_db)):
    rows = db.execute(
        select(Feedback.predicted_label, Feedback.true_label)
    ).all()
    pred = Counter()
    true = Counter()
    for p, t in rows:
        if p:
            pred[p] += 1
        if t:
            true[t] += 1
    return FeedbackStats(by_predicted=dict(pred), by_true=dict(true))
