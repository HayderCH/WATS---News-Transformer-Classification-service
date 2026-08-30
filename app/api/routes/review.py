from datetime import datetime

from fastapi import APIRouter, Depends, Body
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy import select, update

from app.core.config import get_settings
from app.db.session import SessionLocal, engine
from app.db.models import Base, ReviewItem, Feedback
from app.models.schemas import ReviewEnqueueIn, ReviewQueuedAck, ReviewLabelIn
from app.core.security import require_api_key
from app.services.active_learning import active_learning_stats
from app.services.classifier import classify_text


router = APIRouter()
Base.metadata.create_all(bind=engine)
_settings = get_settings()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/review/enqueue", response_model=ReviewQueuedAck)
def enqueue_review(
    item: ReviewEnqueueIn = Body(...),
    db: Session = Depends(get_db),
    _: None = Depends(require_api_key),
):
    """Queue an item for human review if below confidence/margin thresholds."""
    if (
        item.confidence_score < _settings.review_conf_threshold
        or item.confidence_margin < _settings.review_margin_threshold
    ):
        rec = ReviewItem(
            text=item.text,
            predicted_label=item.predicted_label,
            confidence_score=item.confidence_score,
            confidence_margin=item.confidence_margin,
            model_version=item.model_version,
            top_labels=(
                [label.model_dump() for label in item.top_labels]
                if item.top_labels
                else None
            ),
            source=item.source,
            stream_id=item.stream_id,
            anomaly_score=item.anomaly_score,
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return ReviewQueuedAck(queued=True, id=rec.id)
    return ReviewQueuedAck(queued=False, id=None)


@router.get("/review/queue")
def get_review_queue(
    limit: int = 20,
    offset: int = 0,
    predicted_label: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sort_by: str = "id",
    sort_order: str = "asc",
    source: str | None = None,  # New parameter for filtering by source
    db: Session = Depends(get_db),
):
    q = select(ReviewItem).where(ReviewItem.labeled == 0)
    if predicted_label:
        q = q.where(ReviewItem.predicted_label == predicted_label)
    if source:
        q = q.where(ReviewItem.source == source)
    try:
        if date_from:
            q = q.where(ReviewItem.created_at >= datetime.fromisoformat(date_from))
    except ValueError:
        pass
    try:
        if date_to:
            q = q.where(ReviewItem.created_at <= datetime.fromisoformat(date_to))
    except ValueError:
        pass

    sort_by_normalized = sort_by.lower()
    sort_order_normalized = sort_order.lower()
    if sort_by_normalized not in {"id", "created_at"}:
        sort_by_normalized = "id"
    if sort_order_normalized not in {"asc", "desc"}:
        sort_order_normalized = "asc"
    order_column = (
        ReviewItem.created_at if sort_by_normalized == "created_at" else ReviewItem.id
    )
    if sort_order_normalized == "desc":
        order_column = order_column.desc()
    else:
        order_column = order_column.asc()

    q = q.order_by(order_column).offset(offset).limit(limit)
    rows = db.execute(q).scalars().all()

    updates: list[tuple[int, list[dict[str, float]]]] = []
    payload: list[dict[str, object]] = []

    for r in rows:
        top_labels = r.top_labels or []
        if not top_labels:
            result = classify_text(r.text)
            top_labels = result.get("categories") or []
            if top_labels:
                updates.append((r.id, top_labels))

        payload.append(
            {
                "id": r.id,
                "text": r.text,
                "predicted_label": r.predicted_label,
                "confidence_score": r.confidence_score,
                "confidence_margin": r.confidence_margin,
                "model_version": r.model_version,
                "created_at": (r.created_at.isoformat() if r.created_at else None),
                "top_labels": top_labels,
            }
        )

    if updates:
        for item_id, labels in updates:
            db.execute(
                update(ReviewItem)
                .where(ReviewItem.id == item_id)
                .values(top_labels=labels)
            )
        db.commit()

    return payload


@router.get("/review/stream-queue")
def get_streaming_review_queue(
    limit: int = 20,
    offset: int = 0,
    predicted_label: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sort_by: str = "id",
    sort_order: str = "asc",
    db: Session = Depends(get_db),
):
    """Get review queue specifically for streaming articles"""
    return get_review_queue(
        limit=limit,
        offset=offset,
        predicted_label=predicted_label,
        date_from=date_from,
        date_to=date_to,
        sort_by=sort_by,
        sort_order=sort_order,
        source="streaming",
        db=db,
    )


@router.post("/review/label")
def label_review_item(
    payload: ReviewLabelIn = Body(...),
    db: Session = Depends(get_db),
    _: None = Depends(require_api_key),
):
    res = db.execute(
        update(ReviewItem)
        .where(ReviewItem.id == payload.item_id)
        .values(true_label=payload.true_label, labeled=1)
    )
    db.commit()
    return {"updated": res.rowcount}


@router.get("/review/stats")
def review_stats(db: Session = Depends(get_db)):
    total = db.execute(select(func.count()).select_from(ReviewItem)).scalar_one()
    unlabeled = db.execute(
        select(func.count()).select_from(ReviewItem).where(ReviewItem.labeled == 0)
    ).scalar_one()
    labeled = total - unlabeled
    # breakdowns
    by_predicted = dict(
        db.execute(
            select(ReviewItem.predicted_label, func.count())
            .where(ReviewItem.labeled == 0)
            .group_by(ReviewItem.predicted_label)
        ).all()
    )
    by_true = dict(
        db.execute(
            select(ReviewItem.true_label, func.count())
            .where(ReviewItem.labeled == 1)
            .group_by(ReviewItem.true_label)
        ).all()
    )
    # Source breakdowns
    by_source_total = dict(
        db.execute(
            select(ReviewItem.source, func.count()).group_by(ReviewItem.source)
        ).all()
    )
    by_source_unlabeled = dict(
        db.execute(
            select(ReviewItem.source, func.count())
            .where(ReviewItem.labeled == 0)
            .group_by(ReviewItem.source)
        ).all()
    )
    return {
        "total": int(total),
        "unlabeled": int(unlabeled),
        "labeled": int(labeled),
        "by_predicted_unlabeled": by_predicted,
        "by_true_labeled": by_true,
        "by_source_total": by_source_total,
        "by_source_unlabeled": by_source_unlabeled,
    }


@router.get("/review/active-learning")
def review_active_learning(
    threshold: int = 20,
    db: Session = Depends(get_db),
):
    stats = active_learning_stats(db, training_threshold=threshold)
    latest = stats.get("latest_label_at")
    if isinstance(latest, datetime):
        stats["latest_label_at"] = latest.isoformat()
    return stats


@router.get("/export/dataset")
def export_dataset(
    source: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    db: Session = Depends(get_db),
    _: None = Depends(require_api_key),
):
    """
    Export a simple combined dataset of labeled review items and feedback with
    true labels.
    Shape: list of { text, true_label, predicted_label, source,
    model_version, created_at }
    """
    source_normalized = source.lower() if source else "all"
    if source_normalized not in {"review", "feedback", "all"}:
        source_normalized = "all"
    from_dt = None
    to_dt = None
    try:
        if date_from:
            from_dt = datetime.fromisoformat(date_from)
    except ValueError:
        from_dt = None
    try:
        if date_to:
            to_dt = datetime.fromisoformat(date_to)
    except ValueError:
        to_dt = None

    review_rows = []
    fb_rows = []
    if source_normalized in {"review", "all"}:
        review_query = select(
            ReviewItem.text,
            ReviewItem.true_label,
            ReviewItem.predicted_label,
            ReviewItem.model_version,
            ReviewItem.created_at,
        ).where(ReviewItem.labeled == 1)
        if from_dt:
            review_query = review_query.where(ReviewItem.created_at >= from_dt)
        if to_dt:
            review_query = review_query.where(ReviewItem.created_at <= to_dt)
        review_rows = db.execute(review_query).all()

    if source_normalized in {"feedback", "all"}:
        feedback_query = select(
            Feedback.text,
            Feedback.true_label,
            Feedback.predicted_label,
            Feedback.model_version,
            Feedback.created_at,
        ).where(Feedback.true_label.is_not(None))
        if from_dt:
            feedback_query = feedback_query.where(Feedback.created_at >= from_dt)
        if to_dt:
            feedback_query = feedback_query.where(Feedback.created_at <= to_dt)
        fb_rows = db.execute(feedback_query).all()
    data = []
    for t, tl, pl, mv, ts in review_rows:
        data.append(
            {
                "text": t,
                "true_label": tl,
                "predicted_label": pl,
                "source": "review",
                "model_version": mv,
                "created_at": ts.isoformat(),
            }
        )
    for t, tl, pl, mv, ts in fb_rows:
        data.append(
            {
                "text": t,
                "true_label": tl,
                "predicted_label": pl,
                "source": "feedback",
                "model_version": mv,
                "created_at": ts.isoformat(),
            }
        )
    return {"count": len(data), "items": data}
