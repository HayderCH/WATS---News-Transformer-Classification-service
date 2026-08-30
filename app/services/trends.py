"""Utilities for aggregating topic trend statistics."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Tuple

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.db.models import Feedback, ReviewItem


def _iter_trend_rows(
    session: Session,
    model,
    cutoff: datetime,
) -> Iterable[Tuple[str, str, int]]:
    stmt = (
        sa.select(
            sa.func.date(model.created_at).label("bucket"),
            model.predicted_label,
            sa.func.count().label("count"),
        )
        .where(model.created_at >= cutoff)
        .group_by("bucket", model.predicted_label)
    )
    results = session.execute(stmt)
    for bucket_date, label, count in results:
        if not label:
            continue
        if hasattr(bucket_date, "isoformat"):
            bucket_key = bucket_date.isoformat()
        else:
            bucket_key = str(bucket_date)
        yield bucket_key, label, int(count or 0)


def compute_trend_snapshot(
    session: Session,
    *,
    window_days: int = 7,
) -> dict[str, List[dict[str, object]]]:
    """Aggregate recent topic counts from feedback and review queues."""

    cutoff = datetime.utcnow() - timedelta(days=window_days)

    bucket_counts: Dict[Tuple[str, str], int] = {}
    total_counts: Dict[str, int] = {}

    for model in (Feedback, ReviewItem):
        trend_rows = _iter_trend_rows(session, model, cutoff)
        for bucket_key, label, count in trend_rows:
            key = (bucket_key, label)
            bucket_counts[key] = bucket_counts.get(key, 0) + count
            total_counts[label] = total_counts.get(label, 0) + count

    buckets: List[dict[str, object]] = []
    for date_key, label in sorted(bucket_counts.keys()):
        buckets.append(
            {
                "date": date_key,
                "label": label,
                "count": bucket_counts[(date_key, label)],
            }
        )
    totals = [
        {"label": label, "count": count}
        for label, count in sorted(
            total_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]

    return {"buckets": buckets, "totals": totals}
