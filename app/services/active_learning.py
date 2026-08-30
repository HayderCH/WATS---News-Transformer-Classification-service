"""Helpers for active learning workflows (review + feedback queues)."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Feedback, ReviewItem


def _normalize_label(raw: str | None) -> str:
    """Normalize human-entered labels to uppercase with underscores."""
    if not raw:
        return ""
    label = raw.strip()
    if not label:
        return ""
    label = label.replace("-", " ").replace("/", " ")
    parts = [part for part in label.upper().split() if part]
    return "_".join(parts)


def _collect_examples(
    session: Session,
    *,
    min_text_length: int = 5,
    training_threshold: int = 20,
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    """Collect labeled review items + feedback for active learning."""
    review_rows = (
        session.execute(
            select(ReviewItem)
            .where(ReviewItem.labeled == 1, ReviewItem.true_label.is_not(None))
            .order_by(ReviewItem.created_at.desc())
        )
        .scalars()
        .all()
    )

    feedback_rows = (
        session.execute(
            select(Feedback)
            .where(Feedback.true_label.is_not(None))
            .order_by(Feedback.created_at.desc())
        )
        .scalars()
        .all()
    )

    examples: List[Dict[str, str]] = []
    labels_seen: set[str] = set()
    latest: datetime | None = None
    review_count = 0
    feedback_count = 0

    def _append_example(
        text: str | None,
        label: str | None,
        created_at: datetime | None,
    ) -> None:
        nonlocal latest, review_count, feedback_count
        normalized_label = _normalize_label(label)
        content = (text or "").strip()
        if not normalized_label or len(content) < min_text_length:
            return
        examples.append(
            {
                "headline": "",
                "short_description": content,
                "category": normalized_label,
            }
        )
        labels_seen.add(normalized_label)
        if created_at:
            latest = created_at if latest is None else max(latest, created_at)

    for row in review_rows:
        before = len(examples)
        _append_example(row.text, row.true_label, row.created_at)
        if len(examples) > before:
            review_count += 1

    for row in feedback_rows:
        before = len(examples)
        _append_example(row.text, row.true_label, row.created_at)
        if len(examples) > before:
            feedback_count += 1

    total = len(examples)
    stats: Dict[str, object] = {
        "review_labeled": review_count,
        "feedback_labeled": feedback_count,
        "total_examples": total,
        "distinct_labels": sorted(labels_seen),
        "latest_label_at": latest,
        "training_threshold": training_threshold,
        "ready_for_training": (
            bool(total >= training_threshold) if training_threshold else total > 0
        ),
    }
    return examples, stats


def collect_active_learning_examples(
    session: Session,
    *,
    min_text_length: int = 5,
    training_threshold: int = 20,
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    """Return examples plus stats for training orchestration."""
    return _collect_examples(
        session,
        min_text_length=min_text_length,
        training_threshold=training_threshold,
    )


def collect_training_data(
    session: Session,
    *,
    sources: List[str] | None = None,
    max_confidence: float = 0.9,
    min_examples: int = 100,
    include_feedback: bool = True,
) -> List[Dict[str, str]]:
    """
    Collect human-reviewed data for model retraining.

    Args:
        session: Database session
        sources: List of sources to include ('free_classification', 'streaming', 'manual')
        max_confidence: Maximum confidence score to include (focus on uncertain examples)
        min_examples: Minimum number of examples to collect
        include_feedback: Whether to include feedback data

    Returns:
        List of training examples with text and labels
    """
    if sources is None:
        sources = ["free_classification", "streaming"]

    examples: List[Dict[str, str]] = []

    # Collect from ReviewItem table
    query = select(ReviewItem).where(
        ReviewItem.labeled == 1,
        ReviewItem.true_label.isnot(None),
        ReviewItem.confidence_score <= max_confidence,
    )

    if sources:
        query = query.where(ReviewItem.source.in_(sources))

    query = query.order_by(ReviewItem.created_at.desc()).limit(min_examples * 2)

    review_rows = session.execute(query).scalars().all()

    for row in review_rows:
        content = (row.text or "").strip()
        if len(content) >= 10:  # Minimum text length
            examples.append(
                {
                    "headline": "",
                    "short_description": content,
                    "category": _normalize_label(row.true_label),
                    "source": row.source,
                    "confidence": row.confidence_score,
                    "created_at": (
                        row.created_at.isoformat() if row.created_at else None
                    ),
                }
            )

    # Optionally include feedback data
    if include_feedback and len(examples) < min_examples:
        feedback_rows = (
            session.execute(
                select(Feedback)
                .where(Feedback.true_label.isnot(None))
                .order_by(Feedback.created_at.desc())
                .limit(min_examples - len(examples))
            )
            .scalars()
            .all()
        )

        for row in feedback_rows:
            content = (row.text or "").strip()
            if len(content) >= 10:
                examples.append(
                    {
                        "headline": "",
                        "short_description": content,
                        "category": _normalize_label(row.true_label),
                        "source": "feedback",
                        "confidence": row.confidence_score or 0.0,
                        "created_at": (
                            row.created_at.isoformat() if row.created_at else None
                        ),
                    }
                )

    # Limit to min_examples
    return examples[:min_examples]


def active_learning_stats(
    session: Session,
    *,
    min_text_length: int = 5,
    training_threshold: int = 20,
) -> Dict[str, object]:
    """Return stats without materializing the example payload."""
    _, stats = _collect_examples(
        session,
        min_text_length=min_text_length,
        training_threshold=training_threshold,
    )
    return stats
