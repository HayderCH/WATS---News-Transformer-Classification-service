"""Database seeding helpers for local development and demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.db.models import Feedback, ReviewItem


@dataclass(frozen=True)
class SeedResult:
    inserted_feedback: int
    inserted_review_items: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "feedback": self.inserted_feedback,
            "review_items": self.inserted_review_items,
        }


SAMPLE_FEEDBACK = [
    {
        "text": (
            "Climate summit highlights urgent policy gaps as nations debate "
            "carbon targets."
        ),
        "predicted_label": "politics",
        "true_label": "politics",
        "model_version": "clf_v1",
        "confidence_score": 0.91,
    },
    {
        "text": (
            "Tech giants invest billions in next-gen AI chips amid global "
            "supply squeeze."
        ),
        "predicted_label": "technology",
        "true_label": "technology",
        "model_version": "clf_v1",
        "confidence_score": 0.88,
    },
]

SAMPLE_REVIEW_ITEMS = [
    {
        "text": (
            "Rookie quarterback rallies team to overtime upset win in "
            "season opener."
        ),
        "predicted_label": "sports",
        "confidence_score": 0.64,
        "confidence_margin": 0.08,
        "model_version": "clf_v1",
        "labeled": 0,
    },
    {
        "text": (
            "Central bank hints at rate cuts as inflation cools faster "
            "than expected."
        ),
        "predicted_label": "business",
        "confidence_score": 0.59,
        "confidence_margin": 0.12,
        "model_version": "clf_v1",
        "labeled": 0,
    },
]


def seed_initial_data(
    session: Session,
    *,
    overwrite: bool = False,
) -> SeedResult:
    """Seed feedback and review tables with representative sample rows.

    Args:
        session: Active SQLAlchemy session bound to the feedback database.
        overwrite: When True, existing rows are cleared before inserting the
            sample items.

    Returns:
        A SeedResult detailing how many rows were inserted per table.
    """

    if overwrite:
        session.execute(sa.delete(ReviewItem))
        session.execute(sa.delete(Feedback))
        session.commit()

    feedback_count = session.execute(
        sa.select(sa.func.count()).select_from(Feedback)
    ).scalar_one()
    review_count = session.execute(
        sa.select(sa.func.count()).select_from(ReviewItem)
    ).scalar_one()

    inserted_feedback = 0
    inserted_review = 0

    if overwrite or feedback_count == 0:
        for payload in SAMPLE_FEEDBACK:
            session.add(Feedback(**payload))
        inserted_feedback = len(SAMPLE_FEEDBACK)

    if overwrite or review_count == 0:
        for payload in SAMPLE_REVIEW_ITEMS:
            session.add(ReviewItem(**payload))
        inserted_review = len(SAMPLE_REVIEW_ITEMS)

    if inserted_feedback or inserted_review:
        session.commit()
    else:
        session.rollback()

    return SeedResult(inserted_feedback, inserted_review)
