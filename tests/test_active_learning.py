"""Unit tests for active learning helper utilities."""

from __future__ import annotations

from datetime import datetime, timedelta

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, Feedback, ReviewItem
from app.services.active_learning import (
    collect_active_learning_examples,
    active_learning_stats,
)


def _prepare_session(tmp_path):
    db_path = tmp_path / "active_learning.db"
    engine = sa.create_engine(f"sqlite:///{db_path.as_posix()}", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)
    return engine, Session


def test_collect_active_learning_examples_counts(tmp_path):
    engine, Session = _prepare_session(tmp_path)
    try:
        with Session() as session:
            now = datetime.utcnow()
            session.add_all(
                [
                    ReviewItem(
                        text=(
                            "Detailed news analysis on the latest"
                            " tech trends."
                        ),
                        predicted_label="TECH",
                        confidence_score=0.18,
                        confidence_margin=0.02,
                        model_version="v-test",
                        labeled=1,
                        true_label="Tech Innovations",
                        created_at=now - timedelta(hours=1),
                    ),
                    Feedback(
                        text=(
                            "User feedback on business markets "
                            "and finance outlook."
                        ),
                        predicted_label="BUSINESS",
                        true_label="Business",
                        model_version="v-test",
                        confidence_score=0.91,
                        created_at=now,
                    ),
                ]
            )
            session.commit()

            examples, stats = collect_active_learning_examples(
                session,
                training_threshold=1,
            )

        assert len(examples) == 2
        assert stats["review_labeled"] == 1
        assert stats["feedback_labeled"] == 1
        assert stats["total_examples"] == 2
        assert stats["ready_for_training"] is True
        assert set(stats["distinct_labels"]) == {
            "BUSINESS",
            "TECH_INNOVATIONS",
        }
        assert isinstance(stats["latest_label_at"], datetime)
        # Ensure normalized payload matches expected schema
        for row in examples:
            assert set(row.keys()) == {
                "headline",
                "short_description",
                "category",
            }
            assert row["short_description"]
            assert row["category"].isupper()
    finally:
        engine.dispose()


def test_collect_examples_respects_min_text_length(tmp_path):
    engine, Session = _prepare_session(tmp_path)
    try:
        with Session() as session:
            session.add(
                ReviewItem(
                    text="Tiny",
                    predicted_label="TECH",
                    confidence_score=0.3,
                    confidence_margin=0.1,
                    model_version="v-test",
                    labeled=1,
                    true_label="Tech",
                )
            )
            session.commit()
            examples, stats = collect_active_learning_examples(
                session,
                min_text_length=10,
                training_threshold=1,
            )
        assert examples == []
        assert stats["total_examples"] == 0
        assert stats["ready_for_training"] is False
    finally:
        engine.dispose()


def test_active_learning_stats_matches_examples(tmp_path):
    engine, Session = _prepare_session(tmp_path)
    try:
        with Session() as session:
            session.add(
                Feedback(
                    text="Extended commentary on politics and world affairs.",
                    predicted_label="POLITICS",
                    true_label="politics",
                    model_version="v-test",
                    confidence_score=0.77,
                )
            )
            session.commit()
            stats = active_learning_stats(session, training_threshold=2)
        assert stats["feedback_labeled"] == 1
        assert stats["review_labeled"] == 0
        assert stats["total_examples"] == 1
        assert stats["ready_for_training"] is False
        assert stats["training_threshold"] == 2
    finally:
        engine.dispose()
