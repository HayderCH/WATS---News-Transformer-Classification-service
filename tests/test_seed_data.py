"""Unit tests for database seeding helpers."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, Feedback, ReviewItem
from app.db.seed import SAMPLE_FEEDBACK, SAMPLE_REVIEW_ITEMS, seed_initial_data


def _build_session(tmp_path):
    db_path = tmp_path / "seed_test.db"
    engine = sa.create_engine(f"sqlite:///{db_path}", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        future=True,
    )
    return engine, Session


def test_seed_inserts_sample_rows_when_empty(tmp_path):
    engine, Session = _build_session(tmp_path)
    try:
        with Session() as session:
            result = seed_initial_data(session, overwrite=False)
            assert result.inserted_feedback == len(SAMPLE_FEEDBACK)
            assert result.inserted_review_items == len(SAMPLE_REVIEW_ITEMS)

            feedback_count = session.execute(
                sa.select(sa.func.count()).select_from(Feedback)
            ).scalar_one()
            review_count = session.execute(
                sa.select(sa.func.count()).select_from(ReviewItem)
            ).scalar_one()
            assert feedback_count == len(SAMPLE_FEEDBACK)
            assert review_count == len(SAMPLE_REVIEW_ITEMS)
    finally:
        engine.dispose()


def test_seed_is_idempotent_without_overwrite(tmp_path):
    engine, Session = _build_session(tmp_path)
    try:
        with Session() as session:
            seed_initial_data(session, overwrite=False)
            repeat = seed_initial_data(session, overwrite=False)
            assert repeat.inserted_feedback == 0
            assert repeat.inserted_review_items == 0

            total_feedback = session.execute(
                sa.select(sa.func.count()).select_from(Feedback)
            ).scalar_one()
            total_review = session.execute(
                sa.select(sa.func.count()).select_from(ReviewItem)
            ).scalar_one()
            assert total_feedback == len(SAMPLE_FEEDBACK)
            assert total_review == len(SAMPLE_REVIEW_ITEMS)
    finally:
        engine.dispose()


def test_seed_overwrite_replaces_existing_rows(tmp_path):
    engine, Session = _build_session(tmp_path)
    try:
        with Session() as session:
            seed_initial_data(session, overwrite=False)
            session.execute(
                sa.update(Feedback).values(predicted_label="placeholder")
            )
            session.commit()

            refreshed = seed_initial_data(session, overwrite=True)
            assert refreshed.inserted_feedback == len(SAMPLE_FEEDBACK)
            assert refreshed.inserted_review_items == len(SAMPLE_REVIEW_ITEMS)

            labels = (
                session.execute(sa.select(Feedback.predicted_label))
                .scalars()
                .all()
            )
            assert "placeholder" not in labels
    finally:
        engine.dispose()
