from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Integer, String, Float, DateTime, Text, JSON


class Base(DeclarativeBase):
    pass


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    text: Mapped[str] = mapped_column(Text)
    predicted_label: Mapped[str] = mapped_column(String(64))
    true_label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)


class ReviewItem(Base):
    __tablename__ = "review_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    text: Mapped[str] = mapped_column(Text)
    predicted_label: Mapped[str] = mapped_column(String(64))
    confidence_score: Mapped[float] = mapped_column(Float)
    confidence_margin: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    labeled: Mapped[int] = mapped_column(Integer, default=0)  # 0 = false, 1 = true
    true_label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    top_labels: Mapped[list[dict[str, float]] | None] = mapped_column(
        JSON, nullable=True
    )
    source: Mapped[str] = mapped_column(
        String(32), default="free_classification"
    )  # 'free_classification', 'streaming', 'manual'
    stream_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )  # Link to streaming session/article ID
    anomaly_score: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )  # Anomaly detection score for streaming articles
