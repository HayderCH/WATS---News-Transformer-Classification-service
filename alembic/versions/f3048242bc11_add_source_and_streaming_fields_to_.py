"""add source and streaming fields to review items

Revision ID: f3048242bc11
Revises: 20241002_0002
Create Date: 2025-10-11 00:00:00
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "f3048242bc11"
down_revision = "20241002_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "review_items",
        sa.Column(
            "source",
            sa.String(32),
            nullable=False,
            server_default="free_classification",
        ),
    )
    op.add_column(
        "review_items",
        sa.Column("stream_id", sa.String(64), nullable=True),
    )
    op.add_column(
        "review_items",
        sa.Column("anomaly_score", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("review_items", "anomaly_score")
    op.drop_column("review_items", "stream_id")
    op.drop_column("review_items", "source")
