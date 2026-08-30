"""add top labels json column to review items

Revision ID: 20241002_0002
Revises: 20240930_0001
Create Date: 2025-10-02 00:00:00
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20241002_0002"
down_revision = "20240930_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "review_items",
        sa.Column("top_labels", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("review_items", "top_labels")
