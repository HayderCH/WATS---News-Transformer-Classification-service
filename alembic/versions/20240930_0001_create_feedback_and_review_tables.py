"""
Alembic migration: create feedback and review_items tables
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20240930_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "created_at",
            sa.DateTime,
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("predicted_label", sa.String(64), nullable=False),
        sa.Column("true_label", sa.String(64), nullable=True),
        sa.Column("model_version", sa.String(128), nullable=True),
        sa.Column("confidence_score", sa.Float, nullable=True),
        if_not_exists=True,
    )
    op.create_table(
        "review_items",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "created_at",
            sa.DateTime,
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("predicted_label", sa.String(64), nullable=False),
        sa.Column("confidence_score", sa.Float, nullable=False),
        sa.Column("confidence_margin", sa.Float, nullable=False),
        sa.Column("model_version", sa.String(128), nullable=True),
        sa.Column("labeled", sa.Integer, nullable=False, server_default="0"),
        sa.Column("true_label", sa.String(64), nullable=True),
        if_not_exists=True,
    )


def downgrade():
    op.drop_table("review_items")
    op.drop_table("feedback")
