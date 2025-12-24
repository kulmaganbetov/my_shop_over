"""Add status and escalation_reason to chat_sessions.

Revision ID: 001_add_session_status
Revises:
Create Date: 2024-12-24

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_add_session_status'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add status and escalation_reason columns to chat_sessions table."""
    # Add status column with default 'bot'
    op.add_column(
        'chat_sessions',
        sa.Column('status', sa.String(20), nullable=False, server_default='bot')
    )

    # Add escalation_reason column (nullable)
    op.add_column(
        'chat_sessions',
        sa.Column('escalation_reason', sa.Text(), nullable=True)
    )


def downgrade() -> None:
    """Remove status and escalation_reason columns from chat_sessions table."""
    op.drop_column('chat_sessions', 'escalation_reason')
    op.drop_column('chat_sessions', 'status')
