"""add admin users and manager chat

Revision ID: 001_add_admin
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_add_admin'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create admin_users table
    op.create_table('admin_users',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('username', sa.String(length=100), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('role', sa.String(length=20), server_default='manager', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_admin_users_username', 'admin_users', ['username'], unique=True)

    # Create manager_chats table
    op.create_table('manager_chats',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('session_id', sa.String(length=100), nullable=False),
        sa.Column('manager_id', sa.BigInteger(), nullable=True),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.session_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['manager_id'], ['admin_users.id'], ondelete='SET NULL'),
    )
    op.create_index('ix_manager_chats_session_id', 'manager_chats', ['session_id'])


def downgrade() -> None:
    op.drop_index('ix_manager_chats_session_id', table_name='manager_chats')
    op.drop_table('manager_chats')
    op.drop_index('ix_admin_users_username', table_name='admin_users')
    op.drop_table('admin_users')
