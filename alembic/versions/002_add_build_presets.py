"""Add build_presets table for Smart Presets system.

Revision ID: 002_add_build_presets
Revises:
Create Date: 2025-01-13

NOTE: This migration has no down_revision because it may be applied after
various merge states. It's designed to be idempotent - checks if tables exist.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '002_add_build_presets'
down_revision: Union[str, None] = None  # Will be merged manually
branch_labels: Union[str, Sequence[str], None] = ('presets',)
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create build_presets table for expert PC configurations."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()

    # Skip if table already exists
    if 'build_presets' in tables:
        return

    op.create_table(
        'build_presets',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('target_budget', sa.Integer(), nullable=False),
        sa.Column('category_tag', sa.String(length=50), nullable=False),
        sa.Column('purpose', sa.String(length=50), server_default='gaming', nullable=False),
        sa.Column('components', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('priority', sa.Integer(), server_default='0', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes
    op.create_index('ix_build_presets_target_budget', 'build_presets', ['target_budget'])
    op.create_index('ix_build_presets_category_tag', 'build_presets', ['category_tag'])
    op.create_index('ix_build_presets_purpose', 'build_presets', ['purpose'])
    op.create_index('idx_build_presets_budget_category', 'build_presets', ['target_budget', 'category_tag'])

    # Also create sync_status table if it doesn't exist
    if 'sync_status' not in tables:
        op.create_table(
            'sync_status',
            sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column('sync_type', sa.String(length=50), nullable=False),
            sa.Column('last_sync_at', sa.DateTime(timezone=True), nullable=True),
            sa.Column('last_success_at', sa.DateTime(timezone=True), nullable=True),
            sa.Column('status', sa.String(length=20), server_default='idle', nullable=False),
            sa.Column('products_created', sa.Integer(), server_default='0', nullable=False),
            sa.Column('products_updated', sa.Integer(), server_default='0', nullable=False),
            sa.Column('products_zeroed', sa.Integer(), server_default='0', nullable=False),
            sa.Column('errors', sa.Integer(), server_default='0', nullable=False),
            sa.Column('error_message', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index('ix_sync_status_sync_type', 'sync_status', ['sync_type'], unique=True)


def downgrade() -> None:
    """Drop build_presets and sync_status tables."""
    op.drop_index('idx_build_presets_budget_category', table_name='build_presets')
    op.drop_index('ix_build_presets_purpose', table_name='build_presets')
    op.drop_index('ix_build_presets_category_tag', table_name='build_presets')
    op.drop_index('ix_build_presets_target_budget', table_name='build_presets')
    op.drop_table('build_presets')

    # Also drop sync_status if exists
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if 'sync_status' in inspector.get_table_names():
        op.drop_index('ix_sync_status_sync_type', table_name='sync_status')
        op.drop_table('sync_status')
