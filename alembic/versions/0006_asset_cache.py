from alembic import op
import sqlalchemy as sa


revision = '0006_asset_cache'
down_revision = '0005_artifact_local_fields'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'asset_cache',
        sa.Column('hash', sa.String(length=64), primary_key=True),
        sa.Column('type', sa.String(length=16), nullable=False),
        sa.Column('path', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('asset_cache')

