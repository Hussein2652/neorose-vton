from alembic import op
import sqlalchemy as sa


revision = '0007_size_bytes_bigint'
down_revision = '0006_asset_cache'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Alter size_bytes to BIGINT to store large artifact sizes
    with op.batch_alter_table('model_artifacts') as batch_op:
        batch_op.alter_column(
            'size_bytes',
            existing_type=sa.Integer(),
            type_=sa.BigInteger(),
            existing_nullable=True,
        )


def downgrade() -> None:
    with op.batch_alter_table('model_artifacts') as batch_op:
        batch_op.alter_column(
            'size_bytes',
            existing_type=sa.BigInteger(),
            type_=sa.Integer(),
            existing_nullable=True,
        )

