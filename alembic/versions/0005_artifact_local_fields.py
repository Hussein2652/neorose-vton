from alembic import op
import sqlalchemy as sa


revision = '0005_artifact_local_fields'
down_revision = '0004_plan_limits'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('model_artifacts', sa.Column('local_path', sa.Text(), nullable=True))
    op.add_column('model_artifacts', sa.Column('size_bytes', sa.Integer(), nullable=True))
    op.add_column('model_artifacts', sa.Column('downloaded_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column('model_artifacts', 'downloaded_at')
    op.drop_column('model_artifacts', 'size_bytes')
    op.drop_column('model_artifacts', 'local_path')

