from alembic import op
import sqlalchemy as sa


revision = '0002_add_result_url'
down_revision = '0001_init_jobs'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('jobs', sa.Column('result_url', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('jobs', 'result_url')

