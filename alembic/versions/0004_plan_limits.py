from alembic import op
import sqlalchemy as sa


revision = '0004_plan_limits'
down_revision = '0003_billing_and_flags'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('plans', sa.Column('monthly_limit', sa.Integer(), nullable=True))
    op.add_column('plans', sa.Column('per_image_cost', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('plans', 'per_image_cost')
    op.drop_column('plans', 'monthly_limit')

