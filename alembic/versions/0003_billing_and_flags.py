from alembic import op
import sqlalchemy as sa


revision = '0003_billing_and_flags'
down_revision = '0002_add_result_url'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Jobs extra fields
    op.add_column('jobs', sa.Column('user_id', sa.String(length=128), nullable=True))
    op.add_column('jobs', sa.Column('plan', sa.String(length=32), nullable=True))
    op.add_column('jobs', sa.Column('quality', sa.String(length=32), nullable=True))
    op.add_column('jobs', sa.Column('cost_estimate', sa.Float(), nullable=True))

    # Plans
    op.create_table(
        'plans',
        sa.Column('name', sa.String(length=32), primary_key=True),
        sa.Column('default_backend', sa.String(length=16), nullable=False, server_default='local'),
        sa.Column('max_res_long', sa.String(length=16), nullable=False, server_default='1344'),
    )

    # Users
    op.create_table(
        'users',
        sa.Column('id', sa.String(length=128), primary_key=True),
        sa.Column('email', sa.String(length=255), nullable=True),
        sa.Column('plan', sa.String(length=32), nullable=False, server_default='free'),
    )

    # Usage
    op.create_table(
        'usage',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('user_id', sa.String(length=128), nullable=False),
        sa.Column('job_id', sa.String(length=64), nullable=False),
        sa.Column('units', sa.Float(), nullable=False, server_default='1.0'),
        sa.Column('cost', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )

    # Model registry
    op.create_table(
        'model_artifacts',
        sa.Column('name', sa.String(length=128), primary_key=True),
        sa.Column('version', sa.String(length=32), nullable=False),
        sa.Column('sha256', sa.String(length=64), nullable=False),
        sa.Column('s3_path', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )

    # Feature flags
    op.create_table(
        'feature_flags',
        sa.Column('key', sa.String(length=64), primary_key=True),
        sa.Column('value', sa.String(length=256), nullable=False, server_default=''),
        sa.Column('description', sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('feature_flags')
    op.drop_table('model_artifacts')
    op.drop_table('usage')
    op.drop_table('users')
    op.drop_table('plans')
    op.drop_column('jobs', 'cost_estimate')
    op.drop_column('jobs', 'quality')
    op.drop_column('jobs', 'plan')
    op.drop_column('jobs', 'user_id')

