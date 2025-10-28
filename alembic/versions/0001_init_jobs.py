from alembic import op
import sqlalchemy as sa


revision = '0001_init_jobs'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'jobs',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='queued'),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('user_image_path', sa.Text(), nullable=False),
        sa.Column('garment_front_path', sa.Text(), nullable=False),
        sa.Column('garment_side_path', sa.Text(), nullable=True),
        sa.Column('result_path', sa.Text(), nullable=True),
        sa.Column('provider', sa.String(length=32), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('task_id', sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('jobs')

