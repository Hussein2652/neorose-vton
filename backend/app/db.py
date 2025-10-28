from __future__ import annotations

import os
import datetime as dt
from typing import Optional

from sqlalchemy import (
    create_engine,
    String,
    Text,
    DateTime,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session


DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///storage/vfr.sqlite3")


class Base(DeclarativeBase):
    pass


class JobORM(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_image_path: Mapped[str] = mapped_column(Text)
    garment_front_path: Mapped[str] = mapped_column(Text)
    garment_side_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    provider: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow
    )
    task_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


engine = create_engine(DATABASE_URL, echo=False, future=True)


def init_db() -> None:
    # Ensure storage dir exists for SQLite
    if DATABASE_URL.startswith("sqlite"):
        os.makedirs("storage", exist_ok=True)
    Base.metadata.create_all(engine)


def create_job(
    job_id: str,
    user_image_path: str,
    garment_front_path: str,
    garment_side_path: Optional[str] = None,
    provider: Optional[str] = None,
    task_id: Optional[str] = None,
) -> None:
    with Session(engine) as s:
        s.add(
            JobORM(
                id=job_id,
                status="queued",
                user_image_path=user_image_path,
                garment_front_path=garment_front_path,
                garment_side_path=garment_side_path,
                provider=provider,
                task_id=task_id,
            )
        )
        s.commit()


def update_job(
    job_id: str,
    *,
    status: Optional[str] = None,
    error: Optional[str] = None,
    result_path: Optional[str] = None,
    task_id: Optional[str] = None,
) -> None:
    with Session(engine) as s:
        job = s.get(JobORM, job_id)
        if not job:
            return
        if status is not None:
            job.status = status
        if error is not None:
            job.error = error
        if result_path is not None:
            job.result_path = result_path
        if task_id is not None:
            job.task_id = task_id
        s.add(job)
        s.commit()


def get_job(job_id: str) -> Optional[JobORM]:
    with Session(engine) as s:
        job = s.get(JobORM, job_id)
        if job is None:
            return None
        # Detach for safe return
        s.expunge(job)
        return job

