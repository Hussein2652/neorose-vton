from __future__ import annotations

import os
import datetime as dt
from typing import Optional

from sqlalchemy import (
    create_engine,
    String,
    Text,
    DateTime,
    Float,
    Integer,
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
    result_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    provider: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    plan: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    quality: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    cost_estimate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
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
    user_id: Optional[str] = None,
    plan: Optional[str] = None,
    quality: Optional[str] = None,
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
                user_id=user_id,
                plan=plan,
                quality=quality,
            )
        )
        s.commit()


def update_job(
    job_id: str,
    *,
    status: Optional[str] = None,
    error: Optional[str] = None,
    result_path: Optional[str] = None,
    result_url: Optional[str] = None,
    task_id: Optional[str] = None,
    cost_estimate: Optional[float] = None,
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
        if result_url is not None:
            job.result_url = result_url
        if task_id is not None:
            job.task_id = task_id
        if cost_estimate is not None:
            job.cost_estimate = cost_estimate
        s.add(job)
        s.commit()


class PlanORM(Base):
    __tablename__ = "plans"
    name: Mapped[str] = mapped_column(String(32), primary_key=True)
    # simple attributes for demo: backend default and max resolution
    default_backend: Mapped[str] = mapped_column(String(16), default="local")
    max_res_long: Mapped[int] = mapped_column(
        String(16), default="1344"
    )  # keep as string for simplicity
    monthly_limit: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    per_image_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class UserORM(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    plan: Mapped[str] = mapped_column(String(32), default="free")


class UsageORM(Base):
    __tablename__ = "usage"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128))
    job_id: Mapped[str] = mapped_column(String(64))
    units: Mapped[float] = mapped_column(Float, default=1.0)
    cost: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class ModelArtifactORM(Base):
    __tablename__ = "model_artifacts"
    name: Mapped[str] = mapped_column(String(128), primary_key=True)
    version: Mapped[str] = mapped_column(String(32))
    sha256: Mapped[str] = mapped_column(String(64))
    s3_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    local_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    downloaded_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)


class FeatureFlagORM(Base):
    __tablename__ = "feature_flags"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(256), default="")
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


def set_feature_flag(key: str, value: str, description: Optional[str] = None) -> None:
    with Session(engine) as s:
        flag = s.get(FeatureFlagORM, key)
        if not flag:
            flag = FeatureFlagORM(key=key, value=value, description=description)
        else:
            flag.value = value
            if description is not None:
                flag.description = description
        s.add(flag)
        s.commit()


def get_feature_flag(key: str, default: Optional[str] = None) -> Optional[str]:
    with Session(engine) as s:
        flag = s.get(FeatureFlagORM, key)
        return flag.value if flag else default


def list_jobs(limit: int = 50) -> list[JobORM]:
    with Session(engine) as s:
        stmt = select(JobORM).order_by(JobORM.created_at.desc()).limit(limit)
        rows = list(s.scalars(stmt))
        for r in rows:
            s.expunge(r)
        return rows


def record_usage(user_id: Optional[str], job_id: str, units: float, cost: float) -> None:
    if not user_id:
        return
    import uuid
    with Session(engine) as s:
        s.add(UsageORM(id=str(uuid.uuid4())[:12], user_id=user_id, job_id=job_id, units=units, cost=cost))
        s.commit()


def list_artifacts() -> list[ModelArtifactORM]:
    with Session(engine) as s:
        rows = list(s.query(ModelArtifactORM).all())  # type: ignore[attr-defined]
        for r in rows:
            s.expunge(r)
        return rows


def set_artifact_local(name: str, version: str, local_path: str, size_bytes: int) -> None:
    with Session(engine) as s:
        obj = s.get(ModelArtifactORM, name)
        if not obj:
            # minimal record
            obj = ModelArtifactORM(name=name, version=version, sha256="", s3_path=None)
        obj.local_path = local_path
        obj.size_bytes = size_bytes
        obj.downloaded_at = dt.datetime.utcnow()
        s.add(obj)
        s.commit()


def list_plans() -> list[PlanORM]:
    with Session(engine) as s:
        rows = list(s.query(PlanORM).all())  # type: ignore[attr-defined]
        for r in rows:
            s.expunge(r)
        return rows


def upsert_plan(name: str, default_backend: str | None = None, max_res_long: str | None = None, monthly_limit: int | None = None, per_image_cost: float | None = None) -> None:
    with Session(engine) as s:
        obj = s.get(PlanORM, name)
        if not obj:
            obj = PlanORM(name=name)
        if default_backend is not None:
            obj.default_backend = default_backend
        if max_res_long is not None:
            obj.max_res_long = max_res_long
        if monthly_limit is not None:
            obj.monthly_limit = monthly_limit
        if per_image_cost is not None:
            obj.per_image_cost = per_image_cost
        s.add(obj)
        s.commit()


def set_user_plan(user_id: str, plan: str) -> None:
    with Session(engine) as s:
        u = s.get(UserORM, user_id)
        if not u:
            u = UserORM(id=user_id, plan=plan)
        else:
            u.plan = plan
        s.add(u)
        s.commit()


def get_usage_summary(user_id: str) -> dict:
    import datetime as dt
    with Session(engine) as s:
        now = dt.datetime.utcnow()
        month_start = dt.datetime(now.year, now.month, 1)
        q = s.query(UsageORM).filter(UsageORM.user_id == user_id)  # type: ignore[attr-defined]
        q_month = s.query(UsageORM).filter(UsageORM.user_id == user_id, UsageORM.created_at >= month_start)  # type: ignore[attr-defined]
        def agg(query):
            units = 0.0; cost = 0.0
            for row in query:
                units += float(row.units); cost += float(row.cost)
            return units, cost
        total_units, total_cost = agg(q)
        m_units, m_cost = agg(q_month)
        return {"total_units": total_units, "total_cost": total_cost, "month_units": m_units, "month_cost": m_cost}


def get_job(job_id: str) -> Optional[JobORM]:
    with Session(engine) as s:
        job = s.get(JobORM, job_id)
        if job is None:
            return None
        # Detach for safe return
        s.expunge(job)
        return job


def ensure_user(user_id: str, email: Optional[str] = None, default_plan: str = "free") -> UserORM:
    with Session(engine) as s:
        user = s.get(UserORM, user_id)
        if not user:
            user = UserORM(id=user_id, email=email, plan=default_plan)
            s.add(user)
            s.commit()
        s.expunge(user)
        return user


def get_plan(name: str) -> Optional[PlanORM]:
    with Session(engine) as s:
        p = s.get(PlanORM, name)
        if not p:
            return None
        s.expunge(p)
        return p


def get_month_usage_units(user_id: str) -> float:
    import datetime as dt
    with Session(engine) as s:
        now = dt.datetime.utcnow()
        month_start = dt.datetime(now.year, now.month, 1)
        q = s.query(UsageORM).filter(UsageORM.user_id == user_id, UsageORM.created_at >= month_start)  # type: ignore[attr-defined]
        total = 0.0
        for row in q:
            total += float(row.units)
        return total
