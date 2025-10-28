from __future__ import annotations

import strawberry
from typing import Optional
from strawberry.asgi import GraphQL

from .db import get_job, create_job
from .tasks import run_tryon_task
from .storage import Storage


@strawberry.type
class Job:
    job_id: str
    status: str
    error: Optional[str]
    result_path: Optional[str]
    result_url: Optional[str]


@strawberry.type
class Query:
    health: str = strawberry.field(resolver=lambda: "ok")

    @strawberry.field
    def job(self, job_id: str) -> Optional[Job]:
        j = get_job(job_id)
        if not j:
            return None
        return Job(
            job_id=j.id,
            status=j.status,
            error=j.error,
            result_path=j.result_path,
            result_url=j.result_url,
        )


@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_tryon_job_from_paths(
        self,
        user_image_path: str,
        garment_front_path: str,
        garment_side_path: Optional[str] = None,
    ) -> str:
        # Enqueue via Celery if configured else ignore (GraphQL for demo only)
        import os, uuid
        job_id = str(uuid.uuid4())
        use_celery = os.environ.get("USE_CELERY", "0") == "1"
        if use_celery:
            async_result = run_tryon_task.delay(
                job_id=job_id,
                user_image_path=user_image_path,
                garment_front_path=garment_front_path,
                garment_side_path=garment_side_path,
            )
            create_job(job_id, user_image_path, garment_front_path, garment_side_path, provider=os.environ.get("FINISHER_BACKEND", "local"), task_id=async_result.id)
        else:
            create_job(job_id, user_image_path, garment_front_path, garment_side_path, provider=os.environ.get("FINISHER_BACKEND", "local"))
            from .queue import Jobs
            from .pipeline_runner import run_tryon_job
            Jobs.enqueue(
                run_tryon_job,
                queue_job_id=job_id,
                user_image_path=user_image_path,
                garment_front_path=garment_front_path,
                garment_side_path=garment_side_path,
                job_id=job_id,
            )
        return job_id


schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_app = GraphQL(schema, graphiql=True)
