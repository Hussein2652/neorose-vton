from pydantic import BaseModel


class JobCreateResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    error: str | None = None
    result_path: str | None = None
    result_url: str | None = None
