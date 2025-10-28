from pydantic import BaseModel


class JobCreateResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    error: str | None = None
    result_path: str | None = None
    result_url: str | None = None


class JobCreateFromUrlsRequest(BaseModel):
    user_image_url: str
    garment_front_url: str
    garment_side_url: str | None = None
