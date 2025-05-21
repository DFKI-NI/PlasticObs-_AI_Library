"""Main module."""

import uuid
from typing import List
from contextlib import asynccontextmanager

import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse

from ai_inference.inference.routes import router as model_routes
from ai_inference.job import Job, TaskStatus


class JobsResponse(BaseModel):
    job_ids: List[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run at startup
    Initialise the Client and add it to request.state
    """
    app.state.jobs_queue = {}
    yield

    # TODO cancel running predictions and mark jobs as cancelled
    # TODO persist job_queue


app = FastAPI(lifespan=lifespan)
app.include_router(model_routes)


def add_new_job_to_queue(request: Request, status: TaskStatus = TaskStatus.RUNNING, message: str = ""):
    job = Job(uuid.uuid4(), status, message)
    request.app.state.jobs_queue[job.job_id] = job
    return job


@app.get('/')
def index(request: Request):
    return RedirectResponse("/docs")


@app.get("/jobs/")
def jobs(request: Request) -> JobsResponse:
    return {"job_ids": (v.job_id for k, v in request.app.state.jobs_queue.items())}


@app.get("/jobs/{job_id}")
def job_status(job_id: str, request: Request):
    queue = request.app.state.jobs_queue
    for _id in queue:
        if _id == job_id:
            return queue[_id]
    else:
        raise HTTPException(status_code=404, detail=f"No job found: {job_id}")


def main():
    uvicorn.run(app, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
