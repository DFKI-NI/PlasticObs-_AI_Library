from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel


class TaskStatus(Enum):
    RUNNING = "running"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COMPLETE = "complete"


class Job:
    def __init__(self, job_id: UUID | None, status: TaskStatus = TaskStatus.RUNNING, msg: str = ""):
        self.job_id = str(job_id) if job_id else str(uuid4())
        self.status = status
        self.msg = msg

    def __str__(self):
        return f"Job<{self.job_id}>"


class JobStatus(BaseModel):
    job_id: str
    status: TaskStatus
    msg: Optional[str]
