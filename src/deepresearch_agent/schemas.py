from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskState(StrEnum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    DEGRADED = "DEGRADED"
    REPLANNED = "REPLANNED"
    CANCELLED = "CANCELLED"


class TaskNode(BaseModel):
    id: str
    name: str
    description: str
    agent_type: str
    dependencies: list[str] = Field(default_factory=list)
    state: TaskState = TaskState.PENDING
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    timeout_seconds: int = 60
    retry_count: int = 0
    max_retries: int = 1


class Evidence(BaseModel):
    id: str
    task_id: str
    title: str
    content: str
    source_url: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgeScore(BaseModel):
    factuality: int = Field(ge=0, le=100)
    coverage: int = Field(ge=0, le=100)
    reasoning_depth: int = Field(ge=0, le=100)
    citation_quality: int = Field(ge=0, le=100)
    clarity: int = Field(ge=0, le=100)
    overall: int = Field(ge=0, le=100)
    comments: str


class ResearchReport(BaseModel):
    question: str
    markdown: str
    evidences: list[Evidence]
    score: JudgeScore | None = None
