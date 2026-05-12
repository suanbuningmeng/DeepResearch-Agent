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


class ReplanEvent(BaseModel):
    id: str
    reason: str
    failed_task_ids: list[str]
    new_task_ids: list[str]
    created_at: str


class DegradationRecord(BaseModel):
    task_id: str
    reason: str
    original_state: TaskState
    fallback_evidence_id: str | None = None


class SchedulerConfig(BaseModel):
    max_concurrency: int = 3
    task_timeout_seconds: int = 30
    global_timeout_seconds: int = 120
    max_replan_rounds: int = 2
    batch_failure_threshold: int = 2
    min_total_evidences: int = 4
    enable_replan: bool = True


class MemoryStats(BaseModel):
    backend: str
    db_path: str | None = None
    vector_index_path: str | None = None
    inserted_evidence_count: int = 0
    duplicate_evidence_count: int = 0
    total_evidence_count: int = 0
    retrieved_evidence_count: int = 0
    memory_search_top_k: int = 10
    source_quality_summary: dict[str, int] = Field(default_factory=dict)
