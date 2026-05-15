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


class ConflictType(StrEnum):
    DUPLICATE = "duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    ANTONYM_CONTRADICTION = "antonym_contradiction"
    NUMERIC_DIRECTION_CONFLICT = "numeric_direction_conflict"
    SEMANTIC_OPPOSITION = "semantic_opposition"
    SOURCE_QUALITY_CONFLICT = "source_quality_conflict"
    OFF_TOPIC = "off_topic"


class ConflictSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ResolutionAction(StrEnum):
    KEEP_BOTH = "keep_both"
    KEEP_LEFT = "keep_left"
    KEEP_RIGHT = "keep_right"
    DROP_LEFT = "drop_left"
    DROP_RIGHT = "drop_right"
    DOWNWEIGHT_LEFT = "downweight_left"
    DOWNWEIGHT_RIGHT = "downweight_right"
    MARK_FOR_WRITER = "mark_for_writer"


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


class CompressedEvidence(BaseModel):
    evidence: Evidence
    l1_similarity_score: float = 0.0
    l2_textrank_score: float = 0.0
    confidence_score: float = 0.0
    source_quality_score: float = 0.0
    final_score: float = 0.0
    compression_level: str = "L3_ORIGINAL"


class CompressionStats(BaseModel):
    enabled: bool
    l1_input_count: int = 0
    l1_selected_count: int = 0
    l2_input_count: int = 0
    l2_selected_count: int = 0
    final_selected_count: int = 0
    original_token_estimate: int = 0
    compressed_token_estimate: int = 0
    compression_ratio: float = 1.0
    selected_evidence_ids: list[str] = Field(default_factory=list)
    dropped_evidence_ids: list[str] = Field(default_factory=list)
    strategy: str = "L1_embedding_to_L2_textrank_to_L3_original"


class EvidenceConflict(BaseModel):
    id: str
    left_evidence_id: str
    right_evidence_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    reason: str
    left_score: float = 0.0
    right_score: float = 0.0
    resolution_action: ResolutionAction | None = None
    resolved: bool = False


class ConflictStats(BaseModel):
    enabled: bool
    input_evidence_count: int = 0
    output_evidence_count: int = 0
    conflict_count: int = 0
    duplicate_count: int = 0
    near_duplicate_count: int = 0
    antonym_contradiction_count: int = 0
    numeric_direction_conflict_count: int = 0
    semantic_opposition_count: int = 0
    source_quality_conflict_count: int = 0
    off_topic_count: int = 0
    dropped_evidence_ids: list[str] = Field(default_factory=list)
    downweighted_evidence_ids: list[str] = Field(default_factory=list)
    marked_conflict_evidence_ids: list[str] = Field(default_factory=list)
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
    strategy: str = "heuristic_antonym_numeric_semantic_resolution"


from deepresearch_agent.red_blue.schemas import (  # noqa: E402
    BlueRepairResult,
    RedBlueRoundTrace,
    RedBlueStats,
    RedCritique,
    RedIssue,
    RedIssueSeverity,
    RedIssueType,
    RepairActionType,
    RepairPatch,
)
from deepresearch_agent.search.schemas import (  # noqa: E402
    CitationValidationResult,
    FetchedDocument,
    SearchQuery,
    SearchResult,
    SearchStats,
)
