from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class RedIssueType(StrEnum):
    FACTUALITY = "factuality"
    COVERAGE = "coverage"
    REASONING_DEPTH = "reasoning_depth"
    CITATION_QUALITY = "citation_quality"
    EVIDENCE_MISMATCH = "evidence_mismatch"
    CONTRADICTION = "contradiction"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    REPORT_STRUCTURE = "report_structure"
    CLARITY = "clarity"
    CONFLICT_HANDLING = "conflict_handling"


class RedIssueSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RepairActionType(StrEnum):
    ADD = "ADD"
    DELETE = "DELETE"
    MODIFY = "MODIFY"
    VERIFY = "VERIFY"
    NOOP = "NOOP"


class RedIssue(BaseModel):
    id: str
    issue_type: RedIssueType
    severity: RedIssueSeverity
    location: str
    description: str
    evidence_ids: list[str] = Field(default_factory=list)
    suggested_action: RepairActionType
    rationale: str


class RedCritique(BaseModel):
    round_id: int
    issues: list[RedIssue] = Field(default_factory=list)
    high_severity_count: int = 0
    medium_severity_count: int = 0
    low_severity_count: int = 0
    summary: str


class RepairPatch(BaseModel):
    id: str
    action_type: RepairActionType
    target_location: str
    before_text: str | None = None
    after_text: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str
    applied: bool = False


class BlueRepairResult(BaseModel):
    round_id: int
    revised_report: str
    patches: list[RepairPatch] = Field(default_factory=list)
    summary: str


class RedBlueRoundTrace(BaseModel):
    round_id: int
    red_issue_count: int
    high_severity_count: int
    patches_applied_count: int
    judge_score_before: int | None = None
    judge_score_after: int | None = None
    score_delta: int | None = None
    stopped_reason: str | None = None
    red_parse_success: bool = False
    red_fallback_used: bool = False
    blue_parse_success: bool = False
    blue_fallback_used: bool = False


class RedBlueStats(BaseModel):
    enabled: bool
    max_rounds: int = 0
    rounds_completed: int = 0
    initial_overall_score: int | None = None
    final_overall_score: int | None = None
    score_delta: int | None = None
    total_red_issues: int = 0
    total_patches_applied: int = 0
    unresolved_high_severity_count: int = 0
    stopped_reason: str = "disabled"
    rounds: list[RedBlueRoundTrace] = Field(default_factory=list)
    final_report_selected: str = "initial"
