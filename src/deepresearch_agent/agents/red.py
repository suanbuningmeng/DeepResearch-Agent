from __future__ import annotations

import json
from typing import Any

from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.memory.source_quality import SourceQuality, classify_source_url
from deepresearch_agent.red_blue.schemas import (
    RedCritique,
    RedIssue,
    RedIssueSeverity,
    RedIssueType,
    RepairActionType,
)
from deepresearch_agent.red_blue.utils import recount_critique
from deepresearch_agent.schemas import Evidence
from deepresearch_agent.utils.json_utils import safe_json_loads, strip_thinking


class RedAgent:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm
        self.last_stats: dict[str, Any] = {
            "red_parse_success": False,
            "red_fallback_used": False,
            "error": None,
        }

    async def critique(
        self,
        question: str,
        report: str,
        evidences: list[Evidence],
        judge_score: dict | None = None,
        memory_stats: dict | None = None,
        compression_stats: dict | None = None,
        conflict_stats: dict | None = None,
        round_id: int = 1,
    ) -> RedCritique:
        prompt = _build_red_prompt(
            question=question,
            report=report,
            evidences=evidences,
            judge_score=judge_score,
            memory_stats=memory_stats,
            compression_stats=compression_stats,
            conflict_stats=conflict_stats,
            round_id=round_id,
        )
        raw = await self.llm.agenerate(prompt, prompt_type="red")
        try:
            data = safe_json_loads(strip_thinking(raw))
            critique = _critique_from_data(data, round_id)
            self.last_stats = {
                "red_parse_success": True,
                "red_fallback_used": False,
                "error": None,
            }
            return critique
        except (KeyError, TypeError, ValueError) as exc:
            critique = _fallback_critique(
                question=question,
                report=report,
                evidences=evidences,
                judge_score=judge_score or {},
                conflict_stats=conflict_stats or {},
                round_id=round_id,
            )
            self.last_stats = {
                "red_parse_success": False,
                "red_fallback_used": True,
                "error": str(exc)[:200],
            }
            return critique


def _build_red_prompt(
    question: str,
    report: str,
    evidences: list[Evidence],
    judge_score: dict | None,
    memory_stats: dict | None,
    compression_stats: dict | None,
    conflict_stats: dict | None,
    round_id: int,
) -> str:
    evidence_lines = "\n".join(
        f"- {evidence.id}: {evidence.title} | {evidence.content} | source={evidence.source_url}"
        for evidence in evidences[:12]
    )
    return (
        "You are RedAgent, a strict red-team reviewer for a research report.\n"
        "Find problems only; do not rewrite the report.\n"
        "Return JSON only. Do not include markdown fences. Do not include <think> blocks.\n"
        "Review dimensions: factuality, coverage, reasoning_depth, citation_quality, evidence_mismatch, "
        "contradiction, unsupported_claim, report_structure, clarity, conflict_handling.\n"
        f"Round ID: {round_id}\n"
        f"Question: {question}\n"
        f"Judge score JSON: {json.dumps(judge_score or {}, ensure_ascii=False)}\n"
        f"Memory stats JSON: {json.dumps(memory_stats or {}, ensure_ascii=False)}\n"
        f"Compression stats JSON: {json.dumps(compression_stats or {}, ensure_ascii=False)}\n"
        f"Conflict stats JSON: {json.dumps(conflict_stats or {}, ensure_ascii=False)}\n"
        f"Evidence:\n{evidence_lines}\n"
        f"Report:\n{report}\n"
        "Use this schema exactly:\n"
        "{\n"
        '  "issues": [\n'
        "    {\n"
        '      "id": "issue_1",\n'
        '      "issue_type": "citation_quality",\n'
        '      "severity": "medium",\n'
        '      "location": "Evidence Summary",\n'
        '      "description": "string",\n'
        '      "evidence_ids": ["evidence_id"],\n'
        '      "suggested_action": "VERIFY",\n'
        '      "rationale": "string"\n'
        "    }\n"
        "  ],\n"
        '  "summary": "string"\n'
        "}"
    )


def _critique_from_data(data: dict[str, Any], round_id: int) -> RedCritique:
    issues = [
        RedIssue(
            id=str(item.get("id") or f"issue_{index}"),
            issue_type=_issue_type(item.get("issue_type")),
            severity=_severity(item.get("severity")),
            location=str(item.get("location") or "report"),
            description=str(item.get("description") or ""),
            evidence_ids=[str(evidence_id) for evidence_id in item.get("evidence_ids", []) if evidence_id],
            suggested_action=_action(item.get("suggested_action")),
            rationale=str(item.get("rationale") or ""),
        )
        for index, item in enumerate(data.get("issues", []), start=1)
        if isinstance(item, dict)
    ]
    critique = RedCritique(
        round_id=round_id,
        issues=issues,
        summary=str(data.get("summary") or ("No issues found." if not issues else "Issues found.")),
    )
    return recount_critique(critique)


def _fallback_critique(
    question: str,
    report: str,
    evidences: list[Evidence],
    judge_score: dict,
    conflict_stats: dict,
    round_id: int,
) -> RedCritique:
    del question
    issues: list[RedIssue] = []
    citation_quality = _score(judge_score.get("citation_quality"))
    coverage = _score(judge_score.get("coverage"))
    if citation_quality is not None and citation_quality < 80:
        issues.append(
            _issue(
                "issue_citation_quality",
                RedIssueType.CITATION_QUALITY,
                RedIssueSeverity.MEDIUM,
                "Evidence Summary",
                "Citation quality is below the target threshold.",
                [evidence.id for evidence in evidences[:3]],
                RepairActionType.VERIFY,
                "The judge score suggests citation support may be weak.",
            )
        )
    if coverage is not None and coverage < 80:
        issues.append(
            _issue(
                "issue_coverage",
                RedIssueType.COVERAGE,
                RedIssueSeverity.MEDIUM,
                "Key Findings",
                "Coverage is below the target threshold.",
                [evidence.id for evidence in evidences[:3]],
                RepairActionType.ADD,
                "The report may not cover enough of the available evidence.",
            )
        )
    if "example.com" in report.lower() or any(
        classify_source_url(evidence.source_url) == SourceQuality.EXAMPLE_URL for evidence in evidences
    ):
        issues.append(
            _issue(
                "issue_example_source",
                RedIssueType.CITATION_QUALITY,
                RedIssueSeverity.MEDIUM,
                "Citations",
                "The report or evidence includes example.com, which is not a verified source.",
                [],
                RepairActionType.VERIFY,
                "Example URLs should not be treated as validated citations.",
            )
        )
    if int(conflict_stats.get("conflict_count") or 0) > 0 and int(conflict_stats.get("marked_conflict_evidence_ids") and len(conflict_stats.get("marked_conflict_evidence_ids")) or 0) > 0:
        issues.append(
            _issue(
                "issue_conflict_handling",
                RedIssueType.CONFLICT_HANDLING,
                RedIssueSeverity.MEDIUM,
                "Evidence Summary",
                "Some evidence was marked as potentially conflicting and should be presented carefully.",
                list(conflict_stats.get("marked_conflict_evidence_ids") or []),
                RepairActionType.MODIFY,
                "Marked conflicts require cautious language.",
            )
        )
    critique = RedCritique(
        round_id=round_id,
        issues=issues[:3],
        summary="Fallback critique generated from judge scores and local trace signals.",
    )
    return recount_critique(critique)


def _issue(
    issue_id: str,
    issue_type: RedIssueType,
    severity: RedIssueSeverity,
    location: str,
    description: str,
    evidence_ids: list[str],
    action: RepairActionType,
    rationale: str,
) -> RedIssue:
    return RedIssue(
        id=issue_id,
        issue_type=issue_type,
        severity=severity,
        location=location,
        description=description,
        evidence_ids=evidence_ids,
        suggested_action=action,
        rationale=rationale,
    )


def _issue_type(value: object) -> RedIssueType:
    try:
        return RedIssueType(str(value))
    except ValueError:
        return RedIssueType.UNSUPPORTED_CLAIM


def _severity(value: object) -> RedIssueSeverity:
    try:
        return RedIssueSeverity(str(value).lower())
    except ValueError:
        return RedIssueSeverity.MEDIUM


def _action(value: object) -> RepairActionType:
    try:
        return RepairActionType(str(value).upper())
    except ValueError:
        return RepairActionType.VERIFY


def _score(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
