from __future__ import annotations

import json
from typing import Any

from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.red_blue.actions import append_cautionary_note, ensure_report_sections
from deepresearch_agent.red_blue.schemas import (
    BlueRepairResult,
    RedCritique,
    RepairActionType,
    RepairPatch,
)
from deepresearch_agent.schemas import Evidence
from deepresearch_agent.utils.json_utils import safe_json_loads, strip_thinking


class BlueAgent:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm
        self.last_stats: dict[str, Any] = {
            "blue_parse_success": False,
            "blue_fallback_used": False,
            "error": None,
        }

    async def repair(
        self,
        question: str,
        report: str,
        evidences: list[Evidence],
        critique: RedCritique,
        round_id: int = 1,
    ) -> BlueRepairResult:
        prompt = _build_blue_prompt(question, report, evidences, critique, round_id)
        raw = await self.llm.agenerate(prompt, prompt_type="blue")
        try:
            data = safe_json_loads(strip_thinking(raw))
            result = _repair_from_data(data, round_id, report)
            self.last_stats = {
                "blue_parse_success": True,
                "blue_fallback_used": False,
                "error": None,
            }
            return result
        except (KeyError, TypeError, ValueError) as exc:
            result = _fallback_repair(report, critique, round_id)
            self.last_stats = {
                "blue_parse_success": False,
                "blue_fallback_used": True,
                "error": str(exc)[:200],
            }
            return result


def _build_blue_prompt(
    question: str,
    report: str,
    evidences: list[Evidence],
    critique: RedCritique,
    round_id: int,
) -> str:
    evidence_lines = "\n".join(
        f"- {evidence.id}: {evidence.title} | {evidence.content}"
        for evidence in evidences[:12]
    )
    return (
        "You are BlueAgent, a blue-team defender that repairs a research report based on RedAgent critique.\n"
        "Repair the report using only the available evidence. Do not invent new citations.\n"
        "If evidence is insufficient, use cautious language such as 'The available evidence suggests' or "
        "'This point requires further validation'.\n"
        "Return JSON only. Do not include markdown fences. Do not include <think> blocks.\n"
        f"Round ID: {round_id}\n"
        f"Question: {question}\n"
        f"Evidence:\n{evidence_lines}\n"
        f"Red critique JSON: {critique.model_dump_json()}\n"
        f"Current report:\n{report}\n"
        "Use this schema exactly:\n"
        "{\n"
        '  "revised_report": "complete Markdown report",\n'
        '  "patches": [\n'
        "    {\n"
        '      "id": "patch_1",\n'
        '      "action_type": "MODIFY",\n'
        '      "target_location": "Conclusion",\n'
        '      "before_text": "string or null",\n'
        '      "after_text": "string or null",\n'
        '      "evidence_ids": ["evidence_id"],\n'
        '      "rationale": "string",\n'
        '      "applied": true\n'
        "    }\n"
        "  ],\n"
        '  "summary": "string"\n'
        "}"
    )


def _repair_from_data(data: dict[str, Any], round_id: int, fallback_report: str) -> BlueRepairResult:
    revised_report = str(data.get("revised_report") or fallback_report)
    patches = [
        RepairPatch(
            id=str(item.get("id") or f"patch_{index}"),
            action_type=_action(item.get("action_type")),
            target_location=str(item.get("target_location") or "report"),
            before_text=_optional_str(item.get("before_text")),
            after_text=_optional_str(item.get("after_text")),
            evidence_ids=[str(evidence_id) for evidence_id in item.get("evidence_ids", []) if evidence_id],
            rationale=str(item.get("rationale") or ""),
            applied=bool(item.get("applied", False)),
        )
        for index, item in enumerate(data.get("patches", []), start=1)
        if isinstance(item, dict)
    ]
    return BlueRepairResult(
        round_id=round_id,
        revised_report=ensure_report_sections(revised_report),
        patches=patches,
        summary=str(data.get("summary") or "Blue repair completed."),
    )


def _fallback_repair(report: str, critique: RedCritique, round_id: int) -> BlueRepairResult:
    revised_report = ensure_report_sections(append_cautionary_note(report, critique.issues))
    patch = RepairPatch(
        id=f"patch_{round_id}_fallback_caution",
        action_type=RepairActionType.VERIFY,
        target_location="Cautionary Note",
        before_text=None,
        after_text="Cautionary Note added for claims that need further validation.",
        evidence_ids=[evidence_id for issue in critique.issues for evidence_id in issue.evidence_ids][:5],
        rationale="Fallback repair added cautious language because structured BlueAgent output could not be parsed.",
        applied=True,
    )
    return BlueRepairResult(
        round_id=round_id,
        revised_report=revised_report,
        patches=[patch],
        summary="Fallback repair appended a cautionary note.",
    )


def _action(value: object) -> RepairActionType:
    try:
        return RepairActionType(str(value).upper())
    except ValueError:
        return RepairActionType.NOOP


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
