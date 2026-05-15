from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.schemas import Evidence, JudgeScore
from deepresearch_agent.utils.json_utils import safe_json_loads, strip_thinking
from deepresearch_agent.utils.structured_output import repair_json_with_llm, try_parse_json_lenient


SCORE_FIELDS = [
    "factuality",
    "coverage",
    "reasoning_depth",
    "citation_quality",
    "clarity",
    "overall",
]

JUDGE_SCHEMA_HINT = """
{
  "factuality": 0,
  "coverage": 0,
  "reasoning_depth": 0,
  "citation_quality": 0,
  "clarity": 0,
  "overall": 0,
  "comments": "string"
}
"""


class JudgeAgent:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm
        self.last_stats: dict[str, Any] = _default_judge_stats()

    async def judge(
        self,
        question: str,
        report: str,
        evidences: list[Evidence],
    ) -> JudgeScore:
        """Score a report against the question and evidence using the configured LLM."""
        prompt = (
            "You are a judge agent. Score this report from 0 to 100.\n"
            "Evaluate only the REPORT section against the QUESTION and EVIDENCE sections.\n"
            "Do not judge the prompt, schema, instructions, or any missing external context.\n"
            "If the REPORT section contains any non-whitespace text, do not say the report is empty or has no content.\n"
            "Use the EVIDENCE section only to judge support and citation quality.\n"
            "Return valid JSON only.\n"
            "Do not output markdown.\n"
            "Do not include markdown fences.\n"
            "Do not include explanations outside JSON.\n"
            "Do not include <think> blocks.\n"
            "Use this schema exactly:\n"
            "{\n"
            '  "factuality": 0,\n'
            '  "coverage": 0,\n'
            '  "reasoning_depth": 0,\n'
            '  "citation_quality": 0,\n'
            '  "clarity": 0,\n'
            '  "overall": 0,\n'
            '  "comments": "string"\n'
            "}\n"
            "The score fields must be integers from 0 to 100.\n"
            "comments must be a short string.\n"
            "Do not output any other fields.\n"
            "===== QUESTION START =====\n"
            f"{question}\n"
            "===== QUESTION END =====\n"
            "===== REPORT START =====\n"
            f"{report}\n"
            "===== REPORT END =====\n"
            "===== EVIDENCE START =====\n"
            f"Evidence count: {len(evidences)}\n"
            + "\n".join(
                f"- {evidence.id}: {evidence.title} | {evidence.content} | source_url={evidence.source_url}"
                for evidence in evidences[:20]
            )
            + "\n===== EVIDENCE END =====\n"
        )
        raw = await self.llm.agenerate(prompt, prompt_type="judge")
        cleaned = strip_thinking(raw)
        self.last_stats = _default_judge_stats()

        try:
            data = safe_json_loads(cleaned)
            normalized = coerce_judge_data(data)
            if normalized is None:
                raise ValueError("Judge JSON did not contain enough score fields.")
            self.last_stats = {
                "json_parse_success": True,
                "fallback_used": False,
                "extracted_from_text": False,
                "repair_attempted": False,
                "repair_success": False,
                "schema_coercion_success": False,
                "normalized": True,
                "error": None,
            }
            return JudgeScore.model_validate(normalized)
        except Exception as exc:
            parse_error = str(exc)
            _write_debug_judge_output(raw)

        lenient_data = try_parse_json_lenient(cleaned)
        if lenient_data is not None:
            coerced = coerce_judge_data(lenient_data)
            if coerced is not None:
                self.last_stats = {
                    "json_parse_success": False,
                    "fallback_used": False,
                    "extracted_from_text": False,
                    "repair_attempted": False,
                    "repair_success": False,
                    "schema_coercion_success": True,
                    "normalized": True,
                    "error": _short_error(parse_error),
                }
                return JudgeScore.model_validate(coerced)

        extracted = extract_scores_from_text(cleaned)
        if extracted is not None:
            self.last_stats = {
                "json_parse_success": False,
                "fallback_used": False,
                "extracted_from_text": True,
                "repair_attempted": False,
                "repair_success": False,
                "schema_coercion_success": False,
                "normalized": True,
                "error": _short_error(parse_error),
            }
            return JudgeScore.model_validate(extracted)

        repaired, repair_stats = await repair_json_with_llm(
            self.llm,
            raw_output=raw,
            schema_hint=JUDGE_SCHEMA_HINT,
            task_name="judge",
        )
        if repaired is not None:
            normalized = coerce_judge_data(repaired)
            if normalized is None:
                normalized = normalize_judge_score(repaired)
            self.last_stats = {
                "json_parse_success": False,
                "fallback_used": False,
                "extracted_from_text": False,
                "repair_attempted": True,
                "repair_success": True,
                "schema_coercion_success": True,
                "normalized": True,
                "error": _short_error(parse_error),
            }
            return JudgeScore.model_validate(normalized)

        fallback = fallback_judge_score(question, report, evidences, error=parse_error)
        self.last_stats = {
            "json_parse_success": False,
            "fallback_used": True,
            "extracted_from_text": False,
            "repair_attempted": bool(repair_stats["repair_attempted"]),
            "repair_success": False,
            "schema_coercion_success": False,
            "normalized": True,
            "error": _short_error(repair_stats.get("repair_error") or parse_error),
        }
        return JudgeScore.model_validate(fallback)


def coerce_judge_data(data: Any) -> dict[str, Any] | None:
    source: dict[str, Any] = {}
    if isinstance(data, dict):
        nested_scores = data.get("scores")
        if isinstance(nested_scores, dict):
            source.update(nested_scores)
            if "comments" in data:
                source["comments"] = data["comments"]
        else:
            source.update(data)
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            metric = item.get("metric") or item.get("name") or item.get("field")
            if metric is None:
                continue
            source[str(metric)] = item.get("score", item.get("value"))
            if "comments" in item and "comments" not in source:
                source["comments"] = item["comments"]
    else:
        return None

    canonical: dict[str, Any] = {}
    aliases = {
        "factuality": "factuality",
        "factual_accuracy": "factuality",
        "factual accuracy": "factuality",
        "coverage": "coverage",
        "reasoning_depth": "reasoning_depth",
        "reasoning depth": "reasoning_depth",
        "reasoning": "reasoning_depth",
        "citation_quality": "citation_quality",
        "citation quality": "citation_quality",
        "citations": "citation_quality",
        "citation": "citation_quality",
        "clarity": "clarity",
        "readability": "clarity",
        "overall": "overall",
        "final_score": "overall",
        "final score": "overall",
    }
    for key, value in source.items():
        normalized_key = str(key).strip().lower().replace("-", "_")
        normalized_key = re.sub(r"\s+", " ", normalized_key)
        canonical_key = aliases.get(normalized_key) or aliases.get(normalized_key.replace(" ", "_"))
        if canonical_key:
            canonical[canonical_key] = value

    score_count = sum(1 for field in SCORE_FIELDS if field in canonical and field != "overall")
    if "overall" in canonical:
        score_count += 1
    if score_count < 3:
        return None
    canonical["comments"] = source.get("comments", "")
    return normalize_judge_score(canonical)


def normalize_judge_score(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize loosely structured judge data into the canonical score schema."""
    source: dict[str, Any] = data
    nested_scores = data.get("scores")
    if isinstance(nested_scores, dict):
        source = {**nested_scores, "comments": data.get("comments", nested_scores.get("comments", ""))}

    provided_scores: dict[str, int] = {}
    for field in SCORE_FIELDS:
        if field in source:
            provided_scores[field] = _coerce_score(source[field])

    if provided_scores:
        default_score = provided_scores.get("overall")
        if default_score is None:
            default_score = int(round(sum(provided_scores.values()) / len(provided_scores)))
    else:
        default_score = 0

    normalized = {field: provided_scores.get(field, default_score) for field in SCORE_FIELDS}
    normalized["comments"] = str(source.get("comments") or data.get("comments") or "")
    return normalized


def extract_scores_from_text(text: str) -> dict[str, Any] | None:
    """Extract judge scores from natural-language scoring text."""
    label_patterns = {
        "factuality": r"factuality|factual\s+accuracy",
        "coverage": r"coverage",
        "reasoning_depth": r"reasoning[_\s-]*depth|reasoning",
        "citation_quality": r"citation[_\s-]*quality|citation",
        "clarity": r"clarity",
        "overall": r"overall",
    }
    extracted: dict[str, Any] = {}
    for field, label in label_patterns.items():
        pattern = re.compile(
            rf"\b(?:{label})\b(?:\s+(?:score|rating))?(?:\s+(?:is|as))?\s*(?:=|:)?\s*(\d{{1,3}}(?:\.\d+)?)\s*(?:/100)?",
            flags=re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            extracted[field] = match.group(1)

    if len(extracted) < 3:
        return None

    comments = _extract_comment_from_text(text)
    extracted["comments"] = comments
    return normalize_judge_score(extracted)


def fallback_judge_score(
    question: str,
    report: str,
    evidences: list[Evidence],
    error: str | None = None,
) -> dict[str, Any]:
    """Return a local rule-based fallback score when the LLM judge output is unusable."""
    del question
    scores = {
        "factuality": 70,
        "coverage": 70,
        "reasoning_depth": 70,
        "citation_quality": 60,
        "clarity": 70,
        "overall": 70,
    }

    normalized_report = report.lower()
    sections = ["abstract", "key findings", "evidence summary", "limitations", "conclusion"]
    section_hits = sum(1 for section in sections if f"## {section}" in normalized_report or f"# {section}" in normalized_report)
    scores["clarity"] += section_hits * 2
    scores["coverage"] += section_hits * 2

    if len(evidences) >= 4:
        scores["coverage"] += 5
        scores["reasoning_depth"] += 3

    if any(_evidence_has_supported_source(evidence) for evidence in evidences):
        scores["citation_quality"] += 15

    weak_source_count = sum(1 for evidence in evidences if _evidence_has_weak_source(evidence))
    if weak_source_count:
        scores["citation_quality"] -= min(25, weak_source_count * 5)

    if len(report.split()) < 120:
        scores["coverage"] -= 15
        scores["reasoning_depth"] -= 15

    if "unsupported" in normalized_report or "example.com" in normalized_report or "null source" in normalized_report:
        scores["citation_quality"] -= 10

    for field in SCORE_FIELDS:
        scores[field] = _coerce_score(scores[field])
    scores["overall"] = _coerce_score(
        round(
            0.25 * scores["factuality"]
            + 0.2 * scores["coverage"]
            + 0.2 * scores["reasoning_depth"]
            + 0.15 * scores["citation_quality"]
            + 0.2 * scores["clarity"]
        )
    )

    comment = "Fallback judge score was used because the LLM judge output was not valid JSON."
    if error:
        comment += f" Error: {_short_error(error)}"
    scores["comments"] = comment
    return scores


def _coerce_score(value: object) -> int:
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if not match:
            return 0
        value = match.group(0)
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, score))


def _default_judge_stats() -> dict[str, Any]:
    return {
        "json_parse_success": False,
        "fallback_used": False,
        "extracted_from_text": False,
        "repair_attempted": False,
        "repair_success": False,
        "schema_coercion_success": False,
        "normalized": False,
        "error": None,
    }


def _write_debug_judge_output(raw: str) -> None:
    try:
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "debug_last_judge_output.txt").write_text(raw, encoding="utf-8")
    except OSError:
        return


def _short_error(error: str | None) -> str | None:
    if not error:
        return None
    return error[:300]


def _extract_comment_from_text(text: str) -> str:
    compact = " ".join(text.strip().split())
    return compact[:300]


def _evidence_has_supported_source(evidence: Evidence) -> bool:
    if evidence.source_url and not evidence.source_url.startswith("mock://") and "example.com" not in evidence.source_url:
        return True
    status = evidence.metadata.get("citation_validation_status")
    return status == "supported"


def _evidence_has_weak_source(evidence: Evidence) -> bool:
    if evidence.source_url is None:
        return True
    if "example.com" in evidence.source_url:
        return True
    status = evidence.metadata.get("citation_validation_status")
    return status in {"unsupported", "no_source", "url_unreachable"}
