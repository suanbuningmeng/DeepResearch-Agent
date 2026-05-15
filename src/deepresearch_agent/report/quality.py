from __future__ import annotations

import re
from typing import Any


REQUIRED_SECTIONS = [
    "Abstract",
    "Key Findings",
    "Evidence Summary",
    "Limitations",
    "Conclusion",
]


def check_report_completeness(markdown: str) -> dict[str, Any]:
    missing_sections = [
        section
        for section in REQUIRED_SECTIONS
        if not re.search(rf"^\s*#+\s+{re.escape(section)}\b", markdown, flags=re.IGNORECASE | re.MULTILINE)
    ]
    unfinished_markdown_bold = markdown.count("**") % 2 == 1
    dangling_evidence_reference = _has_dangling_evidence_reference(markdown)
    evidence_summary_incomplete = _evidence_summary_incomplete(markdown)
    incomplete_tail = _has_incomplete_bullet_tail(markdown)
    natural_language_tail_incomplete = _natural_language_tail_incomplete(markdown)
    report_incomplete = bool(
        missing_sections
        or incomplete_tail
        or unfinished_markdown_bold
        or dangling_evidence_reference
        or evidence_summary_incomplete
        or natural_language_tail_incomplete
    )
    return {
        "is_complete": not report_incomplete,
        "missing_sections": missing_sections,
        "incomplete_tail": incomplete_tail,
        "unfinished_markdown_bold": unfinished_markdown_bold,
        "evidence_summary_incomplete": evidence_summary_incomplete,
        "dangling_evidence_reference": dangling_evidence_reference,
        "natural_language_tail_incomplete": natural_language_tail_incomplete,
        "repair_applied": False,
        "report_incomplete": report_incomplete,
    }


def repair_incomplete_report(markdown: str) -> str:
    before = check_report_completeness(markdown)
    repaired = markdown.rstrip()

    if before["evidence_summary_incomplete"] or before["dangling_evidence_reference"]:
        repaired = _remove_incomplete_evidence_summary_bullet(repaired)

    if check_report_completeness(repaired)["unfinished_markdown_bold"]:
        repaired = _repair_unfinished_bold(repaired)

    if check_report_completeness(repaired)["natural_language_tail_incomplete"]:
        repaired = _repair_natural_language_tail(repaired)

    additions: list[str] = []
    missing_sections = set(check_report_completeness(repaired)["missing_sections"])
    if "Limitations" in missing_sections:
        additions.append(
            "## Limitations\n"
            "This report may be incomplete because some model outputs were partial or unavailable."
        )
    if "Conclusion" in missing_sections:
        additions.append(
            "## Conclusion\n"
            "The available evidence suggests that long-context LLM evaluation should balance retrieval, synthesis, robustness, and citation quality."
        )
    if additions:
        repaired = repaired.rstrip() + "\n\n" + "\n\n".join(additions)

    after = check_report_completeness(repaired)
    if before["report_incomplete"] and not after["report_incomplete"]:
        repaired = (
            repaired.rstrip()
            + "\n\n"
            + "Note: Some incomplete generated fragments were removed during report post-processing."
        )
    return repaired.rstrip() + "\n"


def ensure_report_completeness(markdown: str) -> tuple[str, dict[str, Any]]:
    before = check_report_completeness(markdown)
    if before["report_incomplete"]:
        markdown = repair_incomplete_report(markdown)
        after = check_report_completeness(markdown)
        after["repair_applied"] = markdown != ""
    else:
        after = dict(before)

    quality = {
        "before_repair": before,
        "after_repair": after,
        "repair_applied": bool(before["report_incomplete"]),
    }
    quality.update(after)
    return markdown, quality


def _has_dangling_evidence_reference(markdown: str) -> bool:
    for line in markdown.splitlines():
        stripped = line.strip()
        if re.match(r"^[-*]\s+\*\*\[[^\]\n]+$", stripped):
            return True
        if re.match(r"^[-*]\s+\*\*\[[^\]\n]+\]\s*$", stripped):
            return True
        if re.match(r"^[-*]\s+\*\*\[[^\]\n]+\]\*\*:\s*$", stripped):
            return True
    return False


def _evidence_summary_incomplete(markdown: str) -> bool:
    section = _extract_section(markdown, "Evidence Summary")
    if not section:
        return False
    bullets = [line.strip() for line in section.splitlines() if re.match(r"^[-*]\s+", line.strip())]
    if not bullets:
        return False
    last_bullet = bullets[-1]
    if _bullet_is_incomplete(last_bullet):
        return True
    return False


def _has_incomplete_bullet_tail(markdown: str) -> bool:
    if _bullet_is_incomplete(markdown.rstrip().splitlines()[-1].strip() if markdown.strip() else ""):
        return True
    for section in REQUIRED_SECTIONS:
        content = _extract_section(markdown, section)
        if not content:
            continue
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if lines and _bullet_is_incomplete(lines[-1]):
            return True
    return False


def _bullet_is_incomplete(line: str) -> bool:
    if not re.match(r"^[-*]\s+", line):
        return False
    content = re.sub(r"^[-*]\s+", "", line).strip()
    if content in {"*", "**", "- **"}:
        return True
    if re.match(r"^\*\*\[[^\]\n]+$", content):
        return True
    if re.match(r"^\*\*\[[^\]\n]+\]\s*$", content):
        return True
    if re.match(r"^\*\*\[[^\]\n]+\]\*\*:\s*$", content):
        return True
    return False


def _remove_incomplete_evidence_summary_bullet(markdown: str) -> str:
    lines = markdown.splitlines()
    section_bounds = _find_section_bounds(lines, "Evidence Summary")
    if section_bounds is None:
        return markdown
    start, end = section_bounds
    for index in range(end - 1, start, -1):
        if _bullet_is_incomplete(lines[index].strip()):
            del lines[index]
            break
    return "\n".join(lines)


def _repair_unfinished_bold(markdown: str) -> str:
    last_bold = markdown.rfind("**")
    if last_bold == -1:
        return markdown
    line_start = markdown.rfind("\n", 0, last_bold) + 1
    line_end = markdown.find("\n", last_bold)
    if line_end == -1:
        line_end = len(markdown)
    line = markdown[line_start:line_end].strip()
    if _bullet_is_incomplete(line):
        return markdown[:line_start].rstrip()
    return markdown + "**"


def _natural_language_tail_incomplete(markdown: str) -> bool:
    paragraph = _last_body_paragraph(markdown)
    if not paragraph:
        return False
    if paragraph.endswith(("```", ".", "!", "?", "。", "！", "？")):
        return False
    lowered = paragraph.lower().rstrip()
    if lowered.endswith((",", ";", ":", "-", "and", "or", "but", "with", "for", "of", "to", "in", "on", "at", "by", "from", "as", "than", "long")):
        return True
    return True


def _repair_natural_language_tail(markdown: str) -> str:
    lines = markdown.rstrip().splitlines()
    paragraph_bounds = _last_body_paragraph_bounds(lines)
    if paragraph_bounds is not None:
        start, end = paragraph_bounds
        del lines[start:end]
    repaired = "\n".join(lines).rstrip()
    if "## Conclusion" not in repaired:
        repaired = repaired + "\n\n## Conclusion"
    conclusion = _extract_section(repaired, "Conclusion")
    if not conclusion.strip() or _natural_language_tail_incomplete(repaired):
        repaired = _replace_section(
            repaired,
            "Conclusion",
            "The available evidence indicates that long-context LLM evaluation should combine retrieval, synthesis, robustness, and citation-quality checks.",
        )
    return repaired


def _last_body_paragraph(markdown: str) -> str:
    bounds = _last_body_paragraph_bounds(markdown.rstrip().splitlines())
    if bounds is None:
        return ""
    start, end = bounds
    return " ".join(line.strip() for line in markdown.rstrip().splitlines()[start:end]).strip()


def _last_body_paragraph_bounds(lines: list[str]) -> tuple[int, int] | None:
    end = len(lines)
    while end > 0 and not lines[end - 1].strip():
        end -= 1
    while end > 0 and lines[end - 1].strip().startswith("Note: Some incomplete generated fragments"):
        end -= 1
        while end > 0 and not lines[end - 1].strip():
            end -= 1
    if end == 0:
        return None
    start = end - 1
    while start >= 0 and lines[start].strip():
        start -= 1
    start += 1
    paragraph_lines = lines[start:end]
    if not paragraph_lines:
        return None
    if all(line.strip().startswith("#") for line in paragraph_lines):
        return None
    return start, end


def _replace_section(markdown: str, section: str, content: str) -> str:
    lines = markdown.splitlines()
    bounds = _find_section_bounds(lines, section)
    if bounds is None:
        return markdown.rstrip() + f"\n\n## {section}\n{content}"
    start, end = bounds
    replacement = [lines[start], content]
    return "\n".join(lines[:start] + replacement + lines[end:])


def _extract_section(markdown: str, section: str) -> str:
    lines = markdown.splitlines()
    bounds = _find_section_bounds(lines, section)
    if bounds is None:
        return ""
    start, end = bounds
    return "\n".join(lines[start + 1 : end])


def _find_section_bounds(lines: list[str], section: str) -> tuple[int, int] | None:
    start = None
    heading_pattern = re.compile(r"^\s*#+\s+(.+?)\s*$")
    for index, line in enumerate(lines):
        match = heading_pattern.match(line)
        if not match:
            continue
        heading = match.group(1).strip().lower()
        if start is None and heading == section.lower():
            start = index
            continue
        if start is not None:
            return start, index
    if start is None:
        return None
    return start, len(lines)
