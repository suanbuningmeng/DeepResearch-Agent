from __future__ import annotations

from deepresearch_agent.red_blue.schemas import RedIssue, RepairActionType, RepairPatch


REQUIRED_SECTIONS = [
    "Abstract",
    "Key Findings",
    "Evidence Summary",
    "Limitations",
    "Conclusion",
]


def apply_patch(report: str, patch: RepairPatch) -> str:
    if not patch.applied:
        return report
    if patch.action_type == RepairActionType.MODIFY:
        if patch.before_text and patch.after_text and patch.before_text in report:
            return report.replace(patch.before_text, patch.after_text, 1)
        if patch.after_text:
            return report.rstrip() + "\n\n" + patch.after_text.strip() + "\n"
    if patch.action_type == RepairActionType.DELETE and patch.before_text:
        return report.replace(patch.before_text, "", 1)
    if patch.action_type in {RepairActionType.ADD, RepairActionType.VERIFY} and patch.after_text:
        return report.rstrip() + "\n\n" + patch.after_text.strip() + "\n"
    return report


def append_cautionary_note(report: str, issues: list[RedIssue]) -> str:
    if "## Cautionary Note" in report:
        return report
    issue_lines = []
    for issue in issues[:5]:
        issue_lines.append(f"- {issue.issue_type.value}: {issue.description}")
    if not issue_lines:
        issue_lines.append("- No high-confidence repair was available; treat unsupported claims cautiously.")
    note = (
        "## Cautionary Note\n"
        "Some claims should be treated as tentative because the available evidence may be incomplete or unverified.\n"
        + "\n".join(issue_lines)
    )
    if "## Conclusion" in report:
        return report.replace("## Conclusion", note + "\n\n## Conclusion", 1)
    return report.rstrip() + "\n\n" + note + "\n"


def ensure_report_sections(report: str) -> str:
    repaired = report.rstrip()
    for section in REQUIRED_SECTIONS:
        if f"## {section}" not in repaired and f"# {section}" not in repaired:
            repaired += f"\n\n## {section}\nThis section requires further validation based on available evidence."
    return repaired.rstrip() + "\n"
