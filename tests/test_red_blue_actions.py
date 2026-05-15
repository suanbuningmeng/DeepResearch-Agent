from deepresearch_agent.red_blue.actions import apply_patch, append_cautionary_note, ensure_report_sections
from deepresearch_agent.red_blue.schemas import (
    RedIssue,
    RedIssueSeverity,
    RedIssueType,
    RepairActionType,
    RepairPatch,
)


def issue() -> RedIssue:
    return RedIssue(
        id="issue_1",
        issue_type=RedIssueType.UNSUPPORTED_CLAIM,
        severity=RedIssueSeverity.HIGH,
        location="Conclusion",
        description="Claim needs verification.",
        evidence_ids=["e1"],
        suggested_action=RepairActionType.VERIFY,
        rationale="No verified citation.",
    )


def test_append_cautionary_note_adds_note() -> None:
    report = "# Report\n\n## Conclusion\nDone."

    revised = append_cautionary_note(report, [issue()])

    assert "## Cautionary Note" in revised
    assert "Claim needs verification" in revised


def test_ensure_report_sections_adds_missing_sections() -> None:
    revised = ensure_report_sections("# Report\n\n## Conclusion\nDone.")

    assert "## Abstract" in revised
    assert "## Key Findings" in revised
    assert "## Evidence Summary" in revised
    assert "## Limitations" in revised


def test_apply_patch_simple_modify() -> None:
    patch = RepairPatch(
        id="patch_1",
        action_type=RepairActionType.MODIFY,
        target_location="Conclusion",
        before_text="absolute claim",
        after_text="tentative claim",
        rationale="Use cautious wording.",
        applied=True,
    )

    assert apply_patch("This is an absolute claim.", patch) == "This is an tentative claim."
