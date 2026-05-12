from deepresearch_agent.agents.writer import select_top_evidences_per_task
from deepresearch_agent.report import (
    check_report_completeness,
    ensure_report_completeness,
    repair_incomplete_report,
)
from deepresearch_agent.schemas import Evidence


def test_report_quality_check_detects_missing_conclusion() -> None:
    markdown = """# Report

## Abstract
Text.

## Key Findings
- Finding.

## Evidence Summary
Summary.

## Limitations
Limit.
"""

    result = check_report_completeness(markdown)

    assert "Conclusion" in result["missing_sections"]
    assert result["report_incomplete"] is True


def test_report_quality_check_detects_incomplete_bullet_tail() -> None:
    markdown = """# Report

## Abstract
Text.

## Key Findings
- Finding.

## Evidence Summary
Summary.

## Limitations
Limit.

## Conclusion
- **
"""

    result = check_report_completeness(markdown)

    assert result["incomplete_tail"] is True
    assert result["report_incomplete"] is True


def test_ensure_report_completeness_appends_missing_sections() -> None:
    markdown = """# Report

## Abstract
Text.

## Key Findings
- Finding.

## Evidence Summary
Summary.
"""

    fixed, result = ensure_report_completeness(markdown)

    assert "## Limitations" in fixed
    assert "## Conclusion" in fixed
    assert result["missing_sections"] == []


def test_writer_selects_top_k_evidence_per_task() -> None:
    evidences = [
        Evidence(id="a1", task_id="task_a", title="A1", content="low", confidence=0.2),
        Evidence(id="a2", task_id="task_a", title="A2", content="high", confidence=0.9),
        Evidence(id="a3", task_id="task_a", title="A3", content="mid", confidence=0.5),
        Evidence(id="b1", task_id="task_b", title="B1", content="high", confidence=0.8),
        Evidence(id="b2", task_id="task_b", title="B2", content="low", confidence=0.1),
    ]

    selected = select_top_evidences_per_task(evidences, top_k_per_task=2)

    assert [evidence.id for evidence in selected] == ["a2", "a3", "b1", "b2"]


def test_report_quality_check_detects_dangling_evidence_reference() -> None:
    markdown = """# Report

## Abstract
Text.

## Key Findings
- Finding.

## Evidence Summary
- **[task_4_evidence_2

## Limitations
Limit.

## Conclusion
Done.
"""

    result = check_report_completeness(markdown)

    assert result["dangling_evidence_reference"] is True
    assert result["report_incomplete"] is True


def test_report_quality_check_detects_unfinished_markdown_bold() -> None:
    markdown = """# Report

## Abstract
This is **unfinished.

## Key Findings
- Finding.

## Evidence Summary
Summary.

## Limitations
Limit.

## Conclusion
Done.
"""

    result = check_report_completeness(markdown)

    assert result["unfinished_markdown_bold"] is True
    assert result["report_incomplete"] is True


def test_repair_incomplete_report_removes_dangling_evidence_bullet() -> None:
    markdown = """# Report

## Abstract
Text.

## Key Findings
- Finding.

## Evidence Summary
- **[task_4_evidence_1]**: Complete evidence sentence.
- **[task_4_evidence_2

## Limitations
Limit.

## Conclusion
Done.
"""

    repaired = repair_incomplete_report(markdown)

    assert "- **[task_4_evidence_2" not in repaired
    assert "post-processing" in repaired


def test_repaired_report_is_complete() -> None:
    markdown = """# Report

## Abstract
Text.

## Key Findings
- Finding.

## Evidence Summary
- **[task_4_evidence_2

## Limitations
Limit.

## Conclusion
Done.
"""

    repaired = repair_incomplete_report(markdown)
    result = check_report_completeness(repaired)

    assert result["report_incomplete"] is False
