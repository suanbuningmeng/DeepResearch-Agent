import asyncio

from deepresearch_agent.agents.writer import WriterAgent, select_top_evidences_per_task
from deepresearch_agent.llm.mock_client import MockLLM
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


def test_writer_prefers_non_degraded_evidence() -> None:
    evidences = [
        Evidence(
            id="degraded",
            task_id="task_a",
            title="Fallback",
            content="Degraded fallback.",
            source_url="mock://degraded/task_a",
            confidence=0.99,
            metadata={"degraded": True},
        ),
        Evidence(
            id="grounded",
            task_id="task_a",
            title="Grounded",
            content="Grounded paper evidence.",
            source_url="http://arxiv.org/abs/2307.03172v1",
            confidence=0.5,
        ),
    ]

    selected = select_top_evidences_per_task(evidences, top_k_per_task=1)

    assert [evidence.id for evidence in selected] == ["grounded"]


def test_writer_prefers_supported_non_fallback_evidence() -> None:
    evidences = [
        Evidence(
            id="fallback",
            task_id="task_a",
            title="Fallback",
            content="Fallback paper snippet.",
            source_url="http://arxiv.org/abs/fallback",
            confidence=0.99,
            metadata={"grounded_fallback": True, "citation_validation_status": "supported"},
        ),
        Evidence(
            id="supported",
            task_id="task_a",
            title="Supported",
            content="Supported paper evidence.",
            source_url="http://arxiv.org/abs/supported",
            confidence=0.5,
            metadata={"citation_validation_status": "supported"},
        ),
    ]

    selected = select_top_evidences_per_task(evidences, top_k_per_task=1)

    assert [evidence.id for evidence in selected] == ["supported"]


def test_writer_outputs_references_and_evidence_ids() -> None:
    async def run() -> None:
        evidence = Evidence(
            id="task_1_grounded_evidence_1",
            task_id="task_1",
            title="LongBench evaluates long-context models",
            content="Long-context evaluation should include retrieval and reasoning.",
            source_url="http://arxiv.org/abs/2308.14508",
            confidence=0.8,
            metadata={"citation_validation_status": "supported"},
        )

        report = await WriterAgent(MockLLM()).write("What are long-context evaluation challenges?", [evidence])

        assert "## References" in report
        assert "[task_1_grounded_evidence_1]" in report
        assert "http://arxiv.org/abs/2308.14508" in report

    asyncio.run(run())


def test_writer_marks_degraded_evidence_in_limitations() -> None:
    async def run() -> None:
        evidence = Evidence(
            id="task_1_degraded_evidence",
            task_id="task_1",
            title="Degraded fallback",
            content="Fallback evidence.",
            source_url="mock://degraded/task_1",
            confidence=0.9,
            metadata={"degraded": True},
        )

        report = await WriterAgent(MockLLM()).write("What are long-context evaluation challenges?", [evidence])
        limitations = report.split("## Limitations", 1)[1].split("## Conclusion", 1)[0]
        key_findings = report.split("## Key Findings", 1)[1].split("## Evidence Summary", 1)[0]

        assert "degraded" in limitations
        assert "task_1_degraded_evidence" in limitations
        assert "task_1_degraded_evidence" not in key_findings

    asyncio.run(run())


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


def test_report_quality_check_detects_natural_language_tail_truncation() -> None:
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
Long-context evaluation should support realistic, real-world, long
"""

    result = check_report_completeness(markdown)

    assert result["natural_language_tail_incomplete"] is True
    assert result["report_incomplete"] is True


def test_repair_incomplete_report_fixes_truncated_conclusion() -> None:
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
Long-context evaluation should support realistic, real-world, long
"""

    repaired = repair_incomplete_report(markdown)

    assert "real-world, long" not in repaired
    assert "## Conclusion" in repaired
    assert repaired.rstrip().endswith("Note: Some incomplete generated fragments were removed during report post-processing.")


def test_repaired_natural_language_tail_report_is_complete() -> None:
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
Long-context evaluation should support realistic, real-world, long
"""

    repaired = repair_incomplete_report(markdown)
    result = check_report_completeness(repaired)

    assert result["natural_language_tail_incomplete"] is False
    assert result["report_incomplete"] is False
