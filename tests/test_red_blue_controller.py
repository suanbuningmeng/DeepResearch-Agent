import asyncio

from deepresearch_agent.red_blue.controller import RedBlueController
from deepresearch_agent.red_blue.schemas import (
    BlueRepairResult,
    RedCritique,
    RedIssue,
    RedIssueSeverity,
    RedIssueType,
    RepairActionType,
    RepairPatch,
)
from deepresearch_agent.schemas import Evidence, JudgeScore


def evidence() -> Evidence:
    return Evidence(
        id="e1",
        task_id="task_1",
        title="Evidence",
        content="Evidence content.",
        source_url="mock://source",
        confidence=0.8,
    )


def issue(severity: RedIssueSeverity = RedIssueSeverity.HIGH) -> RedIssue:
    return RedIssue(
        id="issue_1",
        issue_type=RedIssueType.COVERAGE,
        severity=severity,
        location="Key Findings",
        description="Needs more coverage.",
        evidence_ids=["e1"],
        suggested_action=RepairActionType.ADD,
        rationale="Use available evidence.",
    )


def critique_with(severity: RedIssueSeverity | None) -> RedCritique:
    issues = [] if severity is None else [issue(severity)]
    return RedCritique(
        round_id=1,
        issues=issues,
        high_severity_count=1 if severity == RedIssueSeverity.HIGH else 0,
        medium_severity_count=1 if severity == RedIssueSeverity.MEDIUM else 0,
        low_severity_count=1 if severity == RedIssueSeverity.LOW else 0,
        summary="critique",
    )


class FakeRedAgent:
    def __init__(self, severity: RedIssueSeverity | None = RedIssueSeverity.HIGH) -> None:
        self.severity = severity
        self.last_stats = {"red_parse_success": True, "red_fallback_used": False}

    async def critique(self, **kwargs):
        round_id = kwargs.get("round_id", 1)
        critique = critique_with(self.severity)
        critique.round_id = round_id
        return critique


class FakeBlueAgent:
    def __init__(self, suffix: str = "\n\nRepaired.") -> None:
        self.suffix = suffix
        self.last_stats = {"blue_parse_success": True, "blue_fallback_used": False}

    async def repair(self, question, report, evidences, critique, round_id=1):
        return BlueRepairResult(
            round_id=round_id,
            revised_report=report + self.suffix,
            patches=[
                RepairPatch(
                    id=f"patch_{round_id}",
                    action_type=RepairActionType.ADD,
                    target_location="Key Findings",
                    after_text=self.suffix.strip(),
                    rationale="test",
                    applied=True,
                )
            ],
            summary="repaired",
        )


class SequenceJudge:
    def __init__(self, scores: list[int]) -> None:
        self.scores = list(scores)

    async def judge(self, question, report, evidences):
        score = self.scores.pop(0) if self.scores else 86
        return JudgeScore(
            factuality=score,
            coverage=score,
            reasoning_depth=score,
            citation_quality=score,
            clarity=score,
            overall=score,
            comments="test",
        )


def test_controller_respects_max_rounds() -> None:
    async def run() -> None:
        controller = RedBlueController(
            FakeRedAgent(),
            FakeBlueAgent(),
            SequenceJudge([87, 88]),
            max_rounds=2,
        )

        _report, stats = await controller.run("q", "report", [evidence()], {"overall": 86})

        assert stats.rounds_completed == 2
        assert stats.stopped_reason == "max_rounds_reached"

    asyncio.run(run())


def test_controller_selects_repaired_report_when_score_improves() -> None:
    async def run() -> None:
        controller = RedBlueController(FakeRedAgent(), FakeBlueAgent(), SequenceJudge([90]), max_rounds=1)

        report, stats = await controller.run("q", "report", [evidence()], {"overall": 86})

        assert "Repaired" in report
        assert stats.final_report_selected == "repaired"
        assert stats.score_delta == 4

    asyncio.run(run())


def test_controller_rolls_back_when_repair_degrades_score() -> None:
    async def run() -> None:
        controller = RedBlueController(FakeRedAgent(), FakeBlueAgent(), SequenceJudge([70]), max_rounds=1)

        report, stats = await controller.run("q", "report", [evidence()], {"overall": 86})

        assert report == "report"
        assert stats.stopped_reason == "repair_degraded_score"
        assert stats.final_report_selected == "initial"

    asyncio.run(run())


def test_controller_stops_when_no_high_severity_issue() -> None:
    async def run() -> None:
        controller = RedBlueController(
            FakeRedAgent(RedIssueSeverity.MEDIUM),
            FakeBlueAgent(),
            SequenceJudge([90]),
            max_rounds=2,
            stop_on_no_high_severity=True,
        )

        report, stats = await controller.run("q", "report", [evidence()], {"overall": 86})

        assert report == "report"
        assert stats.rounds_completed == 1
        assert stats.stopped_reason == "no_high_severity_issues"
        assert stats.rounds[0].red_parse_success is True

    asyncio.run(run())


def test_controller_stats_fields_are_complete() -> None:
    async def run() -> None:
        controller = RedBlueController(FakeRedAgent(None), FakeBlueAgent(), SequenceJudge([]), max_rounds=2)

        _report, stats = await controller.run("q", "report", [evidence()], {"overall": 86})

        assert stats.enabled is True
        assert stats.initial_overall_score == 86
        assert stats.final_overall_score == 86
        assert stats.total_red_issues == 0
        assert stats.total_patches_applied == 0
        assert stats.stopped_reason == "no_issues"

    asyncio.run(run())
