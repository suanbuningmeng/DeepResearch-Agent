from __future__ import annotations

from typing import Any

from deepresearch_agent.agents.blue import BlueAgent
from deepresearch_agent.agents.judge import JudgeAgent
from deepresearch_agent.agents.red import RedAgent
from deepresearch_agent.red_blue.scoring import overall_score
from deepresearch_agent.red_blue.schemas import RedBlueRoundTrace, RedBlueStats
from deepresearch_agent.report.quality import ensure_report_completeness
from deepresearch_agent.schemas import Evidence


class RedBlueController:
    def __init__(
        self,
        red_agent: RedAgent,
        blue_agent: BlueAgent,
        judge_agent: JudgeAgent,
        max_rounds: int = 2,
        min_score_delta: int = 1,
        stop_on_no_high_severity: bool = True,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1.")
        self.red_agent = red_agent
        self.blue_agent = blue_agent
        self.judge_agent = judge_agent
        self.max_rounds = max_rounds
        self.min_score_delta = min_score_delta
        self.stop_on_no_high_severity = stop_on_no_high_severity

    async def run(
        self,
        question: str,
        initial_report: str,
        evidences: list[Evidence],
        initial_judge_score: dict | None = None,
        memory_stats: dict | None = None,
        compression_stats: dict | None = None,
        conflict_stats: dict | None = None,
    ) -> tuple[str, RedBlueStats]:
        current_report = initial_report
        current_score_dict = initial_judge_score or {}
        current_overall = overall_score(current_score_dict)
        initial_overall = current_overall
        rounds: list[RedBlueRoundTrace] = []
        total_red_issues = 0
        total_patches_applied = 0
        stopped_reason = "max_rounds_reached"
        final_report_selected = "initial"
        unresolved_high_severity_count = 0

        for round_id in range(1, self.max_rounds + 1):
            score_before = current_overall
            critique = await self.red_agent.critique(
                question=question,
                report=current_report,
                evidences=evidences,
                judge_score=current_score_dict,
                memory_stats=memory_stats,
                compression_stats=compression_stats,
                conflict_stats=conflict_stats,
                round_id=round_id,
            )
            red_stats = dict(self.red_agent.last_stats)
            total_red_issues += len(critique.issues)
            unresolved_high_severity_count = critique.high_severity_count

            if not critique.issues:
                stopped_reason = "no_issues"
                rounds.append(
                    RedBlueRoundTrace(
                        round_id=round_id,
                        red_issue_count=0,
                        high_severity_count=0,
                        patches_applied_count=0,
                        judge_score_before=score_before,
                        judge_score_after=score_before,
                        score_delta=0,
                        stopped_reason=stopped_reason,
                        red_parse_success=bool(red_stats.get("red_parse_success")),
                        red_fallback_used=bool(red_stats.get("red_fallback_used")),
                    )
                )
                break

            if self.stop_on_no_high_severity and critique.high_severity_count == 0:
                stopped_reason = "no_high_severity_issues"
                rounds.append(
                    RedBlueRoundTrace(
                        round_id=round_id,
                        red_issue_count=len(critique.issues),
                        high_severity_count=0,
                        patches_applied_count=0,
                        judge_score_before=score_before,
                        judge_score_after=score_before,
                        score_delta=0,
                        stopped_reason=stopped_reason,
                        red_parse_success=bool(red_stats.get("red_parse_success")),
                        red_fallback_used=bool(red_stats.get("red_fallback_used")),
                    )
                )
                unresolved_high_severity_count = 0
                break

            repair = await self.blue_agent.repair(
                question=question,
                report=current_report,
                evidences=evidences,
                critique=critique,
                round_id=round_id,
            )
            blue_stats = dict(self.blue_agent.last_stats)
            revised_report, _quality = ensure_report_completeness(repair.revised_report)
            after_score = await self.judge_agent.judge(question, revised_report, evidences)
            after_score_dict = after_score.model_dump(mode="json")
            after_overall = overall_score(after_score)
            score_delta = _score_delta(score_before, after_overall)
            patches_applied = sum(1 for patch in repair.patches if patch.applied)
            total_patches_applied += patches_applied

            round_trace = RedBlueRoundTrace(
                round_id=round_id,
                red_issue_count=len(critique.issues),
                high_severity_count=critique.high_severity_count,
                patches_applied_count=patches_applied,
                judge_score_before=score_before,
                judge_score_after=after_overall,
                score_delta=score_delta,
                red_parse_success=bool(red_stats.get("red_parse_success")),
                red_fallback_used=bool(red_stats.get("red_fallback_used")),
                blue_parse_success=bool(blue_stats.get("blue_parse_success")),
                blue_fallback_used=bool(blue_stats.get("blue_fallback_used")),
            )

            if score_before is not None and after_overall is not None and after_overall < score_before:
                stopped_reason = "repair_degraded_score"
                round_trace.stopped_reason = stopped_reason
                rounds.append(round_trace)
                break

            current_report = revised_report
            current_score_dict = after_score_dict
            current_overall = after_overall
            final_report_selected = "repaired"
            round_trace.stopped_reason = "round_completed"
            rounds.append(round_trace)

            if (
                score_delta is not None
                and score_delta < self.min_score_delta
                and critique.high_severity_count == 0
            ):
                stopped_reason = "score_converged"
                round_trace.stopped_reason = stopped_reason
                break
        else:
            stopped_reason = "max_rounds_reached"

        if rounds and rounds[-1].stopped_reason in {
            "no_issues",
            "no_high_severity_issues",
            "score_converged",
            "repair_degraded_score",
        }:
            stopped_reason = rounds[-1].stopped_reason or stopped_reason

        final_overall = current_overall
        stats = RedBlueStats(
            enabled=True,
            max_rounds=self.max_rounds,
            rounds_completed=len(rounds),
            initial_overall_score=initial_overall,
            final_overall_score=final_overall,
            score_delta=_score_delta(initial_overall, final_overall),
            total_red_issues=total_red_issues,
            total_patches_applied=total_patches_applied,
            unresolved_high_severity_count=unresolved_high_severity_count,
            stopped_reason=stopped_reason,
            rounds=rounds,
            final_report_selected=final_report_selected,
        )
        return current_report, stats


def _score_delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return after - before
