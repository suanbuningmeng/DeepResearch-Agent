from __future__ import annotations

from deepresearch_agent.agents.planner import coerce_planner_data


def test_planner_coerces_tasks_key() -> None:
    tasks, partial = coerce_planner_data(
        {
            "tasks": [
                {
                    "task_id": "alpha",
                    "title": "Map challenges",
                    "details": "Map the main challenges.",
                }
            ]
        },
        "Question?",
    )

    assert tasks[0].id == "task_alpha"
    assert tasks[0].name == "Map challenges"
    assert tasks[0].description == "Map the main challenges."
    assert partial is True
    assert len(tasks) >= 3


def test_planner_coerces_direct_list() -> None:
    tasks, partial = coerce_planner_data(
        [
            {"objective": "Study methods"},
            {"task_name": "Compare results", "description": "Compare benchmark results."},
            {"name": "Summarize limitations"},
        ],
        "Question?",
    )

    assert tasks[0].id == "task_1"
    assert tasks[0].name == "Study methods"
    assert tasks[1].name == "Compare results"
    assert partial is False
