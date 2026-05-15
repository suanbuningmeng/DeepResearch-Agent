import json
import importlib.util
import subprocess
import sys
from pathlib import Path


def load_run_demo_module():
    project_root = Path(__file__).resolve().parents[1]
    module_path = project_root / "scripts" / "run_demo.py"
    spec = importlib.util.spec_from_file_location("run_demo_for_tests", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_demo_creates_report_and_trace(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "outputs"

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_demo.py"),
            "--question",
            "What are the main challenges and recent methods for long-context LLM evaluation?",
            "--backend",
            "mock",
            "--output-dir",
            str(output_dir),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    report_path = output_dir / "demo_report.md"
    trace_path = output_dir / "demo_trace.json"

    assert "demo_report.md" in result.stdout
    assert report_path.exists()
    assert trace_path.exists()

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["backend"] == "mock"
    assert len(trace["planned_subtasks"]) >= 3
    assert trace["final_judge_score"]["overall"] == 86


def test_resume_preset_sets_stable_grounded_defaults() -> None:
    run_demo = load_run_demo_module()
    raw_args = ["--preset", "resume", "--backend", "openai-compatible"]
    args = run_demo.build_parser().parse_args(raw_args)

    run_demo._apply_preset(args, raw_args)

    assert args.mode == "dag"
    assert args.max_concurrency == 1
    assert args.global_timeout_seconds == 720
    assert args.request_timeout == 240
    assert args.max_tokens == 2048
    assert args.enable_thinking is False
    assert args.enable_web_search is True
    assert args.search_provider == "arxiv"
    assert args.max_search_queries == 1
    assert args.search_top_k == 1
    assert args.enable_context_compression is False
    assert args.enable_conflict_detection is False
    assert args.enable_red_blue is False


def test_resume_preset_preserves_explicit_overrides() -> None:
    run_demo = load_run_demo_module()
    raw_args = [
        "--preset",
        "resume",
        "--max-concurrency",
        "2",
        "--search-top-k",
        "3",
        "--enable-red-blue",
    ]
    args = run_demo.build_parser().parse_args(raw_args)

    run_demo._apply_preset(args, raw_args)

    assert args.max_concurrency == 2
    assert args.search_top_k == 3
    assert args.enable_red_blue is True
    assert args.enable_web_search is True
