from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepresearch_agent.ui.config import DEFAULT_QUESTION, DemoRunConfig
from deepresearch_agent.search.factory import default_api_key_env
from deepresearch_agent.ui.trace_view import (
    conflict_rows,
    execution_step_rows,
    extract_trace_summary,
    get_conflict_stats,
    get_compression_stats,
    get_memory_stats,
    get_red_blue_stats,
    get_search_stats,
    red_blue_round_rows,
    task_rows,
)
from scripts.run_demo import run_demo_with_config


def main() -> None:
    st.set_page_config(page_title="DeepResearch-Agent Demo", layout="wide")
    st.title("DeepResearch-Agent Demo")

    config = _sidebar_config()

    if st.button("Run DeepResearch-Agent", type="primary"):
        errors = config.validate()
        keyless_search_provider = config.search_provider in {"mock", "arxiv", "paper"}
        if config.enable_web_search and not keyless_search_provider and not os.getenv(config.search_api_key_env):
            errors.append(f"Search API key env var {config.search_api_key_env} is not set.")
        if errors:
            for error in errors:
                st.error(error)
            return
        with st.spinner("Running DeepResearch-Agent..."):
            try:
                trace = asyncio.run(run_demo_with_config(config))
                st.session_state["trace"] = trace
            except Exception as exc:
                st.error(_sanitize_error(str(exc)))
                with st.expander("Debug traceback"):
                    st.code(_sanitize_error(traceback.format_exc()))
                return

    trace = st.session_state.get("trace")
    if trace:
        _render_results(trace)
    else:
        st.info("Choose parameters in the sidebar, then run the demo.")


def _sidebar_config() -> DemoRunConfig:
    with st.sidebar:
        st.header("Parameters")
        question = st.text_area("Question", DEFAULT_QUESTION, height=120)

        backend = st.selectbox("Backend", ["mock", "openai-compatible", "deepseek", "vllm"], index=0)
        mode = st.selectbox("Mode", ["linear", "dag"], index=1)

        api_base = None
        api_key_env = None
        model = None
        temperature = 0.2
        max_tokens = 768
        request_timeout = 180
        enable_thinking = None
        if backend in {"openai-compatible", "deepseek", "vllm"}:
            st.subheader("OpenAI-compatible")
            api_base = st.text_input("API base", "https://api.siliconflow.cn/v1")
            api_key_env = st.text_input("API key env", "SILICONFLOW_API_KEY")
            model = st.text_input("Model", "Qwen/Qwen3-8B")
            temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=0.2, step=0.1)
            max_tokens = st.number_input("Max tokens", min_value=128, max_value=8192, value=768, step=128)
            request_timeout = st.number_input("Request timeout", min_value=30, max_value=600, value=180, step=30)
            enable_thinking = st.checkbox("Enable thinking", value=False)

        st.subheader("DAG")
        max_concurrency = st.slider("Max concurrency", min_value=1, max_value=4, value=1)
        global_timeout_seconds = st.number_input("Global timeout seconds", min_value=30, max_value=1800, value=300, step=30)

        st.subheader("Memory")
        memory_backend = st.selectbox("Memory backend", ["memory", "sqlite"], index=0)
        clear_memory = st.checkbox("Clear memory", value=True)
        memory_db_path = st.text_input("Memory DB path", "data/memory.sqlite")
        vector_index_path = st.text_input("Vector index path", "data/vector_index.npz")
        enable_memory_retrieval = st.checkbox("Enable memory retrieval", value=True)
        memory_search_top_k = st.number_input("Memory search top K", min_value=1, max_value=50, value=10, step=1)

        st.subheader("Context Compression")
        enable_context_compression = st.checkbox("Enable context compression", value=True)
        compression_l1_top_n = st.number_input("Compression L1 top N", min_value=1, max_value=100, value=12, step=1)
        compression_l2_top_k = st.number_input("Compression L2 top K", min_value=1, max_value=50, value=8, step=1)
        writer_top_k_per_task = st.number_input("Writer top K per task", min_value=1, max_value=10, value=2, step=1)

        st.subheader("Conflict Detection")
        enable_conflict_detection = st.checkbox("Enable conflict detection", value=False)
        near_duplicate_threshold = st.number_input(
            "Near duplicate threshold",
            min_value=0.50,
            max_value=1.0,
            value=0.92,
            step=0.01,
        )
        semantic_opposition_threshold = st.number_input(
            "Semantic opposition threshold",
            min_value=0.30,
            max_value=1.0,
            value=0.65,
            step=0.01,
        )

        st.subheader("Red-Blue Repair")
        enable_red_blue = st.checkbox("Enable Red-Blue repair", value=False)
        red_blue_max_rounds = st.number_input("Red-Blue max rounds", min_value=1, max_value=5, value=2, step=1)
        red_blue_min_score_delta = st.number_input("Red-Blue min score delta", min_value=0, max_value=20, value=1, step=1)
        red_blue_stop_on_no_high_severity = st.checkbox("Stop on no high severity", value=True)

        st.subheader("Web Search / Citation Validation")
        enable_web_search = st.checkbox("Enable web search", value=False)
        search_provider = st.selectbox("Search provider", ["mock", "arxiv", "paper", "tavily", "exa", "serper", "brave", "web"], index=0)
        search_top_k = st.number_input("Search top K", min_value=1, max_value=10, value=3, step=1)
        max_search_queries = st.number_input("Max search queries", min_value=1, max_value=5, value=3, step=1)
        allow_network_fetch = st.checkbox("Allow network fetch", value=False)
        search_timeout = st.number_input("Search timeout", min_value=5, max_value=120, value=20, step=5)
        search_api_key_env = st.text_input("Search API key env", default_api_key_env(search_provider))
        search_api_base = st.text_input("Search API base", "")

        st.subheader("Failure Scenario")
        failure_scenario = st.selectbox(
            "Failure scenario",
            ["none", "fail_one", "fail_many", "timeout_one", "global_timeout"],
            index=0,
        )

    return DemoRunConfig(
        question=question,
        backend=backend,
        mode=mode,
        output_dir=Path("outputs"),
        api_base=api_base,
        api_key_env=api_key_env,
        model=model,
        temperature=temperature,
        max_tokens=int(max_tokens),
        request_timeout=int(request_timeout),
        enable_thinking=enable_thinking if backend != "mock" else None,
        max_concurrency=int(max_concurrency),
        global_timeout_seconds=int(global_timeout_seconds),
        writer_top_k_per_task=int(writer_top_k_per_task),
        memory_backend=memory_backend,
        memory_db_path=memory_db_path,
        vector_index_path=vector_index_path,
        enable_memory_retrieval=enable_memory_retrieval,
        memory_search_top_k=int(memory_search_top_k),
        clear_memory=clear_memory,
        enable_context_compression=enable_context_compression,
        compression_l1_top_n=int(compression_l1_top_n),
        compression_l2_top_k=int(compression_l2_top_k),
        enable_conflict_detection=enable_conflict_detection,
        near_duplicate_threshold=float(near_duplicate_threshold),
        semantic_opposition_threshold=float(semantic_opposition_threshold),
        enable_red_blue=enable_red_blue,
        red_blue_max_rounds=int(red_blue_max_rounds),
        red_blue_min_score_delta=int(red_blue_min_score_delta),
        red_blue_stop_on_no_high_severity=red_blue_stop_on_no_high_severity,
        enable_web_search=enable_web_search,
        search_provider=search_provider,
        search_top_k=int(search_top_k),
        max_search_queries=int(max_search_queries),
        allow_network_fetch=allow_network_fetch,
        search_timeout=int(search_timeout),
        search_api_key_env=search_api_key_env,
        search_api_base=search_api_base,
        search_provider_name=search_provider,
        failure_scenario=failure_scenario,
    )


def _render_results(trace: dict) -> None:
    tabs = st.tabs(["Report", "Trace Summary", "Task States", "Execution Steps", "Memory", "Compression", "Conflict", "Red-Blue", "Search & Citations", "Evaluation"])

    outputs = trace.get("outputs", {})
    report_path = Path(outputs.get("report", "outputs/demo_report.md"))
    report_markdown = report_path.read_text(encoding="utf-8") if report_path.exists() else ""

    with tabs[0]:
        st.markdown(report_markdown or "Report file not found.")
        if report_markdown:
            st.download_button("Download Report", report_markdown, file_name="demo_report.md")

    with tabs[1]:
        st.json(extract_trace_summary(trace))

    with tabs[2]:
        rows = task_rows(trace)
        st.dataframe(rows, use_container_width=True) if rows else st.info("No task state data available.")

    with tabs[3]:
        rows = execution_step_rows(trace)
        st.dataframe(rows, use_container_width=True) if rows else st.info("No execution steps available.")

    with tabs[4]:
        stats = get_memory_stats(trace)
        if stats:
            st.json(stats)
        else:
            st.info("Memory stats are not available for this run.")

    with tabs[5]:
        stats = get_compression_stats(trace)
        if not stats:
            st.info("Compression stats are not available for this run.")
        elif not stats.get("enabled"):
            st.info("Context compression is disabled.")
            st.json(stats)
        else:
            st.json(stats)

    with tabs[6]:
        stats = get_conflict_stats(trace)
        if not stats:
            st.info("Conflict stats are not available for this run.")
        elif not stats.get("enabled"):
            st.info("Conflict detection is disabled.")
            st.json(stats)
        else:
            summary = {
                "enabled": stats.get("enabled"),
                "conflict_count": stats.get("conflict_count"),
                "duplicate_count": stats.get("duplicate_count"),
                "near_duplicate_count": stats.get("near_duplicate_count"),
                "antonym_contradiction_count": stats.get("antonym_contradiction_count"),
                "numeric_direction_conflict_count": stats.get("numeric_direction_conflict_count"),
                "semantic_opposition_count": stats.get("semantic_opposition_count"),
                "dropped_evidence_ids": stats.get("dropped_evidence_ids", []),
                "downweighted_evidence_ids": stats.get("downweighted_evidence_ids", []),
                "marked_conflict_evidence_ids": stats.get("marked_conflict_evidence_ids", []),
            }
            st.json(summary)
            rows = conflict_rows(trace)
            st.dataframe(rows, use_container_width=True) if rows else st.info("No conflicts detected.")

    with tabs[7]:
        stats = get_red_blue_stats(trace)
        if not stats:
            st.info("Red-Blue stats are not available for this run.")
        elif not stats.get("enabled"):
            st.info("Red-Blue repair is disabled.")
            st.json(stats)
        else:
            summary = {
                "enabled": stats.get("enabled"),
                "rounds_completed": stats.get("rounds_completed"),
                "initial_overall_score": stats.get("initial_overall_score"),
                "final_overall_score": stats.get("final_overall_score"),
                "score_delta": stats.get("score_delta"),
                "total_red_issues": stats.get("total_red_issues"),
                "total_patches_applied": stats.get("total_patches_applied"),
                "unresolved_high_severity_count": stats.get("unresolved_high_severity_count"),
                "stopped_reason": stats.get("stopped_reason"),
                "final_report_selected": stats.get("final_report_selected"),
            }
            st.json(summary)
            rows = red_blue_round_rows(trace)
            st.dataframe(rows, use_container_width=True) if rows else st.info("No Red-Blue rounds recorded.")

    with tabs[8]:
        stats = get_search_stats(trace)
        if not stats:
            st.info("Search stats are not available for this run.")
        elif not stats.get("enabled"):
            st.info("Web search is disabled.")
            st.json(stats)
        else:
            st.json(
                {
                    "enabled": stats.get("enabled"),
                    "provider": stats.get("provider"),
                    "query_count": stats.get("query_count"),
                    "result_count": stats.get("result_count"),
                    "fetched_document_count": stats.get("fetched_document_count"),
                    "validated_citation_count": stats.get("validated_citation_count"),
                    "supported_count": stats.get("supported_count"),
                    "partially_supported_count": stats.get("partially_supported_count"),
                    "unsupported_count": stats.get("unsupported_count"),
                    "unreachable_count": stats.get("unreachable_count"),
                    "no_source_count": stats.get("no_source_count"),
                    "search_queries": stats.get("search_queries", []),
                    "top_domains": stats.get("top_domains", {}),
                    "fallback_used": stats.get("fallback_used"),
                    "provider_errors": stats.get("provider_errors", []),
                    "api_base_host": stats.get("api_base_host"),
                    "search_timeout": stats.get("search_timeout"),
                    "search_provider_mode": stats.get("search_provider_mode"),
                }
            )

    with tabs[9]:
        st.code(
            "python scripts/run_benchmark.py --benchmark benchmarks/researchbench.jsonl --backend mock --limit 5 --mode dag",
            language="bash",
        )
        st.code(
            "python scripts/run_experiment_matrix.py --benchmark benchmarks/researchbench.jsonl --limit 5",
            language="bash",
        )
        summary_path = st.text_input("Summary JSON path", "eval_outputs/summary.json")
        if st.button("Load Evaluation Summary"):
            path = Path(summary_path)
            if not path.exists():
                st.warning("Summary file not found.")
            else:
                summary = json.loads(path.read_text(encoding="utf-8"))
                st.json(
                    {
                        "benchmark_name": summary.get("benchmark_name"),
                        "case_count": summary.get("case_count"),
                        "success_count": summary.get("success_count"),
                        "error_count": summary.get("error_count"),
                        "average_rule_overall": summary.get("average_rule_overall"),
                        "average_judge_overall": summary.get("average_judge_overall"),
                        "bootstrap_ci_rule_overall": summary.get("bootstrap_ci_rule_overall"),
                        "bootstrap_ci_judge_overall": summary.get("bootstrap_ci_judge_overall"),
                    }
                )


def _sanitize_error(message: str) -> str:
    return message.replace("Authorization", "[redacted]").replace("Bearer ", "Bearer [redacted]")


if __name__ == "__main__":
    main()
