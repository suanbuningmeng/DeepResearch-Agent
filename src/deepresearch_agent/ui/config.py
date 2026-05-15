from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_QUESTION = "What are the main challenges and recent methods for long-context LLM evaluation?"


@dataclass
class DemoRunConfig:
    question: str = DEFAULT_QUESTION
    backend: str = "mock"
    mode: str = "dag"
    output_dir: Path = Path("outputs")
    run_id: str | None = None
    api_base: str | None = None
    api_key_env: str | None = None
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 768
    request_timeout: int = 180
    enable_thinking: bool | None = None
    max_concurrency: int = 1
    global_timeout_seconds: int | None = 300
    writer_top_k_per_task: int = 2
    memory_backend: str = "memory"
    memory_db_path: str = "data/memory.sqlite"
    vector_index_path: str = "data/vector_index.npz"
    enable_memory_retrieval: bool = True
    memory_search_top_k: int = 10
    clear_memory: bool = True
    enable_context_compression: bool = True
    compression_l1_top_n: int = 12
    compression_l2_top_k: int = 8
    enable_conflict_detection: bool = False
    near_duplicate_threshold: float = 0.92
    semantic_opposition_threshold: float = 0.65
    enable_red_blue: bool = False
    red_blue_max_rounds: int = 2
    red_blue_min_score_delta: int = 1
    red_blue_stop_on_no_high_severity: bool = True
    enable_web_search: bool = False
    search_provider: str = "mock"
    search_top_k: int = 3
    max_search_queries: int = 3
    allow_network_fetch: bool = False
    search_timeout: int = 20
    search_api_key_env: str = "SEARCH_API_KEY"
    search_api_base: str = ""
    search_provider_name: str = "mock"
    enable_replan: bool = True
    failure_scenario: str = "none"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.backend in {"openai-compatible", "deepseek", "vllm"}:
            if not self.api_key_env:
                errors.append("api_key_env is required for real LLM backends.")
            elif not os.getenv(self.api_key_env):
                errors.append(f"Environment variable {self.api_key_env} is not set.")
            if self.backend == "openai-compatible" and not self.api_base:
                errors.append("api_base is required for openai-compatible backend.")
            if self.backend in {"openai-compatible", "deepseek", "vllm"} and not self.model:
                errors.append("model is required for real LLM backends.")
        return errors
