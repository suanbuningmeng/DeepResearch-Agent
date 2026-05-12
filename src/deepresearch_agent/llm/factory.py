from __future__ import annotations

import os

from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.llm.deepseek_client import DeepSeekLLM
from deepresearch_agent.llm.mock_client import MockLLM
from deepresearch_agent.llm.openai_compatible_client import OpenAICompatibleLLM
from deepresearch_agent.llm.vllm_client import VLLMLLM


def create_llm(backend: str, **kwargs: object) -> BaseLLM:
    normalized_backend = backend.lower()
    if normalized_backend == "mock":
        return MockLLM()
    if normalized_backend == "openai-compatible":
        api_key_env = str(kwargs.get("api_key_env") or "OPENAI_COMPATIBLE_API_KEY")
        api_key = str(kwargs.get("api_key") or os.getenv(api_key_env, ""))
        api_base = str(kwargs.get("api_base") or os.getenv("OPENAI_COMPATIBLE_API_BASE", ""))
        model = str(kwargs.get("model") or os.getenv("OPENAI_COMPATIBLE_MODEL", ""))
        return OpenAICompatibleLLM(
            api_key=api_key,
            api_base=api_base,
            model=model,
            temperature=float(kwargs.get("temperature") or 0.2),
            max_tokens=int(kwargs.get("max_tokens") or 4096),
            request_timeout=int(kwargs.get("request_timeout") or 180),
            enable_thinking=_optional_bool(kwargs.get("enable_thinking")),
            extra_body=_optional_dict(kwargs.get("extra_body")),
        )
    if normalized_backend == "deepseek":
        api_key_env = str(kwargs.get("api_key_env") or "DEEPSEEK_API_KEY")
        return DeepSeekLLM(
            api_key=str(kwargs.get("api_key") or os.getenv(api_key_env, "")),
            api_base=str(kwargs.get("api_base") or os.getenv("DEEPSEEK_API_BASE", "")),
            model=str(kwargs.get("model") or os.getenv("DEEPSEEK_MODEL", "")),
            temperature=float(kwargs.get("temperature") or 0.2),
            max_tokens=int(kwargs.get("max_tokens") or 4096),
            request_timeout=int(kwargs.get("request_timeout") or 180),
            enable_thinking=_optional_bool(kwargs.get("enable_thinking")),
            extra_body=_optional_dict(kwargs.get("extra_body")),
        )
    if normalized_backend in {"vllm", "vllm-local"}:
        api_key_env = str(kwargs.get("api_key_env") or "VLLM_API_KEY")
        return VLLMLLM(
            api_key=str(kwargs.get("api_key") or os.getenv(api_key_env, "EMPTY")),
            api_base=str(kwargs.get("api_base") or os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")),
            model=str(kwargs.get("model") or os.getenv("VLLM_MODEL", "Qwen2.5-7B-Instruct")),
            temperature=float(kwargs.get("temperature") or 0.2),
            max_tokens=int(kwargs.get("max_tokens") or 4096),
            request_timeout=int(kwargs.get("request_timeout") or 180),
            enable_thinking=_optional_bool(kwargs.get("enable_thinking")),
            extra_body=_optional_dict(kwargs.get("extra_body")),
        )
    raise ValueError(
        "Unknown LLM backend. Expected one of: mock, openai-compatible, deepseek, vllm."
    )


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _optional_dict(value: object) -> dict | None:
    if isinstance(value, dict):
        return value
    return None
