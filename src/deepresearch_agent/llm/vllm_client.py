from __future__ import annotations

import os

from deepresearch_agent.llm.openai_compatible_client import OpenAICompatibleLLM


class VLLMLLM(OpenAICompatibleLLM):
    """Local vLLM client using OpenAI-compatible request format."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: int = 60,
        request_timeout: int = 180,
        enable_thinking: bool | None = None,
        extra_body: dict | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key or os.getenv("VLLM_API_KEY", "EMPTY"),
            api_base=api_base or os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"),
            model=model or os.getenv("VLLM_MODEL", "Qwen2.5-7B-Instruct"),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            request_timeout=request_timeout,
            enable_thinking=enable_thinking,
            extra_body=extra_body,
        )
