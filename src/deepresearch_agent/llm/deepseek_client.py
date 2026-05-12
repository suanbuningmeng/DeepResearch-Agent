from __future__ import annotations

import os

from deepresearch_agent.llm.openai_compatible_client import OpenAICompatibleLLM


class DeepSeekLLM(OpenAICompatibleLLM):
    """DeepSeek chat client using OpenAI-compatible request format."""

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
        resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        resolved_api_base = api_base or os.getenv("DEEPSEEK_API_BASE", "")
        resolved_model = model or os.getenv("DEEPSEEK_MODEL", "")
        if not resolved_api_base:
            raise ValueError("DEEPSEEK_API_BASE is required for deepseek backend.")
        if not resolved_model:
            raise ValueError("DEEPSEEK_MODEL or --model is required for deepseek backend.")
        super().__init__(
            api_key=resolved_api_key,
            api_base=resolved_api_base,
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            request_timeout=request_timeout,
            enable_thinking=enable_thinking,
            extra_body=extra_body,
        )
