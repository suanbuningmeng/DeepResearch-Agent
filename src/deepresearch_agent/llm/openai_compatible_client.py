from __future__ import annotations

from typing import Any

import httpx

from deepresearch_agent.llm.base import BaseLLM


class OpenAICompatibleLLM(BaseLLM):
    """Async client for OpenAI-compatible chat completions APIs."""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: int = 60,
        request_timeout: int = 180,
        enable_thinking: bool | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for openai-compatible backend.")
        if not api_base:
            raise ValueError("api_base is required for openai-compatible backend.")
        if not model:
            raise ValueError("model is required for openai-compatible backend.")

        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.request_timeout = request_timeout
        self.enable_thinking = enable_thinking
        self.extra_body = extra_body or {}

    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        enable_thinking = kwargs.get("enable_thinking", self.enable_thinking)
        payload: dict[str, Any] = {
            "model": str(kwargs.get("model") or self.model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(kwargs.get("temperature") or self.temperature),
            "max_tokens": int(kwargs.get("max_tokens") or self.max_tokens),
        }
        if enable_thinking is not None:
            payload["enable_thinking"] = bool(enable_thinking)
        extra_body = kwargs.get("extra_body") or self.extra_body
        if isinstance(extra_body, dict) and extra_body:
            payload.update(extra_body)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.api_base}/chat/completions"
        timeout = httpx.Timeout(connect=20, read=self.request_timeout, write=20, pool=20)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            raise RuntimeError(
                "LLM request failed "
                f"(api_base={self.api_base}, model={self.model}, request_timeout={self.request_timeout}) "
                f"with HTTP {exc.response.status_code}: {body}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "LLM request failed "
                f"(api_base={self.api_base}, model={self.model}, request_timeout={self.request_timeout}): {exc}"
            ) from exc
        except ValueError as exc:
            raise RuntimeError("LLM response was not valid JSON.") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"LLM response missing choices[0].message.content: {data}") from exc
        if not isinstance(content, str):
            raise RuntimeError("LLM response content is not a string.")
        return content
