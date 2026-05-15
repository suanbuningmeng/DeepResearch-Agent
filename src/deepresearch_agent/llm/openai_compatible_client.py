from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx

from deepresearch_agent.llm.base import BaseLLM


NO_THINKING_SYSTEM_MESSAGE = (
    "No-thinking mode is requested. Answer directly and briefly. "
    "Do not perform step-by-step reasoning. Do not include hidden reasoning, "
    "<think> blocks, analysis text, or explanations before the final answer. "
    "For structured tasks, return only the requested final JSON or final report."
)


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
        self.call_stats: list[dict[str, Any]] = []
        self.last_response_metadata: dict[str, Any] | None = None

    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        enable_thinking = kwargs.get("enable_thinking", self.enable_thinking)
        prompt_type = str(kwargs.get("prompt_type") or "unknown")
        requested_max_tokens = int(kwargs.get("max_tokens") or self.max_tokens)
        thinking_disabled_requested = enable_thinking is False
        payload: dict[str, Any] = {
            "model": str(kwargs.get("model") or self.model),
            "messages": _messages_for_prompt(prompt, thinking_disabled_requested),
            "temperature": float(kwargs.get("temperature") or self.temperature),
            "max_tokens": requested_max_tokens,
        }
        if enable_thinking is not None:
            _apply_thinking_control(
                payload,
                enable_thinking=bool(enable_thinking),
                api_base=self.api_base,
                model=self.model,
            )
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
        self._record_response_metadata(
            data=data,
            prompt_type=prompt_type,
            requested_max_tokens=requested_max_tokens,
            content=content,
            thinking_disabled_requested=thinking_disabled_requested,
            sent_thinking_controls=_sent_thinking_controls(payload),
        )
        return content

    def _record_response_metadata(
        self,
        data: dict[str, Any],
        prompt_type: str,
        requested_max_tokens: int,
        content: str,
        thinking_disabled_requested: bool,
        sent_thinking_controls: dict[str, Any],
    ) -> None:
        choices = data.get("choices")
        first_choice = choices[0] if isinstance(choices, list) and choices else {}
        finish_reason = first_choice.get("finish_reason") if isinstance(first_choice, dict) else None
        usage = data.get("usage")
        metadata = {
            "call_index": len(self.call_stats) + 1,
            "prompt_type": prompt_type,
            "model": self.model,
            "api_base_host": urlparse(self.api_base).netloc or None,
            "requested_max_tokens": requested_max_tokens,
            "thinking_disabled_requested": thinking_disabled_requested,
            "sent_thinking_controls": sent_thinking_controls,
            "finish_reason": finish_reason,
            "usage": usage if isinstance(usage, dict) else None,
            "content_chars": len(content),
            "content_ends_with_json_boundary": content.rstrip().endswith(("}", "]")),
        }
        self.last_response_metadata = metadata
        self.call_stats.append(metadata)


def _messages_for_prompt(prompt: str, thinking_disabled_requested: bool) -> list[dict[str, str]]:
    if not thinking_disabled_requested:
        return [{"role": "user", "content": prompt}]
    return [
        {"role": "system", "content": NO_THINKING_SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]


def _sent_thinking_controls(payload: dict[str, Any]) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    if "enable_thinking" in payload:
        controls["enable_thinking"] = payload["enable_thinking"]
    chat_template_kwargs = payload.get("chat_template_kwargs")
    if isinstance(chat_template_kwargs, dict) and "enable_thinking" in chat_template_kwargs:
        controls["chat_template_kwargs"] = {
            "enable_thinking": chat_template_kwargs["enable_thinking"]
        }
    thinking = payload.get("thinking")
    if isinstance(thinking, dict):
        controls["thinking"] = dict(thinking)
    return controls


def _apply_thinking_control(
    payload: dict[str, Any],
    enable_thinking: bool,
    api_base: str,
    model: str,
) -> None:
    if _is_mimo_token_plan(api_base, model):
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        return
    payload["enable_thinking"] = enable_thinking


def _is_mimo_token_plan(api_base: str, model: str) -> bool:
    text = f"{api_base} {model}".lower()
    return "xiaomimimo" in text or "mimo" in text
