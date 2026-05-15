from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx


DEFAULT_API_BASE = "https://api.xiaomimimo.com/v1"
DEFAULT_MODEL = "xiaomi/mimo-v2-flash"
PROMPT = 'Return JSON only: {"ok": true, "message": "hello"}'
EXPECTED = {"ok": True, "message": "hello"}
OUTPUT_PATH = Path("outputs") / "mimo_thinking_probe.json"


def _sanitize(text: str, api_key: str | None) -> str:
    if api_key:
        return text.replace(api_key, "[redacted]")
    return text


def _chat_url(api_base: str) -> str:
    return f"{api_base.rstrip('/')}/chat/completions"


def _base_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.1,
        "max_tokens": 256,
    }


def _variant_payloads(model: str) -> list[tuple[str, dict[str, Any]]]:
    variants: list[tuple[str, dict[str, Any]]] = []

    baseline = _base_payload(model)
    variants.append(("baseline_no_control", baseline))

    top_level = _base_payload(model)
    top_level["enable_thinking"] = False
    variants.append(("top_level_enable_thinking_false", top_level))

    chat_template = _base_payload(model)
    chat_template["chat_template_kwargs"] = {"enable_thinking": False}
    variants.append(("chat_template_kwargs_enable_thinking_false", chat_template))

    thinking_disabled = _base_payload(model)
    thinking_disabled["thinking"] = {"type": "disabled"}
    variants.append(("thinking_type_disabled", thinking_disabled))

    thinking_enabled = _base_payload(model)
    thinking_enabled["thinking"] = {"type": "enabled"}
    variants.append(("thinking_type_enabled", thinking_enabled))

    reasoning_disabled = _base_payload(model)
    reasoning_disabled["reasoning"] = {"enabled": False}
    variants.append(("reasoning_enabled_false", reasoning_disabled))

    assistant_prefix = _base_payload(model)
    assistant_prefix["messages"] = [
        {"role": "user", "content": PROMPT},
        {"role": "assistant", "content": "<think></think>"},
    ]
    variants.append(("assistant_prefix_empty_think", assistant_prefix))

    return variants


def _extract_choice(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _extract_message(choice: dict[str, Any]) -> dict[str, Any]:
    message = choice.get("message")
    if isinstance(message, dict):
        return message
    return {}


def _content_json_status(content: str) -> tuple[bool, bool]:
    stripped = content.strip()
    try:
        parsed = json.loads(stripped)
    except Exception:
        parsed = None
    parsed_ok = isinstance(parsed, dict)
    complete = parsed == EXPECTED
    if not complete:
        compact = stripped.replace(" ", "")
        complete = '"ok":true' in compact.lower() and '"message":"hello"' in compact.lower()
    return parsed_ok, complete


def _error_text(response: httpx.Response, api_key: str) -> str:
    try:
        text = json.dumps(response.json(), ensure_ascii=False)
    except Exception:
        text = response.text
    return _sanitize(text[:1000], api_key)


def _probe_variant(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    name: str,
    payload: dict[str, Any],
    api_key: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "variant": name,
        "status_code": None,
        "error": None,
        "finish_reason": None,
        "content_preview": "",
        "content_length": 0,
        "has_reasoning_content": False,
        "reasoning_content_length": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "reasoning_tokens": None,
        "content_contains_think": False,
        "content_json_parse_success": False,
        "content_complete_expected_json": False,
    }
    try:
        response = client.post(url, headers=headers, json=payload)
    except httpx.TimeoutException as exc:
        result["error"] = f"timeout: {_sanitize(str(exc), api_key)}"
        return result
    except httpx.RequestError as exc:
        result["error"] = f"request_error: {_sanitize(str(exc), api_key)}"
        return result

    result["status_code"] = response.status_code
    if response.status_code >= 400:
        result["error"] = _error_text(response, api_key)
        return result

    try:
        data = response.json()
    except Exception:
        result["error"] = _sanitize(f"response was not valid JSON: {response.text[:500]}", api_key)
        return result

    choice = _extract_choice(data)
    message = _extract_message(choice)
    content = message.get("content") if isinstance(message.get("content"), str) else ""
    reasoning_content = message.get("reasoning_content")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    completion_details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
    parsed_ok, complete = _content_json_status(content)

    result.update(
        {
            "finish_reason": choice.get("finish_reason"),
            "content_preview": content[:300],
            "content_length": len(content),
            "has_reasoning_content": isinstance(reasoning_content, str) and bool(reasoning_content),
            "reasoning_content_length": len(reasoning_content) if isinstance(reasoning_content, str) else 0,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "reasoning_tokens": completion_details.get("reasoning_tokens"),
            "content_contains_think": "<think" in content.lower(),
            "content_json_parse_success": parsed_ok,
            "content_complete_expected_json": complete,
        }
    )
    return result


def _print_result(result: dict[str, Any]) -> None:
    print("=" * 80)
    print(f"variant: {result['variant']}")
    print(f"status_code: {result['status_code']}")
    if result.get("error"):
        print(f"error: {result['error']}")
    print(f"finish_reason: {result['finish_reason']}")
    print(f"content_preview: {result['content_preview']}")
    print(f"content_length: {result['content_length']}")
    print(f"has_reasoning_content: {result['has_reasoning_content']}")
    print(f"reasoning_content_length: {result['reasoning_content_length']}")
    print(f"prompt_tokens: {result['prompt_tokens']}")
    print(f"completion_tokens: {result['completion_tokens']}")
    print(f"total_tokens: {result['total_tokens']}")
    print(f"reasoning_tokens: {result['reasoning_tokens']}")
    print(f"content_contains_think: {result['content_contains_think']}")
    print(f"content_json_parse_success: {result['content_json_parse_success']}")
    print(f"content_complete_expected_json: {result['content_complete_expected_json']}")


def _conclusion(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in results if item.get("status_code") == 200]
    rejected = [item["variant"] for item in results if item.get("status_code") and int(item["status_code"]) >= 400]
    with_reasoning = [
        item
        for item in successful
        if isinstance(item.get("reasoning_tokens"), int)
    ]
    best_reasoning = min(with_reasoning, key=lambda item: int(item["reasoning_tokens"])) if with_reasoning else None
    json_stable = [
        item["variant"]
        for item in successful
        if item.get("content_json_parse_success") and item.get("content_complete_expected_json")
    ]
    baseline = next((item for item in results if item["variant"] == "baseline_no_control"), None)
    baseline_reasoning = baseline.get("reasoning_tokens") if isinstance(baseline, dict) else None
    clearly_disabled = False
    if best_reasoning and isinstance(best_reasoning.get("reasoning_tokens"), int):
        best_value = int(best_reasoning["reasoning_tokens"])
        if isinstance(baseline_reasoning, int):
            clearly_disabled = best_value <= max(10, int(baseline_reasoning * 0.25))
        else:
            clearly_disabled = best_value <= 10

    message = "At least one variant appears to reduce reasoning tokens substantially." if clearly_disabled else (
        "MiMo Token Plan may ignore client-side thinking controls or still account internal reasoning tokens."
    )
    return {
        "lowest_reasoning_tokens_variant": best_reasoning["variant"] if best_reasoning else None,
        "lowest_reasoning_tokens": best_reasoning.get("reasoning_tokens") if best_reasoning else None,
        "json_stable_variants": json_stable,
        "rejected_variants": rejected,
        "any_variant_clearly_disabled_reasoning": clearly_disabled,
        "message": message,
    }


def main() -> int:
    api_key = os.getenv("XIAOMI_MIMO_API_KEY")
    if not api_key:
        print("XIAOMI_MIMO_API_KEY is not set.")
        return 1

    api_base = os.getenv("XIAOMI_MIMO_API_BASE", DEFAULT_API_BASE)
    model = os.getenv("XIAOMI_MIMO_MODEL", DEFAULT_MODEL)
    url = _chat_url(api_base)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    print("Testing Xiaomi MiMo thinking / reasoning control variants...")
    print(f"api_base: {api_base}")
    print(f"model: {model}")
    print(f"output: {OUTPUT_PATH}")

    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=60) as client:
        for index, (name, payload) in enumerate(_variant_payloads(model), start=1):
            if index > 1:
                time.sleep(1)
            result = _probe_variant(client, url, headers, name, payload, api_key)
            results.append(result)
            _print_result(result)

    summary = _conclusion(results)
    payload = {
        "api_base": api_base,
        "model": model,
        "prompt": PROMPT,
        "results": results,
        "summary": summary,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
