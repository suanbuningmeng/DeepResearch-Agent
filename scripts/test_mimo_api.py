from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

DEFAULT_API_BASE = "https://api.xiaomimimo.com/v1"
DEFAULT_MODEL = "xiaomi/mimo-v2-flash"
PROMPT = "Say hello in one short sentence."


def _sanitize(text: str, api_key: str | None) -> str:
    if api_key:
        return text.replace(api_key, "[redacted]")
    return text


def _build_chat_url(api_base: str) -> str:
    return f"{api_base.rstrip('/')}/chat/completions"


def _extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"].strip()
    if isinstance(first.get("text"), str):
        return first["text"].strip()
    return ""


def _response_text(response: httpx.Response, api_key: str) -> str:
    try:
        payload = response.json()
        text = str(payload)
    except Exception:
        text = response.text
    return _sanitize(text, api_key)


def _print_status_hint(status_code: int) -> None:
    if status_code in {401, 403}:
        print("Hint: API key is invalid, expired, or does not have permission. Check XIAOMI_MIMO_API_KEY.")
    elif status_code == 404:
        print("Hint: API base, endpoint, or model may be wrong. Check XIAOMI_MIMO_API_BASE and XIAOMI_MIMO_MODEL.")
    elif status_code == 429:
        print("Hint: Rate limit or quota issue. Wait and retry, or check account quota.")
    elif status_code == 400:
        print("Hint: Payload parameters may be incompatible. Try keeping only model/messages/max_tokens.")
    elif status_code >= 500:
        print("Hint: Server-side error from the provider. Retry later or check provider status.")


def run_mimo_api_test() -> int:
    api_key = os.getenv("XIAOMI_MIMO_API_KEY")
    if not api_key:
        print("XIAOMI_MIMO_API_KEY is not set.")
        return 1

    api_base = os.getenv("XIAOMI_MIMO_API_BASE", DEFAULT_API_BASE)
    model = os.getenv("XIAOMI_MIMO_MODEL", DEFAULT_MODEL)
    url = _build_chat_url(api_base)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.2,
        "max_tokens": 128,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print("Testing Xiaomi MiMo OpenAI-compatible chat/completions endpoint...")
    print(f"api_base: {api_base}")
    print(f"model: {model}")

    try:
        with httpx.Client(timeout=60) as client:
            response = client.post(url, headers=headers, json=payload)
    except httpx.TimeoutException:
        print("Request timed out.")
        print("Hint: This is usually a network or proxy issue. Try setting or clearing HTTP_PROXY / HTTPS_PROXY / ALL_PROXY.")
        return 1
    except httpx.RequestError as exc:
        print(f"Request failed: {_sanitize(str(exc), api_key)}")
        print("Hint: Check network connectivity, API base, and proxy settings.")
        return 1

    print(f"status_code: {response.status_code}")

    if response.status_code >= 400:
        print(f"error: {_response_text(response, api_key)}")
        _print_status_hint(response.status_code)
        return 1

    try:
        data = response.json()
    except Exception:
        print(f"error: response is not valid JSON: {_sanitize(response.text, api_key)}")
        return 1

    content = _extract_content(data)
    if content:
        print(f"response: {content[:500]}")
    else:
        print("response: <no assistant content found>")
    return 0


def main() -> int:
    return run_mimo_api_test()


if __name__ == "__main__":
    raise SystemExit(main())
