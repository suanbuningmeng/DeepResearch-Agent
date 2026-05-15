from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import test_mimo_api


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict[str, Any]:
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeClient:
    def __init__(self, response: FakeResponse | None = None, exception: Exception | None = None) -> None:
        self.response = response
        self.exception = exception

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> FakeResponse:
        assert headers["Authorization"].startswith("Bearer ")
        assert "model" in json
        if self.exception:
            raise self.exception
        assert self.response is not None
        return self.response


def _patch_client(monkeypatch, response: FakeResponse | None = None, exception: Exception | None = None) -> None:
    monkeypatch.setattr(test_mimo_api.httpx, "Client", lambda timeout: FakeClient(response=response, exception=exception))


def test_missing_api_key_prints_clear_error(monkeypatch, capsys) -> None:
    monkeypatch.delenv("XIAOMI_MIMO_API_KEY", raising=False)

    exit_code = test_mimo_api.main()

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "XIAOMI_MIMO_API_KEY is not set." in output


def test_success_response_is_parsed_without_printing_api_key(monkeypatch, capsys) -> None:
    secret = "mimo_secret_test_key"
    monkeypatch.setenv("XIAOMI_MIMO_API_KEY", secret)
    monkeypatch.setenv("XIAOMI_MIMO_API_BASE", "https://api.example.test/v1")
    monkeypatch.setenv("XIAOMI_MIMO_MODEL", "xiaomi/test-model")
    _patch_client(
        monkeypatch,
        response=FakeResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": "Hello from MiMo."
                        }
                    }
                ]
            },
        ),
    )

    exit_code = test_mimo_api.main()

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "status_code: 200" in output
    assert "model: xiaomi/test-model" in output
    assert "Hello from MiMo." in output
    assert secret not in output


def test_401_prints_key_hint_without_leaking_key(monkeypatch, capsys) -> None:
    secret = "mimo_secret_401"
    monkeypatch.setenv("XIAOMI_MIMO_API_KEY", secret)
    _patch_client(
        monkeypatch,
        response=FakeResponse(401, {"error": {"message": f"invalid token {secret}"}}),
    )

    exit_code = test_mimo_api.main()

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "status_code: 401" in output
    assert "Check XIAOMI_MIMO_API_KEY" in output
    assert secret not in output
    assert "[redacted]" in output


def test_404_prints_base_or_model_hint(monkeypatch, capsys) -> None:
    monkeypatch.setenv("XIAOMI_MIMO_API_KEY", "mimo_secret_404")
    _patch_client(monkeypatch, response=FakeResponse(404, {"error": {"message": "model not found"}}))

    exit_code = test_mimo_api.main()

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "status_code: 404" in output
    assert "Check XIAOMI_MIMO_API_BASE and XIAOMI_MIMO_MODEL" in output


def test_timeout_prints_proxy_hint(monkeypatch, capsys) -> None:
    monkeypatch.setenv("XIAOMI_MIMO_API_KEY", "mimo_secret_timeout")
    _patch_client(monkeypatch, exception=httpx.TimeoutException("timeout"))

    exit_code = test_mimo_api.main()

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Request timed out." in output
    assert "HTTP_PROXY / HTTPS_PROXY / ALL_PROXY" in output
