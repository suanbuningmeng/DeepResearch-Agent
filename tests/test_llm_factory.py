import asyncio

import pytest

from deepresearch_agent.llm import MockLLM, OpenAICompatibleLLM, create_llm


def test_llm_factory_creates_mock_backend() -> None:
    llm = create_llm("mock")

    assert isinstance(llm, MockLLM)


def test_llm_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        create_llm("unknown-backend")


def test_llm_factory_passes_timeout_and_thinking_options() -> None:
    llm = create_llm(
        "openai-compatible",
        api_key="test-key",
        api_base="https://example.test/v1",
        model="test-model",
        request_timeout=222,
        enable_thinking=False,
    )

    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.request_timeout == 222
    assert llm.enable_thinking is False


def test_openai_compatible_records_finish_reason_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_headers = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {"content": '{"subtasks": []}'},
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 34,
                    "total_tokens": 46,
                },
            }

    class FakeAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
            captured_headers.update(headers)
            assert json["max_tokens"] == 123
            return FakeResponse()

    monkeypatch.setattr(
        "deepresearch_agent.llm.openai_compatible_client.httpx.AsyncClient",
        FakeAsyncClient,
    )
    llm = OpenAICompatibleLLM(
        api_key="secret-test-key",
        api_base="https://example.test/v1",
        model="test-model",
        max_tokens=123,
    )

    content = asyncio.run(llm.agenerate("hello", prompt_type="planner"))

    assert content == '{"subtasks": []}'
    assert llm.last_response_metadata is not None
    assert llm.last_response_metadata["prompt_type"] == "planner"
    assert llm.last_response_metadata["finish_reason"] == "length"
    assert llm.last_response_metadata["usage"]["completion_tokens"] == 34
    assert llm.last_response_metadata["api_base_host"] == "example.test"
    assert llm.last_response_metadata["content_ends_with_json_boundary"] is True
    assert "secret-test-key" not in str(llm.call_stats)
    assert captured_headers["Authorization"] == "Bearer secret-test-key"


def test_openai_compatible_disable_thinking_adds_direct_system_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payload = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {"content": "{}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2},
            }

    class FakeAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
            captured_payload.update(json)
            return FakeResponse()

    monkeypatch.setattr(
        "deepresearch_agent.llm.openai_compatible_client.httpx.AsyncClient",
        FakeAsyncClient,
    )
    llm = OpenAICompatibleLLM(
        api_key="secret-test-key",
        api_base="https://example.test/v1",
        model="test-model",
        enable_thinking=False,
    )

    asyncio.run(llm.agenerate("return json", prompt_type="planner"))

    assert captured_payload["enable_thinking"] is False
    assert captured_payload["messages"][0]["role"] == "system"
    assert "No-thinking mode is requested" in captured_payload["messages"][0]["content"]
    assert captured_payload["messages"][1] == {"role": "user", "content": "return json"}
    assert llm.last_response_metadata is not None
    assert llm.last_response_metadata["thinking_disabled_requested"] is True
    assert llm.last_response_metadata["sent_thinking_controls"] == {"enable_thinking": False}
