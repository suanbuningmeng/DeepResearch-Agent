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
