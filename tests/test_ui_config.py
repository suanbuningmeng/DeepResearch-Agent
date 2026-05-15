from deepresearch_agent.ui.config import DEFAULT_QUESTION, DemoRunConfig


def test_demo_run_config_default_mock_config() -> None:
    config = DemoRunConfig()

    assert config.question == DEFAULT_QUESTION
    assert config.backend == "mock"
    assert config.mode == "dag"
    assert config.validate() == []


def test_demo_run_config_reports_missing_api_key_env(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_API_KEY", raising=False)
    config = DemoRunConfig(
        backend="openai-compatible",
        api_base="https://api.example.com/v1",
        api_key_env="MISSING_API_KEY",
        model="test-model",
    )

    errors = config.validate()

    assert "Environment variable MISSING_API_KEY is not set." in errors
