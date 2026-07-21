"""Tests for consistent project environment loading."""

import os

from src.llm.config import load_config as load_llm_config
from src.utils.environment import get_project_env_path, load_project_environment


def test_loads_env_from_project_root(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("CURRENCY_ASSISTANT_TEST_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("CURRENCY_ASSISTANT_ROOT", str(tmp_path))
    monkeypatch.delenv("CURRENCY_ASSISTANT_ENV_FILE", raising=False)
    monkeypatch.delenv("CURRENCY_ASSISTANT_TEST_KEY", raising=False)

    assert get_project_env_path() == env_file.resolve()
    assert load_project_environment() is True
    assert os.environ["CURRENCY_ASSISTANT_TEST_KEY"] == "from-dotenv"


def test_existing_environment_variable_takes_priority(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("CURRENCY_ASSISTANT_TEST_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("CURRENCY_ASSISTANT_ROOT", str(tmp_path))
    monkeypatch.delenv("CURRENCY_ASSISTANT_ENV_FILE", raising=False)
    monkeypatch.setenv("CURRENCY_ASSISTANT_TEST_KEY", "from-shell")

    load_project_environment()

    assert os.environ["CURRENCY_ASSISTANT_TEST_KEY"] == "from-shell"


def test_supports_explicit_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / "development.env"
    env_file.write_text("CURRENCY_ASSISTANT_TEST_KEY=explicit\n", encoding="utf-8")
    monkeypatch.setenv("CURRENCY_ASSISTANT_ROOT", str(tmp_path))
    monkeypatch.setenv("CURRENCY_ASSISTANT_ENV_FILE", "development.env")
    monkeypatch.delenv("CURRENCY_ASSISTANT_TEST_KEY", raising=False)

    assert get_project_env_path() == env_file.resolve()
    assert load_project_environment() is True
    assert os.environ["CURRENCY_ASSISTANT_TEST_KEY"] == "explicit"


def test_llm_config_bootstraps_project_environment(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "CURRENCY_ASSISTANT_LLM_BOOTSTRAP_KEY=loaded\n",
        encoding="utf-8",
    )
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
llm:
  default_provider: openai
  providers:
    openai_main:
      model: test-model
      enabled: true
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("CURRENCY_ASSISTANT_ROOT", str(tmp_path))
    monkeypatch.delenv("CURRENCY_ASSISTANT_ENV_FILE", raising=False)
    monkeypatch.delenv("CURRENCY_ASSISTANT_LLM_BOOTSTRAP_KEY", raising=False)

    config = load_llm_config(str(config_file))

    assert config.default_provider == "openai"
    assert os.environ["CURRENCY_ASSISTANT_LLM_BOOTSTRAP_KEY"] == "loaded"
