from __future__ import annotations

from dataclasses import replace

import pytest

from autotagger.config import (
    DEFAULT_OPENAI_BASE_URL,
    ConfigurationError,
    Settings,
)


def test_base_url_defaults_when_environment_is_missing_or_empty(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert Settings.from_env().openai_base_url == DEFAULT_OPENAI_BASE_URL
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    assert Settings.from_env().openai_base_url == DEFAULT_OPENAI_BASE_URL


def test_existing_github_secrets_are_sufficient(monkeypatch):
    credentials = {
        "FLICKR_API_KEY": "flickr-key",
        "FLICKR_API_SECRET": "flickr-secret",
        "FLICKR_OAUTH_TOKEN": '{"oauth_token": "existing-token"}',
        "OPENAI_API_KEY": "openai-key",
    }
    for name, value in credentials.items():
        monkeypatch.setenv(name, value)

    # Missing GitHub repository variables are exposed to the job as empty strings.
    for name in (
        "OPENAI_BASE_URL",
        "AZURE_OPENAI_ENDPOINT",
        "OPENAI_API_VERSION",
    ):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("OPENAI_PROVIDER", "openai")

    settings = Settings.from_env()
    settings.validate(require_openai=True, require_flickr=True)

    assert settings.openai_base_url == DEFAULT_OPENAI_BASE_URL
    assert settings.openai_model == "gpt-5-mini"
    assert settings.openai_api_key == "openai-key"


def test_custom_base_url_is_loaded(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    assert Settings.from_env().openai_base_url == "http://localhost:11434/v1"


def test_full_chat_completions_endpoint_is_rejected(settings):
    with pytest.raises(ConfigurationError, match="base URL"):
        replace(
            settings,
            openai_base_url="https://api.example.test/v1/chat/completions",
        ).validate(require_openai=True, require_flickr=True)


def test_azure_provider_prefers_azure_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_PROVIDER", "azure")
    monkeypatch.setenv("OPENAI_API_KEY", "ordinary-key")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("OPENAI_API_VERSION", "2026-01-01")
    assert Settings.from_env().openai_api_key == "azure-key"


def test_json_lists_are_parsed_without_eval(monkeypatch):
    monkeypatch.setenv("SKIP_PREFIX", '["!", "private-"]')
    assert Settings.from_env().skip_prefixes == ("!", "private-")


def test_invalid_json_list_is_rejected(monkeypatch):
    monkeypatch.setenv("SKIP_PREFIX", '[__import__("os").getcwd()]')
    with pytest.raises(ConfigurationError, match="valid JSON list"):
        Settings.from_env()


def test_privacy_filter_all_omits_filter(monkeypatch):
    monkeypatch.setenv("FLICKR_PRIVACY_FILTER", "all")
    assert Settings.from_env().flickr_privacy_filter is None


def test_invalid_update_field_is_rejected(settings):
    with pytest.raises(ConfigurationError, match="UPDATE_FIELDS"):
        replace(settings, update_fields=("title", "permissions")).validate(
            require_openai=True, require_flickr=True
        )


def test_required_credentials_are_validated(settings):
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        replace(settings, openai_api_key="").validate(require_openai=True, require_flickr=True)
