from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from autotagger.analysis import (
    AnalysisValidationError,
    OpenAIImageAnalyzer,
    parse_analysis_response,
    render_prompt,
)
from autotagger.models import PhotoCandidate, Usage


def test_parse_analysis_normalises_and_limits_keywords():
    result = parse_analysis_response(
        '```json\n{"title":" Beach ","description":" Waves ",'
        '"keywords":["Golden Hour", "SEA!", "sea", "sand"]}\n```',
        max_keywords=3,
        max_title_chars=100,
        max_description_chars=100,
        usage=Usage(10, 5, 0.1),
    )
    assert result.title == "Beach"
    assert result.description == "Waves"
    assert result.keywords == ("golden-hour", "sea", "sand")
    assert result.usage.cost == 0.1


@pytest.mark.parametrize(
    "payload",
    (
        "not json",
        "[]",
        '{"title":"","description":"ok","keywords":["tag"]}',
        '{"title":"ok","description":"ok","keywords":"tag"}',
    ),
)
def test_parse_analysis_rejects_invalid_schema(payload):
    with pytest.raises(AnalysisValidationError):
        parse_analysis_response(
            payload,
            max_keywords=10,
            max_title_chars=100,
            max_description_chars=100,
        )


def test_prompt_file_tokens_are_replaced(settings, tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("{{LANGUAGE}} {{MAX_KEYWORDS}} {{DESCRIPTION_STYLE}}")
    custom = replace(settings, prompt_file=prompt, language="French", max_keywords=7)
    assert render_prompt(custom).startswith("French 7")


def test_custom_base_url_is_passed_to_client(settings, monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("autotagger.analysis.AsyncOpenAI", FakeClient)
    OpenAIImageAnalyzer(replace(settings, openai_base_url="http://localhost:11434/v1"))
    assert captured["base_url"] == "http://localhost:11434/v1"


def test_azure_provider_uses_dedicated_endpoint_and_api_version(settings, monkeypatch):
    captured = {}

    class FakeAzureClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("autotagger.analysis.AsyncAzureOpenAI", FakeAzureClient)
    OpenAIImageAnalyzer(
        replace(
            settings,
            openai_provider="azure",
            azure_openai_endpoint="https://example.openai.azure.com",
            azure_openai_api_version="2026-01-01",
        )
    )
    assert captured["azure_endpoint"] == "https://example.openai.azure.com"
    assert captured["api_version"] == "2026-01-01"


def test_analyze_builds_configurable_request(settings):
    captured = {}

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"title":"Title","description":"Description",'
                            '"keywords":["one"]}'
                        )
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    analyzer = OpenAIImageAnalyzer(settings, client=client)
    photo = PhotoCandidate(
        id="1",
        photoset_id="set",
        photoset_title="Album",
        photoset_description="Context",
        image_url="https://example.test/photo.jpg",
    )
    result = __import__("asyncio").run(analyzer.analyze(photo))
    assert captured["max_completion_tokens"] == settings.openai_max_output_tokens
    assert captured["response_format"] == {"type": "json_object"}
    assert result.usage.prompt_tokens == 100
