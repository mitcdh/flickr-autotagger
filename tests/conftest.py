from __future__ import annotations

from dataclasses import replace

import pytest

from autotagger.config import Settings


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path):
    for name in (
        "FLICKR_API_KEY",
        "FLICKR_API_SECRET",
        "FLICKR_OAUTH_TOKEN",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "DESCRIPTIONS_TO_ANALYZE",
        "SKIP_PREFIX",
        "FLICKR_PRIVACY_FILTER",
    ):
        monkeypatch.delenv(name, raising=False)
    value = Settings.from_env()
    return replace(
        value,
        flickr_api_key="flickr-key",
        flickr_api_secret="flickr-secret",
        openai_api_key="openai-key",
        checkpoint_file=tmp_path / "checkpoint.jsonl",
        updated_metadata_file=tmp_path / "summary.json",
        resume=False,
    )
