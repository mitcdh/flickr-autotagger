"""Configuration loading and validation.

CLI values are applied by ``autotagger.cli`` after this module has loaded the
environment. Keeping validation here prevents network clients from being
constructed with partly valid configuration.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
VALID_UPDATE_FIELDS = frozenset({"title", "description", "tags"})


class ConfigurationError(ValueError):
    """Raised when configuration cannot be parsed or validated."""


def _string(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _integer(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = _string(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    return value


def _optional_integer(name: str) -> int | None:
    raw = _string(name)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value


def _float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = _string(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    return value


def _optional_float(name: str) -> float | None:
    raw = _string(name)
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = _string(name, "true" if default else "false").lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true or false, got {raw!r}")


def _list(name: str, default: Iterable[str] = ()) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return tuple(default)
    raw = raw.strip()
    if raw.startswith("["):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"{name} must contain a valid JSON list") from exc
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ConfigurationError(f"{name} must be a JSON list of strings")
        return tuple(item.strip() for item in value if item.strip())
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _privacy_filter() -> int | None:
    raw = _string("FLICKR_PRIVACY_FILTER", "1").lower()
    if raw in {"", "all", "none", "0"}:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(
            "FLICKR_PRIVACY_FILTER must be all or an integer from 1 to 5"
        ) from exc
    if value not in range(1, 6):
        raise ConfigurationError("FLICKR_PRIVACY_FILTER must be all or an integer from 1 to 5")
    return value


@dataclass(frozen=True)
class Settings:
    flickr_api_key: str
    flickr_api_secret: str
    flickr_oauth_token: str
    flickr_token_file: Path
    flickr_privacy_filter: int | None
    flickr_image_url: str
    flickr_photoset_ids: tuple[str, ...]
    photoset_allowlist: tuple[str, ...]
    photoset_denylist: tuple[str, ...]
    skip_prefixes: tuple[str, ...]
    descriptions_to_analyze: tuple[str, ...]
    description_policy: str
    flickr_max_retries: int
    flickr_retry_backoff: float

    openai_provider: str
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    openai_timeout: float
    openai_max_retries: int
    openai_instruction_role: str
    openai_tokens_parameter: str
    openai_json_mode: bool
    openai_max_output_tokens: int
    openai_image_detail: str
    openai_cost_per_1m_prompt_tokens: float
    openai_cost_per_1m_completion_tokens: float
    openai_vision_cost_per_image: float
    azure_openai_endpoint: str
    azure_openai_api_version: str

    analysis_concurrency: int
    max_keywords: int
    max_title_chars: int
    max_description_chars: int
    language: str
    description_style: str
    prompt_file: Path | None
    max_photos: int | None
    max_total_cost: float | None

    update_fields: tuple[str, ...]
    tag_mode: str
    checkpoint_file: Path
    updated_metadata_file: Path
    resume: bool
    strict: bool

    @classmethod
    def from_env(cls) -> Settings:
        photoset_ids = list(_list("FLICKR_PHOTOSET_IDS"))
        legacy_photoset_id = _string("FLICKR_PHOTOSET_ID")
        if legacy_photoset_id and legacy_photoset_id not in photoset_ids:
            photoset_ids.append(legacy_photoset_id)

        prompt_file_raw = _string("PROMPT_FILE")
        openai_provider = _string("OPENAI_PROVIDER", "openai").lower()
        openai_api_key = (
            _string("AZURE_OPENAI_API_KEY")
            if openai_provider == "azure"
            else _string("OPENAI_API_KEY")
        )
        settings = cls(
            flickr_api_key=_string("FLICKR_API_KEY"),
            flickr_api_secret=_string("FLICKR_API_SECRET"),
            flickr_oauth_token=_string("FLICKR_OAUTH_TOKEN"),
            flickr_token_file=Path(_string("FLICKR_TOKEN_FILE", "flickr_token.json")),
            flickr_privacy_filter=_privacy_filter(),
            flickr_image_url=_string("FLICKR_IMAGE_URL", "url_m"),
            flickr_photoset_ids=tuple(photoset_ids),
            photoset_allowlist=_list("PHOTOSET_ALLOWLIST"),
            photoset_denylist=_list("PHOTOSET_DENYLIST"),
            skip_prefixes=_list("SKIP_PREFIX", ("#", "@")),
            descriptions_to_analyze=_list(
                "DESCRIPTIONS_TO_ANALYZE",
                ("OLYMPUS DIGITAL CAMERA", "Untitled", "DSC_", "IMG_", "DCIM"),
            ),
            description_policy=_string("DESCRIPTION_POLICY", "missing-or-placeholder").lower(),
            flickr_max_retries=_integer("FLICKR_MAX_RETRIES", 3, minimum=0),
            flickr_retry_backoff=_float("FLICKR_RETRY_BACKOFF", 1.0, minimum=0),
            openai_provider=openai_provider,
            openai_api_key=openai_api_key,
            openai_base_url=_string("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL,
            openai_model=_string("OPENAI_MODEL", "gpt-5-mini"),
            openai_timeout=_float("OPENAI_TIMEOUT", 60.0, minimum=0.1),
            openai_max_retries=_integer("OPENAI_MAX_RETRIES", 2, minimum=0),
            openai_instruction_role=_string("OPENAI_INSTRUCTION_ROLE", "developer").lower(),
            openai_tokens_parameter=_string(
                "OPENAI_TOKENS_PARAMETER", "max_completion_tokens"
            ).lower(),
            openai_json_mode=_boolean("OPENAI_JSON_MODE", True),
            openai_max_output_tokens=_integer("OPENAI_MAX_OUTPUT_TOKENS", 800, minimum=1),
            openai_image_detail=_string("OPENAI_IMAGE_DETAIL", "low").lower(),
            openai_cost_per_1m_prompt_tokens=_float(
                "OPENAI_COST_PER_1M_PROMPT_TOKEN", 0.25, minimum=0
            ),
            openai_cost_per_1m_completion_tokens=_float(
                "OPENAI_COST_PER_1M_COMPLETION_TOKEN", 2.0, minimum=0
            ),
            openai_vision_cost_per_image=_float("OPENAI_VISION_COST_PER_IMAGE", 0.0, minimum=0),
            azure_openai_endpoint=_string("AZURE_OPENAI_ENDPOINT"),
            azure_openai_api_version=_string("OPENAI_API_VERSION"),
            analysis_concurrency=_integer("ANALYSIS_CONCURRENCY", 3, minimum=1),
            max_keywords=_integer("MAX_KEYWORDS", 10, minimum=1),
            max_title_chars=_integer("MAX_TITLE_CHARS", 120, minimum=1),
            max_description_chars=_integer("MAX_DESCRIPTION_CHARS", 1200, minimum=1),
            language=_string("METADATA_LANGUAGE", "English"),
            description_style=_string("DESCRIPTION_STYLE", "detailed, factual, and concise"),
            prompt_file=Path(prompt_file_raw) if prompt_file_raw else None,
            max_photos=_optional_integer("MAX_PHOTOS"),
            max_total_cost=_optional_float("MAX_TOTAL_COST"),
            update_fields=_list("UPDATE_FIELDS", ("title", "description", "tags")),
            tag_mode=_string("TAG_MODE", "merge").lower(),
            checkpoint_file=Path(_string("CHECKPOINT_FILE", ".autotagger-checkpoint.jsonl")),
            updated_metadata_file=Path(_string("UPDATED_METADATA_FILE", "updated_metadata.json")),
            resume=_boolean("RESUME", True),
            strict=_boolean("FAIL_ON_ERROR", False),
        )
        settings.validate(require_openai=False, require_flickr=False)
        return settings

    def validate(self, *, require_openai: bool, require_flickr: bool) -> None:
        errors: list[str] = []
        if require_flickr:
            if not self.flickr_api_key:
                errors.append("FLICKR_API_KEY is required")
            if not self.flickr_api_secret:
                errors.append("FLICKR_API_SECRET is required")
        if require_openai and not self.openai_api_key:
            key_name = (
                "AZURE_OPENAI_API_KEY" if self.openai_provider == "azure" else "OPENAI_API_KEY"
            )
            errors.append(f"{key_name} is required")

        parsed_url = urlparse(self.openai_base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            errors.append("OPENAI_BASE_URL must be an absolute HTTP(S) URL")
        elif parsed_url.username or parsed_url.password:
            errors.append("OPENAI_BASE_URL must not contain embedded credentials")
        elif parsed_url.path.rstrip("/").endswith("/chat/completions"):
            errors.append("OPENAI_BASE_URL must be a base URL, not the chat completions endpoint")
        if self.openai_provider not in {"openai", "azure"}:
            errors.append("OPENAI_PROVIDER must be openai or azure")
        if self.openai_provider == "azure":
            if not self.azure_openai_endpoint:
                errors.append("AZURE_OPENAI_ENDPOINT is required for the Azure provider")
            if not self.azure_openai_api_version:
                errors.append("OPENAI_API_VERSION is required for the Azure provider")
        if self.openai_instruction_role not in {"developer", "system"}:
            errors.append("OPENAI_INSTRUCTION_ROLE must be developer or system")
        if self.openai_tokens_parameter not in {"max_completion_tokens", "max_tokens"}:
            errors.append("OPENAI_TOKENS_PARAMETER must be max_completion_tokens or max_tokens")
        if self.openai_image_detail not in {"low", "high", "auto"}:
            errors.append("OPENAI_IMAGE_DETAIL must be low, high, or auto")
        if self.description_policy not in {"missing-or-placeholder", "always", "never"}:
            errors.append("DESCRIPTION_POLICY must be missing-or-placeholder, always, or never")
        invalid_fields = set(self.update_fields) - VALID_UPDATE_FIELDS
        if invalid_fields or not self.update_fields:
            errors.append("UPDATE_FIELDS must contain one or more of title, description, and tags")
        if self.tag_mode not in {"merge", "replace"}:
            errors.append("TAG_MODE must be merge or replace")
        if not re.fullmatch(r"url_[a-z]+", self.flickr_image_url):
            errors.append("FLICKR_IMAGE_URL must be a Flickr URL extra such as url_m")
        if self.prompt_file and not self.prompt_file.is_file():
            errors.append(f"PROMPT_FILE does not exist: {self.prompt_file}")
        if errors:
            raise ConfigurationError("; ".join(errors))
