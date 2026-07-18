"""OpenAI-backed image analysis with strict local response validation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from openai import AsyncAzureOpenAI, AsyncOpenAI

from .config import Settings
from .models import Analysis, PhotoCandidate, Usage


class AnalysisValidationError(ValueError):
    """Raised when a model response is not usable as Flickr metadata."""


class ImageAnalyzer(Protocol):
    fingerprint: str

    async def analyze(self, photo: PhotoCandidate) -> Analysis: ...


def _strip_markdown_fence(value: str) -> str:
    value = value.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
        if value.lower().startswith("json"):
            value = value[4:].strip()
    return value


def _normalise_keyword(value: str) -> str:
    value = re.sub(r"\s+", "-", value.strip().lower())
    return re.sub(r"[^\w-]", "", value, flags=re.UNICODE)


def parse_analysis_response(
    content: str,
    *,
    max_keywords: int,
    max_title_chars: int,
    max_description_chars: int,
    usage: Usage | None = None,
) -> Analysis:
    try:
        value = json.loads(_strip_markdown_fence(content))
    except json.JSONDecodeError as exc:
        raise AnalysisValidationError("model response was not valid JSON") from exc
    if not isinstance(value, dict):
        raise AnalysisValidationError("model response must be a JSON object")

    title = value.get("title")
    description = value.get("description")
    keywords = value.get("keywords")
    if not isinstance(title, str) or not title.strip():
        raise AnalysisValidationError("title must be a non-empty string")
    if not isinstance(description, str) or not description.strip():
        raise AnalysisValidationError("description must be a non-empty string")
    if not isinstance(keywords, list) or not all(isinstance(item, str) for item in keywords):
        raise AnalysisValidationError("keywords must be a list of strings")

    normalised_keywords: list[str] = []
    seen: set[str] = set()
    for item in keywords:
        keyword = _normalise_keyword(item)
        if keyword and keyword not in seen:
            normalised_keywords.append(keyword)
            seen.add(keyword)
        if len(normalised_keywords) >= max_keywords:
            break
    if not normalised_keywords:
        raise AnalysisValidationError("keywords must contain at least one usable value")

    return Analysis(
        title=title.strip()[:max_title_chars],
        description=description.strip()[:max_description_chars],
        keywords=tuple(normalised_keywords),
        usage=usage or Usage(),
    )


def render_prompt(settings: Settings) -> str:
    replacements = {
        "{{MAX_KEYWORDS}}": str(settings.max_keywords),
        "{{MAX_TITLE_CHARS}}": str(settings.max_title_chars),
        "{{MAX_DESCRIPTION_CHARS}}": str(settings.max_description_chars),
        "{{LANGUAGE}}": settings.language,
        "{{DESCRIPTION_STYLE}}": settings.description_style,
    }
    if settings.prompt_file:
        prompt = Path(settings.prompt_file).read_text(encoding="utf-8")
        for token, replacement in replacements.items():
            prompt = prompt.replace(token, replacement)
        return prompt
    return (
        "Generate Flickr metadata as one valid JSON object with exactly these fields:\n"
        f'- "title": a self-contained title in {settings.language}, no more than '
        f"{settings.max_title_chars} characters.\n"
        f'- "description": a {settings.description_style} description in '
        f"{settings.language}, no more than {settings.max_description_chars} characters.\n"
        f'- "keywords": up to {settings.max_keywords} lowercase keywords without spaces '
        "or symbols.\n"
        "Use only visible image evidence and the optional album/location context. Do not "
        "mention that context or make unsupported location claims."
    )


class OpenAIImageAnalyzer:
    def __init__(self, settings: Settings, client: Any | None = None):
        self.settings = settings
        self.prompt = render_prompt(settings)
        self.client = client or self._create_client()
        fingerprint_value = json.dumps(
            {
                "provider": settings.openai_provider,
                "base_url": settings.openai_base_url,
                "model": settings.openai_model,
                "role": settings.openai_instruction_role,
                "json_mode": settings.openai_json_mode,
                "image_detail": settings.openai_image_detail,
                "prompt": self.prompt,
            },
            sort_keys=True,
        )
        self.fingerprint = hashlib.sha256(fingerprint_value.encode()).hexdigest()[:16]

    def _create_client(self) -> AsyncOpenAI | AsyncAzureOpenAI:
        common: dict[str, Any] = {
            "api_key": self.settings.openai_api_key,
            "timeout": self.settings.openai_timeout,
            "max_retries": self.settings.openai_max_retries,
        }
        if self.settings.openai_provider == "azure":
            return AsyncAzureOpenAI(
                **common,
                azure_endpoint=self.settings.azure_openai_endpoint,
                api_version=self.settings.azure_openai_api_version,
            )
        return AsyncOpenAI(**common, base_url=self.settings.openai_base_url)

    async def analyze(self, photo: PhotoCandidate) -> Analysis:
        context: dict[str, Any] = {}
        if photo.photoset_title:
            context["albumTitle"] = photo.photoset_title
        if photo.photoset_description:
            context["albumDescription"] = photo.photoset_description
        if photo.location:
            context["location"] = photo.location

        request: dict[str, Any] = {
            "model": self.settings.openai_model,
            "messages": [
                {"role": self.settings.openai_instruction_role, "content": self.prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json.dumps(context)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": photo.image_url,
                                "detail": self.settings.openai_image_detail,
                            },
                        },
                    ],
                },
            ],
            self.settings.openai_tokens_parameter: self.settings.openai_max_output_tokens,
        }
        if self.settings.openai_json_mode:
            request["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**request)
        response_usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(response_usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(response_usage, "completion_tokens", 0) or 0)
        cost = (
            prompt_tokens * self.settings.openai_cost_per_1m_prompt_tokens / 1_000_000
            + completion_tokens * self.settings.openai_cost_per_1m_completion_tokens / 1_000_000
            + self.settings.openai_vision_cost_per_image
        )
        usage = Usage(prompt_tokens, completion_tokens, cost)
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise AnalysisValidationError("model returned no text content")
        return parse_analysis_response(
            content,
            max_keywords=self.settings.max_keywords,
            max_title_chars=self.settings.max_title_chars,
            max_description_chars=self.settings.max_description_chars,
            usage=usage,
        )


async def analyze_concurrently(
    analyzer: ImageAnalyzer,
    photos: Sequence[PhotoCandidate],
    concurrency: int,
) -> list[Analysis | Exception]:
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(photo: PhotoCandidate) -> Analysis | Exception:
        async with semaphore:
            try:
                return await analyzer.analyze(photo)
            except Exception as exc:  # returned for per-photo audit and continuation
                return exc

    return list(await asyncio.gather(*(guarded(photo) for photo in photos)))
