"""Typed data exchanged between the Flickr, OpenAI, and pipeline layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Photoset:
    id: str
    title: str
    description: str = ""


@dataclass(frozen=True)
class PhotoCandidate:
    id: str
    photoset_id: str
    photoset_title: str
    photoset_description: str
    image_url: str
    title: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()
    latitude: str | float | None = None
    longitude: str | float | None = None

    @property
    def location(self) -> dict[str, str | float] | None:
        if self.latitude is None or self.longitude is None:
            return None
        if self.latitude == "" or self.longitude == "":
            return None
        return {"latitude": self.latitude, "longitude": self.longitude}

    def to_record_context(self) -> dict[str, Any]:
        return {
            "photo_id": self.id,
            "photoset_id": self.photoset_id,
            "photoset_title": self.photoset_title,
            "image_url": self.image_url,
            "existing_title": self.title,
            "existing_description": self.description,
            "existing_tags": list(self.tags),
        }


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0


@dataclass(frozen=True)
class Analysis:
    title: str
    description: str
    keywords: tuple[str, ...]
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["keywords"] = list(self.keywords)
        return result
