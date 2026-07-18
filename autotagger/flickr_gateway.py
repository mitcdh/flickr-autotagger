"""Flickr authentication, pagination, normalisation, and safe metadata writes."""

from __future__ import annotations

import json
import logging
import os
import shlex
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TypeVar

import flickrapi
import requests

from .config import Settings
from .models import Analysis, PhotoCandidate, Photoset

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


class PartialUpdateError(RuntimeError):
    def __init__(self, message: str, applied_fields: tuple[str, ...]):
        super().__init__(message)
        self.applied_fields = applied_fields


def _content(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("_content", "")
    return value.strip() if isinstance(value, str) else ""


def _parse_tags(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            return tuple(tag for tag in shlex.split(value) if tag)
        except ValueError:
            return tuple(tag for tag in value.split() if tag)
    if isinstance(value, dict):
        value = value.get("tag", ())
    if isinstance(value, list):
        tags = []
        for item in value:
            tag = item.get("raw") or item.get("_content") if isinstance(item, dict) else item
            if isinstance(tag, str) and tag.strip():
                tags.append(tag.strip())
        return tuple(tags)
    return ()


def _format_tags(tags: tuple[str, ...]) -> str:
    formatted = []
    for tag in tags:
        cleaned = tag.strip().replace('"', "")
        if not cleaned:
            continue
        formatted.append(f'"{cleaned}"' if any(char.isspace() for char in cleaned) else cleaned)
    return " ".join(formatted)


def _merge_tags(existing: tuple[str, ...], generated: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for tag in (*existing, *generated):
        key = tag.casefold()
        if key not in seen:
            merged.append(tag)
            seen.add(key)
    return tuple(merged)


def _write_private_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


class FlickrGateway:
    def __init__(self, settings: Settings, api: Any | None = None):
        self.settings = settings
        self.api = api or self._authenticate()

    def _token_from_dict(self, value: dict[str, Any]) -> Any:
        return flickrapi.auth.FlickrAccessToken(
            value["oauth_token"],
            value["oauth_token_secret"],
            value["access_level"],
            value["fullname"],
            value["username"],
            value["user_nsid"],
        )

    def _authenticated_api(self, token_dict: dict[str, Any]) -> Any:
        return flickrapi.FlickrAPI(
            self.settings.flickr_api_key,
            self.settings.flickr_api_secret,
            token=self._token_from_dict(token_dict),
            format="parsed-json",
        )

    def _authenticate(self) -> Any:
        token_file = self.settings.flickr_token_file
        if token_file.exists():
            try:
                with token_file.open("r", encoding="utf-8") as handle:
                    return self._authenticated_api(json.load(handle))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                LOGGER.warning("Ignoring invalid Flickr token file %s: %s", token_file, exc)

        if self.settings.flickr_oauth_token:
            try:
                return self._authenticated_api(json.loads(self.settings.flickr_oauth_token))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError("FLICKR_OAUTH_TOKEN is not a valid Flickr token object") from exc

        last_error: Exception | None = None
        for attempt in range(2):
            api = flickrapi.FlickrAPI(
                self.settings.flickr_api_key,
                self.settings.flickr_api_secret,
                format="parsed-json",
            )
            try:
                api.token_cache.forget()
                LOGGER.info("Performing interactive Flickr OAuth authentication")
                api.get_request_token(oauth_callback="oob")
                authorize_url = api.auth_url(perms="write")
                print(f"Please visit this URL to authorize the application: {authorize_url}")
                api.get_access_token(input("Enter the verifier code: ").strip())
                token = api.token_cache.token
                token_dict = {
                    "oauth_token": token.token,
                    "oauth_token_secret": token.token_secret,
                    "access_level": token.access_level,
                    "fullname": token.fullname,
                    "username": token.username,
                    "user_nsid": token.user_nsid,
                }
                _write_private_json(token_file, token_dict)
                return api
            except flickrapi.FlickrError as exc:
                last_error = exc
                LOGGER.warning(
                    "Flickr authentication failed (attempt %d of 2): %s",
                    attempt + 1,
                    exc,
                )
        raise RuntimeError("Flickr authentication failed after two attempts") from last_error

    def _call(self, operation: Callable[..., T], **kwargs: Any) -> T:
        attempts = self.settings.flickr_max_retries + 1
        for attempt in range(attempts):
            try:
                return operation(**kwargs)
            except flickrapi.FlickrError as exc:
                if exc.code not in {105, 106}:
                    raise
                if attempt == attempts - 1:
                    raise
                delay = self.settings.flickr_retry_backoff * (2**attempt)
                LOGGER.warning("Transient Flickr error; retrying in %.1f seconds", delay)
                time.sleep(delay)
            except requests.RequestException:
                if attempt == attempts - 1:
                    raise
                delay = self.settings.flickr_retry_backoff * (2**attempt)
                LOGGER.warning("Transient Flickr error; retrying in %.1f seconds", delay)
                time.sleep(delay)
        raise AssertionError("unreachable")

    def get_photosets(self) -> list[Photoset]:
        if self.settings.flickr_photoset_ids:
            result = []
            for photoset_id in self.settings.flickr_photoset_ids:
                response = self._call(self.api.photosets.getInfo, photoset_id=photoset_id)
                value = response["photoset"]
                result.append(
                    Photoset(
                        id=str(value["id"]),
                        title=_content(value.get("title")),
                        description=_content(value.get("description")),
                    )
                )
            return result

        photosets: list[Photoset] = []
        page = 1
        while True:
            response = self._call(
                self.api.photosets.getList, page=page, per_page=500, extras="description"
            )
            container = response["photosets"]
            for value in container.get("photoset", []):
                photosets.append(
                    Photoset(
                        id=str(value["id"]),
                        title=_content(value.get("title")),
                        description=_content(value.get("description")),
                    )
                )
            if page >= int(container.get("pages", 1)):
                break
            page += 1
        return photosets

    def iter_photo_pages(self, photoset: Photoset) -> Iterator[list[PhotoCandidate]]:
        page = 1
        extras = f"{self.settings.flickr_image_url},description,geo,tags"
        while True:
            kwargs: dict[str, Any] = {
                "photoset_id": photoset.id,
                "extras": extras,
                "media": "photos",
                "page": page,
                "per_page": 500,
            }
            if self.settings.flickr_privacy_filter is not None:
                kwargs["privacy_filter"] = self.settings.flickr_privacy_filter
            response = self._call(self.api.photosets.getPhotos, **kwargs)
            container = response["photoset"]
            candidates = [
                self._normalise_photo(value, photoset) for value in container.get("photo", [])
            ]
            yield candidates
            if page >= int(container.get("pages", 1)):
                break
            page += 1

    def _normalise_photo(self, value: dict[str, Any], photoset: Photoset) -> PhotoCandidate:
        description = _content(value.get("description"))
        title = _content(value.get("title"))
        tags = _parse_tags(value.get("tags"))
        latitude = value.get("latitude")
        longitude = value.get("longitude")

        if "description" not in value:
            info = self._call(self.api.photos.getInfo, photo_id=value["id"])["photo"]
            description = _content(info.get("description"))
            title = title or _content(info.get("title"))
            tags = tags or _parse_tags(info.get("tags"))
            location = info.get("location", {})
            latitude = latitude if latitude not in {None, ""} else location.get("latitude")
            longitude = longitude if longitude not in {None, ""} else location.get("longitude")

        image_url = value.get(self.settings.flickr_image_url)
        if not isinstance(image_url, str) or not image_url:
            image_field = self.settings.flickr_image_url
            raise ValueError(f"Flickr did not return {image_field} for photo {value.get('id')}")
        return PhotoCandidate(
            id=str(value["id"]),
            photoset_id=photoset.id,
            photoset_title=photoset.title,
            photoset_description=photoset.description,
            image_url=image_url,
            title=title,
            description=description,
            tags=tags,
            latitude=latitude,
            longitude=longitude,
        )

    def update_metadata(self, photo: PhotoCandidate, analysis: Analysis) -> tuple[str, ...]:
        applied: list[str] = []
        fields = set(self.settings.update_fields)
        try:
            meta: dict[str, Any] = {"photo_id": photo.id}
            if "title" in fields:
                meta["title"] = analysis.title
            if "description" in fields:
                meta["description"] = analysis.description
            if len(meta) > 1:
                self._call(self.api.photos.setMeta, **meta)
                applied.extend(field for field in ("title", "description") if field in fields)

            if "tags" in fields:
                tags = analysis.keywords
                if self.settings.tag_mode == "merge":
                    tags = _merge_tags(photo.tags, analysis.keywords)
                self._call(self.api.photos.setTags, photo_id=photo.id, tags=_format_tags(tags))
                applied.append("tags")
            return tuple(applied)
        except Exception as exc:
            if applied:
                raise PartialUpdateError(str(exc), tuple(applied)) from exc
            raise
