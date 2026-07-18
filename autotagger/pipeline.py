"""Resumable autotagging orchestration."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .analysis import ImageAnalyzer, analyze_concurrently
from .checkpoint import CheckpointStore, load_plan, write_summary
from .config import Settings
from .flickr_gateway import FlickrGateway, PartialUpdateError
from .models import Analysis, PhotoCandidate, Usage

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSummary:
    records: tuple[dict[str, Any], ...]
    analyzed: int
    updated: int
    failed: int
    skipped: int
    total_cost: float


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _analysis_from_dict(value: dict[str, Any]) -> Analysis:
    usage_value = value.get("usage", {})
    return Analysis(
        title=value["title"],
        description=value["description"],
        keywords=tuple(value["keywords"]),
        usage=Usage(
            prompt_tokens=int(usage_value.get("prompt_tokens", 0)),
            completion_tokens=int(usage_value.get("completion_tokens", 0)),
            cost=float(usage_value.get("cost", 0)),
        ),
    )


def _photo_from_record(record: dict[str, Any]) -> PhotoCandidate:
    return PhotoCandidate(
        id=str(record["photo_id"]),
        photoset_id=str(record.get("photoset_id", "")),
        photoset_title=str(record.get("photoset_title", "")),
        photoset_description="",
        image_url=str(record.get("image_url", "")),
        title=str(record.get("existing_title", "")),
        description=str(record.get("existing_description", "")),
        tags=tuple(record.get("existing_tags", ())),
    )


class AutotaggerPipeline:
    def __init__(
        self,
        settings: Settings,
        flickr: FlickrGateway,
        analyzer: ImageAnalyzer | None,
        checkpoint: CheckpointStore | None = None,
    ):
        self.settings = settings
        self.flickr = flickr
        self.analyzer = analyzer
        self.checkpoint = checkpoint or CheckpointStore(settings.checkpoint_file)
        self.run_id = str(uuid.uuid4())
        self.records: list[dict[str, Any]] = []
        self.total_cost = 0.0
        self.analyzed_count = 0
        self.updated_count = 0
        self.failed_count = 0
        self.skipped_count = 0

    def _record(
        self,
        status: str,
        photo: PhotoCandidate,
        **values: Any,
    ) -> dict[str, Any]:
        record = {
            "timestamp": _timestamp(),
            "run_id": self.run_id,
            "status": status,
            **photo.to_record_context(),
            **values,
        }
        self.records.append(record)
        self.checkpoint.append(record)
        if status == "updated":
            self.updated_count += 1
        elif status in {"failed", "partially_updated"}:
            self.failed_count += 1
        elif status == "skipped":
            self.skipped_count += 1
        return record

    def _photoset_selected(self, photoset_id: str, title: str) -> bool:
        if any(title.startswith(prefix) for prefix in self.settings.skip_prefixes):
            return False
        identity = {photoset_id, title}
        if self.settings.photoset_allowlist and identity.isdisjoint(
            self.settings.photoset_allowlist
        ):
            return False
        return identity.isdisjoint(self.settings.photoset_denylist)

    def _photo_selected(self, photo: PhotoCandidate) -> tuple[bool, str]:
        if self.settings.description_policy == "always":
            return True, ""
        if self.settings.description_policy == "never":
            return False, "description policy is never"
        description = photo.description.strip()
        if not description:
            return True, ""
        if any(description.startswith(value) for value in self.settings.descriptions_to_analyze):
            return True, ""
        return False, "existing description"

    def _can_analyze_more(self) -> bool:
        if self.settings.max_photos is not None and self.analyzed_count >= self.settings.max_photos:
            return False
        return not (
            self.settings.max_total_cost is not None
            and self.total_cost >= self.settings.max_total_cost
        )

    async def run(self, *, apply: bool) -> RunSummary:
        if self.analyzer is None:
            raise RuntimeError("An analyzer is required to run image analysis")
        latest = self.checkpoint.load_latest() if self.settings.resume else {}
        processed: set[str] = set()
        stop = False

        for photoset in self.flickr.get_photosets():
            if not self._photoset_selected(photoset.id, photoset.title):
                LOGGER.info("Skipping photoset %s (%s)", photoset.title, photoset.id)
                continue
            LOGGER.info("Processing photoset %s (%s)", photoset.title, photoset.id)
            for page in self.flickr.iter_photo_pages(photoset):
                pending: list[PhotoCandidate] = []
                cached: list[tuple[PhotoCandidate, dict[str, Any]]] = []
                for photo in page:
                    if photo.id in processed:
                        self._record("skipped", photo, reason="duplicate photo in this run")
                        continue
                    processed.add(photo.id)
                    selected, reason = self._photo_selected(photo)
                    if not selected:
                        self._record("skipped", photo, reason=reason)
                        continue

                    prior = latest.get(photo.id)
                    if prior and prior.get("status") == "updated":
                        self._record("skipped", photo, reason="already updated in checkpoint")
                        continue
                    if (
                        prior
                        and prior.get("status") == "analysed"
                        and prior.get("analyzer_fingerprint") == self.analyzer.fingerprint
                        and isinstance(prior.get("analysis"), dict)
                    ):
                        cached.append((photo, prior))
                        continue
                    if not self._can_analyze_more():
                        stop = True
                        break
                    pending.append(photo)
                    if self.settings.max_photos is not None:
                        remaining = self.settings.max_photos - self.analyzed_count
                        if len(pending) >= remaining:
                            stop = True
                            break

                for photo, prior in cached:
                    if apply:
                        self._apply(photo, _analysis_from_dict(prior["analysis"]), cached=True)
                    else:
                        self._record(
                            "analysed",
                            photo,
                            analysis=prior["analysis"],
                            analyzer_fingerprint=self.analyzer.fingerprint,
                            cached_analysis=True,
                        )

                for start in range(0, len(pending), self.settings.analysis_concurrency):
                    if not self._can_analyze_more():
                        stop = True
                        break
                    batch = pending[start : start + self.settings.analysis_concurrency]
                    outcomes = await analyze_concurrently(
                        self.analyzer, batch, self.settings.analysis_concurrency
                    )
                    for photo, outcome in zip(batch, outcomes, strict=True):
                        self.analyzed_count += 1
                        if isinstance(outcome, Exception):
                            self._record(
                                "failed",
                                photo,
                                stage="analysis",
                                error=f"{type(outcome).__name__}: {outcome}",
                                analyzer_fingerprint=self.analyzer.fingerprint,
                            )
                            continue
                        self.total_cost += outcome.usage.cost
                        self._record(
                            "analysed",
                            photo,
                            analysis=outcome.to_dict(),
                            analyzer_fingerprint=self.analyzer.fingerprint,
                        )
                        if apply:
                            self._apply(photo, outcome)
                    if not self._can_analyze_more():
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break

        write_summary(self.settings.updated_metadata_file, self.records)
        return self._summary()

    def _apply(self, photo: PhotoCandidate, analysis: Analysis, *, cached: bool = False) -> None:
        try:
            applied_fields = self.flickr.update_metadata(photo, analysis)
            self._record(
                "updated",
                photo,
                analysis=analysis.to_dict(),
                applied_fields=list(applied_fields),
                cached_analysis=cached,
            )
        except PartialUpdateError as exc:
            self._record(
                "partially_updated",
                photo,
                stage="flickr_update",
                analysis=analysis.to_dict(),
                applied_fields=list(exc.applied_fields),
                error=str(exc),
            )
        except Exception as exc:
            self._record(
                "failed",
                photo,
                stage="flickr_update",
                analysis=analysis.to_dict(),
                error=f"{type(exc).__name__}: {exc}",
            )

    def apply_plan(self, path: Path) -> RunSummary:
        latest_by_photo: dict[str, dict[str, Any]] = {}
        for record in load_plan(path):
            photo_id = record.get("photo_id")
            if not isinstance(photo_id, str):
                continue
            if (
                record.get("status") == "updated"
                or isinstance(record.get("analysis"), dict)
                and (latest_by_photo.get(photo_id, {}).get("status") != "updated")
            ):
                latest_by_photo[photo_id] = record

        for record in latest_by_photo.values():
            analysis_value = record.get("analysis")
            if record.get("status") == "updated" or not isinstance(analysis_value, dict):
                continue
            try:
                photo = _photo_from_record(record)
                analysis = _analysis_from_dict(analysis_value)
                self._apply(photo, analysis, cached=True)
            except Exception as exc:
                photo_id = str(record.get("photo_id", "unknown"))
                placeholder = PhotoCandidate(
                    id=photo_id,
                    photoset_id=str(record.get("photoset_id", "")),
                    photoset_title=str(record.get("photoset_title", "")),
                    photoset_description="",
                    image_url=str(record.get("image_url", "")),
                )
                self._record(
                    "failed",
                    placeholder,
                    stage="plan_validation",
                    error=f"{type(exc).__name__}: {exc}",
                )
        write_summary(self.settings.updated_metadata_file, self.records)
        return self._summary()

    def _summary(self) -> RunSummary:
        return RunSummary(
            records=tuple(self.records),
            analyzed=self.analyzed_count,
            updated=self.updated_count,
            failed=self.failed_count,
            skipped=self.skipped_count,
            total_cost=self.total_cost,
        )
