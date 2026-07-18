from __future__ import annotations

import asyncio
from dataclasses import replace

from autotagger.flickr_gateway import PartialUpdateError
from autotagger.models import Analysis, PhotoCandidate, Photoset, Usage
from autotagger.pipeline import AutotaggerPipeline


class FakeFlickr:
    def __init__(self, photos):
        self.photos = photos
        self.updates = []

    def get_photosets(self):
        return [Photoset("set-1", "Album", "Album context")]

    def iter_photo_pages(self, photoset):
        yield self.photos

    def update_metadata(self, photo, analysis):
        self.updates.append((photo.id, analysis.title))
        return ("title", "description", "tags")


class FakeAnalyzer:
    fingerprint = "fake-analysis-v1"

    def __init__(self):
        self.calls = []

    async def analyze(self, photo):
        self.calls.append(photo.id)
        return Analysis(
            title=f"Title {photo.id}",
            description="Description",
            keywords=("keyword",),
            usage=Usage(100, 20, 0.01),
        )


class PartialFailureFlickr(FakeFlickr):
    def update_metadata(self, photo, analysis):
        raise PartialUpdateError("tag update failed", ("title", "description"))


def photo(photo_id="1", description=""):
    return PhotoCandidate(
        id=photo_id,
        photoset_id="set-1",
        photoset_title="Album",
        photoset_description="Context",
        image_url=f"https://example.test/{photo_id}.jpg",
        description=description,
        tags=("existing",),
    )


def test_dry_run_analyzes_without_updating(settings):
    flickr = FakeFlickr([photo()])
    analyzer = FakeAnalyzer()
    pipeline = AutotaggerPipeline(settings, flickr, analyzer)
    summary = asyncio.run(pipeline.run(apply=False))
    assert summary.analyzed == 1
    assert summary.updated == 0
    assert flickr.updates == []
    assert any(record["status"] == "analysed" for record in summary.records)
    assert settings.updated_metadata_file.exists()


def test_apply_updates_after_analysis(settings):
    flickr = FakeFlickr([photo()])
    pipeline = AutotaggerPipeline(settings, flickr, FakeAnalyzer())
    summary = asyncio.run(pipeline.run(apply=True))
    assert summary.updated == 1
    assert flickr.updates == [("1", "Title 1")]
    assert [record["status"] for record in summary.records] == ["analysed", "updated"]


def test_dry_run_checkpoint_can_be_resumed_and_applied(settings):
    resumable = replace(settings, resume=True)
    first_flickr = FakeFlickr([photo()])
    first_analyzer = FakeAnalyzer()
    asyncio.run(AutotaggerPipeline(resumable, first_flickr, first_analyzer).run(apply=False))

    second_flickr = FakeFlickr([photo()])
    second_analyzer = FakeAnalyzer()
    summary = asyncio.run(
        AutotaggerPipeline(resumable, second_flickr, second_analyzer).run(apply=True)
    )
    assert second_analyzer.calls == []
    assert summary.updated == 1
    assert second_flickr.updates == [("1", "Title 1")]


def test_repeated_dry_run_keeps_cached_analysis_in_current_report(settings):
    resumable = replace(settings, resume=True)
    asyncio.run(
        AutotaggerPipeline(resumable, FakeFlickr([photo()]), FakeAnalyzer()).run(apply=False)
    )
    analyzer = FakeAnalyzer()
    summary = asyncio.run(
        AutotaggerPipeline(resumable, FakeFlickr([photo()]), analyzer).run(apply=False)
    )
    assert analyzer.calls == []
    assert summary.records[0]["status"] == "analysed"
    assert summary.records[0]["cached_analysis"] is True


def test_existing_description_is_skipped(settings):
    flickr = FakeFlickr([photo(description="A human description")])
    analyzer = FakeAnalyzer()
    summary = asyncio.run(AutotaggerPipeline(settings, flickr, analyzer).run(apply=True))
    assert analyzer.calls == []
    assert summary.skipped == 1


def test_duplicate_photo_ids_are_not_analyzed_twice(settings):
    flickr = FakeFlickr([photo(), photo()])
    analyzer = FakeAnalyzer()
    summary = asyncio.run(AutotaggerPipeline(settings, flickr, analyzer).run(apply=False))
    assert analyzer.calls == ["1"]
    assert summary.skipped == 1


def test_limit_bounds_new_analyses(settings):
    limited = replace(settings, max_photos=1)
    flickr = FakeFlickr([photo("1"), photo("2")])
    analyzer = FakeAnalyzer()
    summary = asyncio.run(AutotaggerPipeline(limited, flickr, analyzer).run(apply=False))
    assert summary.analyzed == 1
    assert analyzer.calls == ["1"]


def test_partial_flickr_update_is_audited(settings):
    summary = asyncio.run(
        AutotaggerPipeline(settings, PartialFailureFlickr([photo()]), FakeAnalyzer()).run(
            apply=True
        )
    )
    assert summary.failed == 1
    partial = next(record for record in summary.records if record["status"] == "partially_updated")
    assert partial["applied_fields"] == ["title", "description"]


def test_dry_run_report_can_be_applied_without_openai(settings):
    source_flickr = FakeFlickr([photo()])
    asyncio.run(AutotaggerPipeline(settings, source_flickr, FakeAnalyzer()).run(apply=False))

    target_flickr = FakeFlickr([])
    summary = AutotaggerPipeline(settings, target_flickr, analyzer=None).apply_plan(
        settings.updated_metadata_file
    )
    assert summary.updated == 1
    assert target_flickr.updates == [("1", "Title 1")]
