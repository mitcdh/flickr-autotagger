from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import flickrapi
import pytest

from autotagger.flickr_gateway import FlickrGateway
from autotagger.models import Analysis, Photoset


class Recorder:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class ErrorThenSuccess:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise self.error
        return {"ok": True}


def test_metadata_update_merges_tags_and_uses_space_delimiter(settings):
    set_meta = Recorder({})
    set_tags = Recorder({})
    api = SimpleNamespace(photos=SimpleNamespace(setMeta=set_meta, setTags=set_tags))
    gateway = FlickrGateway(settings, api=api)
    photo = SimpleNamespace(id="1", tags=("existing", "two words"))
    analysis = Analysis("Title", "Description", ("existing", "new"))

    applied = gateway.update_metadata(photo, analysis)

    assert applied == ("title", "description", "tags")
    assert set_tags.calls[0]["tags"] == 'existing "two words" new'
    assert set_meta.calls[0]["title"] == "Title"


def test_replace_mode_does_not_preserve_existing_tags(settings):
    set_tags = Recorder({})
    api = SimpleNamespace(photos=SimpleNamespace(setMeta=Recorder({}), setTags=set_tags))
    gateway = FlickrGateway(replace(settings, tag_mode="replace"), api=api)
    photo = SimpleNamespace(id="1", tags=("existing",))
    gateway.update_metadata(photo, Analysis("Title", "Description", ("new",)))
    assert set_tags.calls[0]["tags"] == "new"


def test_photo_pages_are_streamed_and_missing_descriptions_use_get_info(settings):
    get_photos = Recorder(
        {
            "photoset": {
                "page": 1,
                "pages": 1,
                "photo": [
                    {
                        "id": "1",
                        "title": "Raw title",
                        "url_m": "https://example.test/1.jpg",
                        "tags": "one two",
                    }
                ],
            }
        }
    )
    get_info = Recorder(
        {
            "photo": {
                "description": {"_content": "Existing description"},
                "tags": {"tag": [{"raw": "one"}, {"raw": "two"}]},
            }
        }
    )
    api = SimpleNamespace(
        photosets=SimpleNamespace(getPhotos=get_photos),
        photos=SimpleNamespace(getInfo=get_info),
    )
    gateway = FlickrGateway(settings, api=api)
    pages = list(gateway.iter_photo_pages(Photoset("set", "Album")))
    assert pages[0][0].description == "Existing description"
    assert pages[0][0].tags == ("one", "two")
    assert get_info.calls == [{"photo_id": "1"}]


def test_all_privacy_does_not_send_invalid_zero_filter(settings):
    get_photos = Recorder({"photoset": {"page": 1, "pages": 1, "photo": []}})
    api = SimpleNamespace(
        photosets=SimpleNamespace(getPhotos=get_photos),
        photos=SimpleNamespace(),
    )
    gateway = FlickrGateway(replace(settings, flickr_privacy_filter=None), api=api)
    list(gateway.iter_photo_pages(Photoset("set", "Album")))
    assert "privacy_filter" not in get_photos.calls[0]


def test_only_transient_flickr_errors_are_retried(settings):
    gateway = FlickrGateway(replace(settings, flickr_retry_backoff=0), api=SimpleNamespace())
    transient = ErrorThenSuccess(flickrapi.FlickrError("unavailable", code=105))
    assert gateway._call(transient) == {"ok": True}
    assert transient.calls == 2

    permanent = ErrorThenSuccess(flickrapi.FlickrError("bad request", code=1))
    with pytest.raises(flickrapi.FlickrError):
        gateway._call(permanent)
    assert permanent.calls == 1
