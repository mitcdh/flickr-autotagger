from __future__ import annotations

import json

from autotagger.checkpoint import CheckpointStore, load_plan, write_summary


def test_checkpoint_preserves_state_when_a_skip_event_follows(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    store = CheckpointStore(path)
    store.append({"photo_id": "1", "status": "analysed", "analysis": {"title": "x"}})
    store.append({"photo_id": "1", "status": "skipped", "reason": "cached"})
    assert CheckpointStore(path).load_latest()["1"]["status"] == "analysed"


def test_summary_is_valid_json_and_plan_loader_supports_both_formats(tmp_path):
    records = [{"photo_id": "1", "status": "analysed"}]
    summary = tmp_path / "summary.json"
    write_summary(summary, records)
    assert json.loads(summary.read_text()) == records
    assert load_plan(summary) == records

    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text('{"photo_id":"1"}\n{"photo_id":"2"}\n')
    assert [record["photo_id"] for record in load_plan(checkpoint)] == ["1", "2"]
