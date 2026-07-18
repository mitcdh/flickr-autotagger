"""Durable JSON Lines checkpoints and atomic run summaries."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class CheckpointStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._latest: dict[str, dict[str, Any]] | None = None

    def load_latest(self) -> dict[str, dict[str, Any]]:
        if self._latest is not None:
            return self._latest
        latest: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in {self.path} at line {line_number}"
                        ) from exc
                    photo_id = record.get("photo_id")
                    if isinstance(photo_id, str) and (
                        record.get("status") != "skipped" or photo_id not in latest
                    ):
                        latest[photo_id] = record
        self._latest = latest
        return latest

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if self._latest is None:
                self._latest = {}
            photo_id = record.get("photo_id")
            if isinstance(photo_id, str) and (
                record.get("status") != "skipped" or photo_id not in self._latest
            ):
                self._latest[photo_id] = record


def write_summary(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(list(records), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_plan(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"Plan must be a JSON array: {path}")
        return [item for item in value if isinstance(item, dict)]
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
        if isinstance(value, dict):
            records.append(value)
    return records
