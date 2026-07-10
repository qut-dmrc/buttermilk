"""Test AsyncDataUploader shutdown-drain behaviour (regression for issue #422).

When a pipeline finishes and enters graceful shutdown, the background uploader must
finish draining its queue before it stops, otherwise queued records are silently
dropped (issue #422: misleading ``success=100.0%`` while data is lost).

These tests exercise ``finalize_processing()`` with REAL ``FileStorage`` over a
tempfile (no mocks of internal code), matching the conventions in
``test_async_uploader_timestamp.py``.

Note: ``FileStorage`` defaults to OVERWRITE mode, so each test is configured so that
exactly one flush happens — at finalization — with ``buffer_size`` larger than the
record count and a long ``flush_interval``. This keeps the assertions about "no
records dropped" meaningful (a mid-run flush would overwrite earlier records).
"""

import time

import pytest
from structlog.testing import capture_logs

from buttermilk._core.storage_config import FileStorageConfig
from buttermilk._core.types import Record
from buttermilk.storage.file import FileStorage
from buttermilk.utils.uploader import AsyncDataUploader


def _make_records(n: int) -> list[Record]:
    return [
        Record(record_id=f"rec_{i:04d}", content=f"content {i}", dataset_name="test", split_type="test")
        for i in range(n)
    ]


class _SlowFileStorage(FileStorage):
    """FileStorage whose save() is artificially slow, to simulate a high-latency
    backend (e.g. BigQuery streaming inserts) so the drain exceeds the warn budget."""

    save_seconds: float = 0.2

    def save(self, records) -> None:  # type: ignore[override]
        time.sleep(self.save_seconds)
        return super().save(records)


@pytest.mark.anyio
async def test_finalize_drains_all_records(real_bm, tmp_path):
    """finalize_processing() must not drop queued records (the #422 regression).

    With buffer_size larger than the record count, nothing flushes mid-run; the
    records sit in the queue/buffer until finalization drains them.
    """
    path = tmp_path / "traces.json"
    config = FileStorageConfig(type="file", path=str(path), dataset_name="test", split_type="test")
    storage = FileStorage(config)

    records = _make_records(50)
    uploader = AsyncDataUploader(storage=storage, buffer_size=1000, flush_interval=3600)
    for r in records:
        await uploader.add(r)

    ok = await uploader.finalize_processing()
    assert ok is True

    saved = list(storage)
    assert len(saved) == 50, f"expected all 50 records, got {len(saved)}"
    assert {r.record_id for r in saved} == {r.record_id for r in records}


@pytest.mark.anyio
async def test_finalize_warns_when_flush_exceeds_budget(real_bm, tmp_path):
    """If draining takes longer than the warn budget, a WARNING pointing at
    buffer_size is emitted — but records are still fully written (loss-free)."""
    path = tmp_path / "traces.json"  # non-existent -> no timestamp suffix -> slow storage is used for flush
    config = FileStorageConfig(type="file", path=str(path), dataset_name="test", split_type="test")
    storage = _SlowFileStorage(config)

    records = _make_records(5)
    # Tiny warn budget so the (slow) single flush at finalization overshoots it.
    uploader = AsyncDataUploader(storage=storage, buffer_size=1000, flush_interval=3600, shutdown_warn_seconds=0.01)
    for r in records:
        await uploader.add(r)

    with capture_logs() as logs:
        ok = await uploader.finalize_processing()

    assert ok is True

    warnings = [e for e in logs if e.get("log_level") == "warning" and "buffer_size" in e.get("event", "")]
    assert warnings, f"expected a buffer_size warning; captured: {[e.get('event') for e in logs]}"

    saved = list(storage)
    assert len(saved) == 5, f"records must not be dropped even when slow; got {len(saved)}"
    assert {r.record_id for r in saved} == {r.record_id for r in records}


@pytest.mark.anyio
async def test_finalize_no_warning_on_fast_flush(real_bm, tmp_path):
    """The warning must NOT fire on a normal fast shutdown (guards against a
    spurious warning on every run)."""
    path = tmp_path / "traces.json"
    config = FileStorageConfig(type="file", path=str(path), dataset_name="test", split_type="test")
    storage = FileStorage(config)

    records = _make_records(5)
    uploader = AsyncDataUploader(storage=storage, buffer_size=1000, flush_interval=3600)  # default 10s budget
    for r in records:
        await uploader.add(r)

    with capture_logs() as logs:
        ok = await uploader.finalize_processing()

    assert ok is True
    warnings = [e for e in logs if e.get("log_level") == "warning" and "buffer_size" in e.get("event", "")]
    assert not warnings, f"unexpected buffer_size warning on fast flush: {[e.get('event') for e in warnings]}"

    saved = list(storage)
    assert len(saved) == 5
