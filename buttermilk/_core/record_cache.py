"""Stage-aware Record caching utilities.

Provides a lightweight filesystem cache for `Record` objects after each
processing stage (e.g., preprocessing, enrichment, chunking, embedding).

Cache layout:
    <base> / records / <stage_name> / <record_id>.json

Environment variables:
    BM_RECORD_CACHE_DIR       Base directory (default: .cache)
    BM_DISABLE_RECORD_CACHE   If set to a truthy (non-zero) value, disables cache.

Design goals:
    * Safe (never raises on IO errors; logs at debug level)
    * Atomic writes (temp file then rename)
    * Lazy chunk rehydration (avoids import cycles)
    * Stage isolation (independent invalidation)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from buttermilk._core.log import logger
from buttermilk._core.types import Record

CACHE_VERSION = 1


def _default_base_dir() -> Path:
    return Path(os.getenv("BM_RECORD_CACHE_DIR", ".cache")) / "records"


class RecordCache:
    """Filesystem-backed cache for Record objects per processing stage."""

    def __init__(self, base_dir: str | Path | None = None, enabled: bool | None = None):
        self.base_dir = Path(base_dir) if base_dir else _default_base_dir()
        if enabled is None:
            disabled_env = os.getenv("BM_DISABLE_RECORD_CACHE", "0")
            self.enabled = disabled_env.strip() not in {"1", "true", "TRUE", "yes", "on"}
        else:
            self.enabled = enabled
        if self.enabled:
            try:
                self.base_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:  # pragma: no cover
                logger.debug(f"Could not create record cache base dir {self.base_dir}: {e}")
                self.enabled = False

    # -------------------- Path helpers --------------------
    def _stage_dir(self, stage: str) -> Path:
        return self.base_dir / stage

    def _record_path(self, stage: str, record_id: str) -> Path:
        return self._stage_dir(stage) / f"{record_id}.json"

    # -------------------- Public API ----------------------
    def has(self, record_id: str, stage: str) -> bool:
        if not self.enabled:
            return False
        return self._record_path(stage, record_id).exists()

    def load(self, record_id: str, stage: str) -> Record | None:
        if not self.has(record_id, stage):
            return None
        path = self._record_path(stage, record_id)
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:  # pragma: no cover
            logger.debug(f"Failed reading cache {path}: {e}")
            return None
        if payload.get("_schema_version") != CACHE_VERSION:
            return None
        data = payload.get("record")
        if not isinstance(data, dict):
            return None
        try:
            record = Record(**data)
        except Exception as e:  # pragma: no cover
            logger.debug(f"Failed to rehydrate Record {record_id}: {e}")
            return None
        # Rehydrate chunks (optional)
        raw_chunks = payload.get("chunks")
        if isinstance(raw_chunks, list) and raw_chunks:
            try:
                from buttermilk.data.vector import ChunkedDocument  # type: ignore

                hydrated = []
                for c in raw_chunks:
                    if isinstance(c, dict):
                        try:
                            hydrated.append(ChunkedDocument(**c))
                        except Exception as ce:  # pragma: no cover
                            logger.debug(f"Skipping bad chunk for {record_id}: {ce}")
                if hydrated:
                    record.chunks = hydrated
            except Exception as e:  # pragma: no cover
                logger.debug(f"Chunk rehydration skipped: {e}")
        return record

    def save(self, record: Record, stage: str, include_chunks: bool = True) -> bool:
        if not self.enabled:
            return False
        if not record or not getattr(record, "record_id", None):
            return False
        stage_dir = self._stage_dir(stage)
        try:
            stage_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover
            logger.debug(f"Could not create stage dir {stage_dir}: {e}")
            return False
        path = self._record_path(stage, record.record_id)
        tmp_path = path.with_suffix(".tmp")
        try:
            payload: dict[str, Any] = {
                "_schema_version": CACHE_VERSION,
                "stage": stage,
                "record_id": record.record_id,
                "record": record.model_dump(),
            }
            if include_chunks and getattr(record, "chunks", None):
                serializable: list[dict[str, Any]] = []
                for ch in record.chunks:  # type: ignore[attr-defined]
                    if hasattr(ch, "model_dump"):
                        serializable.append(ch.model_dump())
                    elif hasattr(ch, "__dict__"):
                        serializable.append({k: v for k, v in ch.__dict__.items() if not k.startswith("_")})
                    else:
                        try:
                            serializable.append(asdict(ch))
                        except Exception:  # pragma: no cover
                            continue
                payload["chunks"] = serializable
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            tmp_path.replace(path)
            logger.debug(f"🗂️  Cached record {record.record_id} at stage '{stage}' -> {path}")
            return True
        except Exception as e:  # pragma: no cover
            logger.debug(f"Failed to cache record {record.record_id} at stage {stage}: {e}")
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            return False

    def purge_stage(self, stage: str) -> int:
        stage_dir = self._stage_dir(stage)
        if not stage_dir.exists():
            return 0
        removed = 0
        for p in stage_dir.glob("*.json"):
            try:
                p.unlink()
                removed += 1
            except Exception:  # pragma: no cover
                continue
        return removed

    def purge_all(self) -> int:
        if not self.base_dir.exists():
            return 0
        removed = 0
        for p in self.base_dir.rglob("*.json"):
            try:
                p.unlink()
                removed += 1
            except Exception:  # pragma: no cover
                continue
        return removed


__all__ = ["RecordCache"]
