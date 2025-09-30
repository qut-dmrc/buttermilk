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
from buttermilk.utils.utils import scrub_serializable

CACHE_VERSION = 1


def _default_base_dir() -> Path:
    return Path(os.getenv("BM_RECORD_CACHE_DIR", ".cache")) / "records"


class RecordCache:
    """Filesystem-backed cache for Record objects per processing stage."""

    def __init__(self, base_dir: str | Path | None = None, enabled: bool | None = None):
        if base_dir:
            # Expand paths if provided as string
            if isinstance(base_dir, str):
                base_dir = os.path.expandvars(os.path.expanduser(base_dir))
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = _default_base_dir()
        if enabled is None:
            disabled_env = os.getenv("BM_DISABLE_RECORD_CACHE", "0")
            self.enabled = disabled_env.strip() not in {"1", "true", "TRUE", "yes", "on"}
        else:
            self.enabled = enabled

        # Debug logging for cache initialization
        logger.info(
            "🗂️  RecordCache initialized",
            base_dir=str(self.base_dir),
            enabled=self.enabled,
            working_dir=os.getcwd(),
            cache_env_var=os.getenv("BM_RECORD_CACHE_DIR"),
            disable_env_var=os.getenv("BM_DISABLE_RECORD_CACHE")
        )

        if self.enabled:
            try:
                self.base_dir.mkdir(parents=True, exist_ok=True)
                logger.debug("📁 Created cache base directory", base_dir=str(self.base_dir))
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
            logger.debug("🚫 Cache disabled", record_id=record_id, stage=stage)
            return False
        path = self._record_path(stage, record_id)
        exists = path.exists()
        logger.debug(
            "🔍 Cache lookup",
            record_id=record_id,
            stage=stage,
            path=str(path),
            exists=exists
        )
        return exists

    def load(self, record_id: str, stage: str) -> Record | None:
        if not self.has(record_id, stage):
            logger.debug("❌ Cache miss", record_id=record_id, stage=stage)
            return None
        path = self._record_path(stage, record_id)
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:  # pragma: no cover
            logger.debug("💥 Failed reading cache file", path=str(path), error=str(e))
            return None
        if payload.get("_schema_version") != CACHE_VERSION:
            logger.debug("🚫 Cache version mismatch", record_id=record_id, stage=stage, path=str(path))
            return None
        data = payload.get("record")
        if not isinstance(data, dict):
            logger.debug("🚫 Invalid record data in cache", record_id=record_id, stage=stage, path=str(path))
            return None

        # Prepare data for Record creation
        data_copy = data.copy()
        data_copy.pop("chunks", None)  # Remove chunks from record data temporarily

        # Rehydrate chunks BEFORE creating the Record
        raw_chunks = payload.get("chunks")
        chunks_count = 0
        hydrated_chunks = None

        # Always restore the chunks field - either as rehydrated objects or original value
        if isinstance(raw_chunks, list) and raw_chunks:
            try:
                from buttermilk.data.vector import ChunkedDocument  # type: ignore

                hydrated = []
                for i, c in enumerate(raw_chunks):
                    if isinstance(c, dict):
                        try:
                            hydrated.append(ChunkedDocument(**c))
                            chunks_count += 1
                        except Exception as ce:
                            logger.warning(f"Failed to rehydrate chunk {i} for record {record_id}: {ce}")
                            logger.debug(f"Chunk data: {c}")

                if hydrated:
                    hydrated_chunks = hydrated
                else:
                    logger.warning(f"No chunks successfully rehydrated for record {record_id} (had {len(raw_chunks)} raw chunks)")
                    # If rehydration failed, set to empty list to avoid dict objects
                    hydrated_chunks = []
            except Exception as e:
                logger.error(f"Chunk rehydration failed completely for record {record_id}: {e}")
                # If chunk rehydration fails, set chunks to empty list to avoid dict objects
                hydrated_chunks = []
        else:
            # No chunks in cache or chunks was empty - restore original chunks value from record data
            original_chunks = data.get("chunks")
            if original_chunks is None:
                # Original record had no chunks field - don't set it
                hydrated_chunks = None
            elif isinstance(original_chunks, list):
                # Original record had empty list or list of chunks
                hydrated_chunks = original_chunks if not original_chunks else []
            else:
                # Fallback for other chunk values
                hydrated_chunks = []

        # Add hydrated chunks to data if they exist
        if hydrated_chunks is not None:
            data_copy["chunks"] = hydrated_chunks

        try:
            # Create Record with chunks already included
            record = Record(**data_copy)
        except Exception as e:  # pragma: no cover
            logger.debug("💥 Failed to rehydrate Record", record_id=record_id, error=str(e))
            return None

        logger.info(
            "✅ Cache hit - loaded record",
            record_id=record_id,
            stage=stage,
            path=str(path),
            chunks_count=chunks_count
        )
        return record

    def save(self, record: Record, stage: str, include_chunks: bool = True) -> bool:
        if not self.enabled:
            logger.debug("🚫 Cache disabled - not saving", record_id=getattr(record, "record_id", "unknown"), stage=stage)
            return False
        if not record or not getattr(record, "record_id", None):
            logger.debug("🚫 Invalid record - not saving", record=record, stage=stage)
            return False

        stage_dir = self._stage_dir(stage)
        try:
            stage_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover
            logger.debug("💥 Could not create stage dir", stage_dir=str(stage_dir), error=str(e))
            return False

        path = self._record_path(stage, record.record_id)
        tmp_path = path.with_suffix(".tmp")
        chunks_count = len(getattr(record, "chunks", []))

        try:
            payload: dict[str, Any] = {
                "_schema_version": CACHE_VERSION,
                "stage": stage,
                "record_id": record.record_id,
                "record": scrub_serializable(record.model_dump()),  # Convert numpy arrays and clean data
            }
            if include_chunks and getattr(record, "chunks", None):
                serializable: list[dict[str, Any]] = []
                for ch in record.chunks:  # type: ignore[attr-defined]
                    if hasattr(ch, "model_dump"):
                        # Convert numpy arrays in chunks (especially embeddings) and clean data
                        chunk_data = scrub_serializable(ch.model_dump())
                        serializable.append(chunk_data)
                    elif hasattr(ch, "__dict__"):
                        chunk_data = {k: v for k, v in ch.__dict__.items() if not k.startswith("_")}
                        chunk_data = scrub_serializable(chunk_data)
                        serializable.append(chunk_data)
                    else:
                        try:
                            chunk_data = scrub_serializable(asdict(ch))
                            serializable.append(chunk_data)
                        except Exception:  # pragma: no cover
                            continue
                payload["chunks"] = serializable
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            tmp_path.replace(path)

            logger.info(
                "💾 Cached record",
                record_id=record.record_id,
                stage=stage,
                path=str(path),
                chunks_count=chunks_count,
                include_chunks=include_chunks
            )
            return True
        except Exception as e:  # pragma: no cover
            logger.debug("💥 Failed to cache record", record_id=record.record_id, stage=stage, error=str(e))
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
