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
from pathlib import Path
from typing import Any

from buttermilk._core.log import logger
from buttermilk._core.types import BaseRecord
from buttermilk.utils.utils import scrub_serializable

CACHE_VERSION = 1


def _default_base_dir() -> Path:
    return Path(os.getenv("BM_RECORD_CACHE_DIR", ".cache")) / "records"


class RecordCache:
    """Filesystem-backed cache for Record objects per processing stage."""

    def __init__(self, base_dir: str | Path | None = None, enabled: bool | None = None):
        # Store the configured base_dir (may be None for lazy resolution)
        self._base_dir_config = base_dir
        self._base_dir: Path | None = None

        if enabled is None:
            disabled_env = os.getenv("BM_DISABLE_RECORD_CACHE", "0")
            self.enabled = disabled_env.strip() not in {
                "1",
                "true",
                "TRUE",
                "yes",
                "on",
            }
        else:
            self.enabled = enabled

    @property
    def base_dir(self) -> Path:
        """Lazily resolve base_dir on first access.

        If base_dir was not provided, tries to get it from bm.session_info.cache_dir.
        Falls back to default if bm is not available.

        Includes project_name for project isolation:
        - With bm: {cache_dir}/{project_name}/records
        - Without bm: {default_cache_dir}/records
        """
        if self._base_dir is None:
            if self._base_dir_config:
                # Expand paths if provided as string
                if isinstance(self._base_dir_config, str):
<<<<<<< HEAD
                    expanded = os.path.expandvars(os.path.expanduser(self._base_dir_config))
=======
                    expanded = os.path.expandvars(
                        os.path.expanduser(self._base_dir_config)
                    )
>>>>>>> origin/stable
                    self._base_dir = Path(expanded)
                else:
                    self._base_dir = Path(self._base_dir_config)
            else:
                # Try to get from bm.session_info.cache_dir and project_name if available
                try:
                    from buttermilk import bm

                    # Include project_name for project isolation
                    project_name = bm.session_info.project_name
<<<<<<< HEAD
                    self._base_dir = Path(bm.session_info.cache_dir) / project_name / "records"
=======
                    self._base_dir = (
                        Path(bm.session_info.cache_dir) / project_name / "records"
                    )
>>>>>>> origin/stable
                    logger.debug(
                        "📁 RecordCache using cache_dir with project isolation",
                        base_dir=str(self._base_dir),
                        project_name=project_name,
                    )
                except Exception:
                    # Fall back to default if bm not available
                    self._base_dir = _default_base_dir()
                    logger.debug(
                        "📁 RecordCache using default cache dir",
                        base_dir=str(self._base_dir),
                    )

            # Log initialization
            logger.debug(
                "🗂️  RecordCache base_dir resolved",
                base_dir=str(self._base_dir),
                enabled=self.enabled,
                working_dir=os.getcwd(),
                cache_env_var=os.getenv("BM_RECORD_CACHE_DIR"),
                disable_env_var=os.getenv("BM_DISABLE_RECORD_CACHE"),
            )

            # Create directory if enabled
            if self.enabled:
                try:
                    self._base_dir.mkdir(parents=True, exist_ok=True)
<<<<<<< HEAD
                    logger.debug("📁 Created cache base directory", base_dir=str(self._base_dir))
                except Exception as e:  # pragma: no cover
                    logger.debug(f"Could not create record cache base dir {self._base_dir}: {e}")
=======
                    logger.debug(
                        "📁 Created cache base directory", base_dir=str(self._base_dir)
                    )
                except Exception as e:  # pragma: no cover
                    logger.debug(
                        f"Could not create record cache base dir {self._base_dir}: {e}"
                    )
>>>>>>> origin/stable
                    self.enabled = False

        return self._base_dir

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
            exists=exists,
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
            logger.debug(
                "🚫 Cache version mismatch",
                record_id=record_id,
                stage=stage,
                path=str(path),
            )
            return None

        # Extract record data (remove metadata fields)
<<<<<<< HEAD
        data = {k: v for k, v in payload.items() if k not in ["_schema_version", "stage"]}
=======
        data = {
            k: v for k, v in payload.items() if k not in ["_schema_version", "stage"]
        }
>>>>>>> origin/stable
        if not isinstance(data, dict) or not data.get("record_id"):
            logger.debug(
                "🚫 Invalid record data in cache",
                record_id=record_id,
                stage=stage,
                path=str(path),
            )
            return None

        try:
            # Create BaseRecord directly from serialized data
            record = BaseRecord(**data)
        except Exception as e:  # pragma: no cover
<<<<<<< HEAD
            logger.debug("💥 Failed to rehydrate Record", record_id=record_id, error=str(e))
=======
            logger.debug(
                "💥 Failed to rehydrate Record", record_id=record_id, error=str(e)
            )
>>>>>>> origin/stable
            return None

        logger.debug(
            f"⚡ Cache hit for {stage},  loaded record {record_id}",
            record_id=record_id,
            stage=stage,
            path=str(path),
        )
        return record

<<<<<<< HEAD
    def save(self, record: BaseRecord, stage: str, include_chunks: bool = True, cache_key: str | None = None) -> bool:
=======
    def save(
        self, record: BaseRecord, stage: str, include_chunks: bool = True, cache_key: str | None = None
    ) -> bool:
>>>>>>> origin/stable
        """Save a record to cache.

        Args:
            record: The record to cache
            stage: Pipeline stage name
            include_chunks: Whether to include chunks in cache
            cache_key: Optional explicit cache key (defaults to record.record_id).
                       Use this for 1:N transformations where multiple outputs share
                       the same record_id but need separate cache entries.
        """
        if not self.enabled:
            logger.debug(
                "🚫 Cache disabled - not saving",
                record_id=getattr(record, "record_id", "unknown"),
                stage=stage,
            )
            return False
        if not record or not getattr(record, "record_id", None):
            logger.debug("🚫 Invalid record - not saving", record=record, stage=stage)
            return False

        stage_dir = self._stage_dir(stage)
        try:
            stage_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover
<<<<<<< HEAD
            logger.debug("💥 Could not create stage dir", stage_dir=str(stage_dir), error=str(e))
=======
            logger.debug(
                "💥 Could not create stage dir", stage_dir=str(stage_dir), error=str(e)
            )
>>>>>>> origin/stable
            return False

        # Use explicit cache_key if provided, otherwise use record_id
        effective_cache_key = cache_key if cache_key else record.record_id
        path = self._record_path(stage, effective_cache_key)
        tmp_path = path.with_suffix(".tmp")
        chunks_count = len(getattr(record, "chunks", []))

        try:
            # Store complete record data with chunks included
<<<<<<< HEAD
            record_data = scrub_serializable(record.model_dump()) if hasattr(record, "model_dump") else record
=======
            record_data = (
                scrub_serializable(record.model_dump())
                if hasattr(record, "model_dump")
                else record
            )
>>>>>>> origin/stable
            payload: dict[str, Any] = {
                "_schema_version": CACHE_VERSION,
                "stage": stage,
                "record_id": record.record_id,
                **record_data,  # Include all record data directly
            }
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            tmp_path.replace(path)

            logger.debug(
                "💾 Cached record",
                record_id=record.record_id,
                stage=stage,
                path=str(path),
                chunks_count=chunks_count,
                include_chunks=include_chunks,
            )
            return True
        except Exception as e:  # pragma: no cover
            logger.debug(
                "💥 Failed to cache record",
                record_id=record.record_id,
                stage=stage,
                error=str(e),
            )
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
