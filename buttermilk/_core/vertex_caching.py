"""Vertex AI context caching for criteria templates.

This module provides cache management for Vertex AI's explicit context caching,
enabling efficient reuse of large criteria prompts across multiple records.

Usage:
    from buttermilk._core.vertex_caching import CriteriaCacheManager
    from buttermilk import bm

    cache_manager = CriteriaCacheManager(client=bm.genai)
    cache_name = cache_manager.get_or_create_cache(
        model="gemini-2.5-flash",
        system_instruction="You are a judge...",
        criteria_content="Evaluate based on...",
        display_name="tja_criteria_A",
    )
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

if TYPE_CHECKING:
    pass  # Type hints imported at runtime

logger = logging.getLogger(__name__)


class CriteriaCacheManager(BaseModel):
    """Manage Vertex AI context caches for criteria templates.

    Creates and retrieves explicit context caches for large, repeated criteria
    content. Caches are keyed by content hash to avoid duplicates.

    Attributes:
        client: Google GenAI client (from bm.genai)
        ttl: Cache time-to-live in seconds format (e.g., "3600s" for 1 hour)
        model: Default model for cache creation

    Example:
        cache_manager = CriteriaCacheManager(client=bm.genai, model="gemini-2.5-flash")
        cache_name = cache_manager.get_or_create_cache(
            system_instruction="You are a content moderator...",
            criteria_content="Evaluate the following content for hate speech...",
            display_name="hate_speech_criteria",
        )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any = Field(..., description="Google GenAI client instance")
    ttl: str = Field(default="3600s", description="Cache TTL (e.g., '3600s' for 1 hour)")
    model: str = Field(
        default="gemini-2.5-flash",
        description="Model to use for cache creation",
    )

    # Internal cache registry: hash -> CachedContent
    _cache_registry: dict[str, Any] = PrivateAttr(default_factory=dict)
    # Map display_name -> cache_name for lookup
    _name_to_cache: dict[str, str] = PrivateAttr(default_factory=dict)

    def _compute_cache_key(
        self,
        model: str,
        system_instruction: str | None,
        criteria_content: str,
    ) -> str:
        """Compute a deterministic cache key from content.

        Args:
            model: Model identifier
            system_instruction: System prompt content
            criteria_content: Criteria template content

        Returns:
            SHA256 hash of the combined content
        """
        components = [
            model,
            system_instruction or "",
            criteria_content,
        ]
        combined = "\n---\n".join(components)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

    def get_or_create_cache(
        self,
        criteria_content: str,
        display_name: str,
        *,
        system_instruction: str | None = None,
        model: str | None = None,
    ) -> str:
        """Get existing cache or create new one for criteria content.

        Args:
            criteria_content: The criteria template text to cache
            display_name: Human-readable name for the cache
            system_instruction: Optional system prompt to include in cache
            model: Model to use (defaults to self.model)

        Returns:
            Cache resource name (e.g., "projects/.../cachedContents/abc123")

        Raises:
            RuntimeError: If cache creation fails
        """
        from google.genai.types import Content, CreateCachedContentConfig, Part

        effective_model = model or self.model
        cache_key = self._compute_cache_key(effective_model, system_instruction, criteria_content)

        # Check local registry first
        if cache_key in self._cache_registry:
            cached = self._cache_registry[cache_key]
            logger.debug(f"Cache hit for {display_name}: {cached.name}")
            return cached.name

        # Check if we have it by display_name (might be from previous session)
        if display_name in self._name_to_cache:
            cache_name = self._name_to_cache[display_name]
            try:
                # Verify cache still exists
                cached = self.client.caches.get(name=cache_name)
                self._cache_registry[cache_key] = cached
                logger.debug(f"Retrieved existing cache {display_name}: {cache_name}")
                return cache_name
            except Exception:
                # Cache expired or deleted, will recreate
                del self._name_to_cache[display_name]

        # Create new cache
        logger.info(f"Creating new cache for {display_name} (model={effective_model})")

        # Build content structure for caching
        # The criteria content becomes a user message that will be cached
        contents = [
            Content(
                role="user",
                parts=[Part(text=criteria_content)],
            )
        ]

        try:
            config = CreateCachedContentConfig(
                contents=contents,
                display_name=display_name,
                ttl=self.ttl,
            )

            if system_instruction:
                config.system_instruction = system_instruction

            cached = self.client.caches.create(
                model=effective_model,
                config=config,
            )

            self._cache_registry[cache_key] = cached
            self._name_to_cache[display_name] = cached.name

            logger.info(
                f"Created cache {display_name}: {cached.name}",
                extra={
                    "cache_name": cached.name,
                    "model": effective_model,
                    "ttl": self.ttl,
                },
            )

            return cached.name

        except Exception as e:
            raise RuntimeError(f"Failed to create cache for {display_name}: {e}") from e

    def get_cache_info(self, cache_name: str) -> dict[str, Any]:
        """Get information about a cache.

        Args:
            cache_name: Cache resource name

        Returns:
            Dictionary with cache metadata
        """
        try:
            cached = self.client.caches.get(name=cache_name)
            return {
                "name": cached.name,
                "display_name": cached.display_name,
                "model": cached.model,
                "create_time": str(cached.create_time) if cached.create_time else None,
                "expire_time": str(cached.expire_time) if cached.expire_time else None,
                "usage_metadata": cached.usage_metadata,
            }
        except Exception as e:
            logger.warning(f"Failed to get cache info for {cache_name}: {e}")
            return {"name": cache_name, "error": str(e)}

    def delete_cache(self, cache_name: str) -> bool:
        """Delete a cache.

        Args:
            cache_name: Cache resource name to delete

        Returns:
            True if deleted, False if not found or failed
        """
        try:
            self.client.caches.delete(name=cache_name)

            # Remove from local registries
            keys_to_remove = [k for k, v in self._cache_registry.items() if v.name == cache_name]
            for key in keys_to_remove:
                del self._cache_registry[key]

            names_to_remove = [k for k, v in self._name_to_cache.items() if v == cache_name]
            for name in names_to_remove:
                del self._name_to_cache[name]

            logger.info(f"Deleted cache: {cache_name}")
            return True

        except Exception as e:
            logger.warning(f"Failed to delete cache {cache_name}: {e}")
            return False

    def list_caches(self) -> list[dict[str, Any]]:
        """List all caches for the current project.

        Returns:
            List of cache info dictionaries
        """
        try:
            caches = list(self.client.caches.list())
            return [
                {
                    "name": c.name,
                    "display_name": c.display_name,
                    "model": c.model,
                    "expire_time": str(c.expire_time) if c.expire_time else None,
                }
                for c in caches
            ]
        except Exception as e:
            logger.warning(f"Failed to list caches: {e}")
            return []

    def cleanup_expired(self) -> int:
        """Remove expired caches from local registry.

        Returns:
            Number of entries removed
        """
        removed = 0
        for cache_key in list(self._cache_registry.keys()):
            cached = self._cache_registry[cache_key]
            try:
                # Try to get cache - if it fails, it's expired/deleted
                self.client.caches.get(name=cached.name)
            except Exception:
                del self._cache_registry[cache_key]
                # Also remove from name mapping
                names_to_remove = [k for k, v in self._name_to_cache.items() if v == cached.name]
                for name in names_to_remove:
                    del self._name_to_cache[name]
                removed += 1

        if removed:
            logger.info(f"Cleaned up {removed} expired cache entries")
        return removed
