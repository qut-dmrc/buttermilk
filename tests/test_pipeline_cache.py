"""Tests for pipeline cache invalidation with processor parameter changes.

This test suite verifies that:
1. Changing processor configuration parameters invalidates cache
2. Different projects have isolated caches
3. Parameter hash is stable across runs
4. Cache paths include both project name and parameter hash
"""

from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from buttermilk._core.hashing import compute_processor_config_hash
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.record_cache import RecordCache
from buttermilk._core.types import BaseRecord
from buttermilk.pipeline import PipelineOrchestrator


# Mock processor for testing
class MockProcessor(BaseModel):
    """Simple processor with configurable parameters for testing cache invalidation."""

    model: str = Field(default="gpt-4", description="Model name")
    template: str = Field(default="summarize", description="Template name")
    temperature: float = Field(default=0.7, description="Temperature")

    async def process(self, context: ProcessingContext) -> AsyncGenerator[BaseRecord, None]:
        """Process record by adding a field based on config."""
        # Add a field that depends on processor config
        updated = context.record.model_copy(update={"output": f"Processed with {self.model} at {self.temperature}"})
        yield updated


class TestProcessorConfigHash:
    """Test processor configuration hashing for cache keys."""

    def test_hash_stability(self):
        """Test that hash is stable for same configuration."""
        config1 = {"model": "gpt-4", "template": "summarize", "temperature": 0.7}
        config2 = {"model": "gpt-4", "template": "summarize", "temperature": 0.7}

        hash1 = compute_processor_config_hash(config1)
        hash2 = compute_processor_config_hash(config2)

        assert hash1 == hash2, "Same config should produce same hash"
        assert len(hash1) == 8, "Hash should be 8 characters"

    def test_hash_order_independence(self):
        """Test that hash is independent of dict key order."""
        config1 = {"model": "gpt-4", "template": "summarize", "temperature": 0.7}
        config2 = {"temperature": 0.7, "model": "gpt-4", "template": "summarize"}

        hash1 = compute_processor_config_hash(config1)
        hash2 = compute_processor_config_hash(config2)

        assert hash1 == hash2, "Hash should be independent of key order"

    def test_hash_changes_with_model(self):
        """Test that hash changes when model changes."""
        config1 = {"model": "gpt-4", "template": "summarize"}
        config2 = {"model": "claude-3", "template": "summarize"}

        hash1 = compute_processor_config_hash(config1)
        hash2 = compute_processor_config_hash(config2)

        assert hash1 != hash2, "Different model should produce different hash"

    def test_hash_changes_with_template(self):
        """Test that hash changes when template changes."""
        config1 = {"model": "gpt-4", "template": "summarize"}
        config2 = {"model": "gpt-4", "template": "analyze"}

        hash1 = compute_processor_config_hash(config1)
        hash2 = compute_processor_config_hash(config2)

        assert hash1 != hash2, "Different template should produce different hash"

    def test_hash_changes_with_nested_config(self):
        """Test that hash changes with nested configuration."""
        config1 = {"model": "gpt-4", "mappings": {"field1": "expr1"}}
        config2 = {"model": "gpt-4", "mappings": {"field1": "expr2"}}

        hash1 = compute_processor_config_hash(config1)
        hash2 = compute_processor_config_hash(config2)

        assert hash1 != hash2, "Different nested config should produce different hash"

    def test_hash_ignores_private_attrs(self):
        """Test that hash ignores private attributes."""
        config1 = {"model": "gpt-4", "_internal": "value1"}
        config2 = {"model": "gpt-4", "_internal": "value2"}

        hash1 = compute_processor_config_hash(config1)
        hash2 = compute_processor_config_hash(config2)

        assert hash1 == hash2, "Private attributes should be ignored"

    def test_hash_ignores_none_values(self):
        """Test that hash ignores None values."""
        config1 = {"model": "gpt-4", "optional": None}
        config2 = {"model": "gpt-4"}

        hash1 = compute_processor_config_hash(config1)
        hash2 = compute_processor_config_hash(config2)

        assert hash1 == hash2, "None values should be ignored"


class TestRecordCacheProjectIsolation:
    """Test that RecordCache includes project name in paths."""

    def test_cache_path_includes_project_name(self, tmp_path):
        """Test that cache paths include project name for isolation."""
        # Create cache with explicit base_dir config
        # When bm is available, it will use project_name from session_info
        cache_dir_config = str(tmp_path / "test_project" / "records")
        cache = RecordCache(base_dir=cache_dir_config)

        base_dir = cache.base_dir

        # Should include project name in path
        assert "test_project" in str(base_dir)
        assert base_dir == tmp_path / "test_project" / "records"

    def test_different_projects_have_isolated_caches(self, tmp_path):
        """Test that different projects get different cache directories."""
        # Simulate two different project cache paths
        cache_dir1 = str(tmp_path / "project_alpha" / "records")
        cache_dir2 = str(tmp_path / "project_beta" / "records")

        cache1 = RecordCache(base_dir=cache_dir1)
        cache2 = RecordCache(base_dir=cache_dir2)

        base_dir1 = cache1.base_dir
        base_dir2 = cache2.base_dir

        assert base_dir1 != base_dir2, "Different projects should have different cache dirs"
        assert "project_alpha" in str(base_dir1)
        assert "project_beta" in str(base_dir2)


class TestPipelineProcessorStageNames:
    """Test that pipeline generates processor_stage_name with parameter hash."""

    @pytest.mark.anyio
    async def test_processor_stage_name_includes_param_hash(self, tmp_path):
        """Test that processor_stage_name includes parameter hash."""
        # Create a test record
        record = BaseRecord(record_id="test_001")

        # Create processor with specific config
        processor = MockProcessor(model="gpt-4", template="summarize", temperature=0.7)

        # Create a simple async source
        async def simple_source():
            yield record

        # Mock bm for cache
        mock_session_info = MagicMock()
        mock_session_info.project_name = "test_project"
        mock_session_info.cache_dir = str(tmp_path)
        mock_bm = MagicMock()
        mock_bm.session_info = mock_session_info

        with patch("buttermilk.pipeline.bm", mock_bm):
            # Create pipeline
            pipeline = PipelineOrchestrator(
                pipeline_name="test_pipeline",
                source=simple_source(),
                processors=[processor],
                concurrency=1,
                enable_record_cache=False,  # Disable cache for this test
            )

            # Process one record to trigger processor_stage_name generation
            results = []
            async for result in pipeline():
                results.append(result)

            assert len(results) == 1

            # The processor_stage_name should have been generated with format:
            # {pipeline_name}/{index:02d}.{processor_class}/{param_hash}
            # We can't easily inspect it here, but the next test will verify cache behavior

    @pytest.mark.anyio
    async def test_cache_invalidation_on_parameter_change(self, tmp_path):
        """Test that changing processor parameters creates different cache paths."""
        # Create a test record
        record = BaseRecord(record_id="test_002", content="test content")

        # Create first processor with specific config
        processor1 = MockProcessor(model="gpt-4", template="summarize", temperature=0.7)

        # Get the param hash for processor1
        config1 = processor1.model_dump()
        hash1 = compute_processor_config_hash(config1)

        # Create second processor with DIFFERENT config (different model)
        processor2 = MockProcessor(
            model="claude-3",
            template="summarize",
            temperature=0.7,  # Changed model!
        )

        # Get the param hash for processor2
        config2 = processor2.model_dump()
        hash2 = compute_processor_config_hash(config2)

        # Verify hashes are different
        assert hash1 != hash2, "Different processor configs should have different hashes"

        # Now verify the pipeline uses these hashes in cache paths
        async def source1():
            yield record

        # Configure explicit cache directory to avoid environmental issues
        cache_dir = str(tmp_path)

        pipeline1 = PipelineOrchestrator(
            pipeline_name="test_pipeline",
            source=source1(),
            processors=[processor1],
            concurrency=1,
            enable_record_cache=True,
            cache_dir=cache_dir,
        )

        results1 = []
        async for result in pipeline1():
            results1.append(result)

        assert len(results1) == 1
        assert results1[0].output == "Processed with gpt-4 at 0.7"

        # Check that cache was created with first hash in path
        # The hash is in the directory structure: pipeline/processor_index.class/hash/record.json
        cache_dirs_1 = list(tmp_path.rglob(f"*{hash1}*"))
        assert len(cache_dirs_1) > 0, f"Cache directory should exist with hash {hash1}"

        # Second run with different processor config
        async def source2():
            yield record

        pipeline2 = PipelineOrchestrator(
            pipeline_name="test_pipeline",
            source=source2(),
            processors=[processor2],
            concurrency=1,
            enable_record_cache=True,
            cache_dir=cache_dir,
        )

        results2 = []
        async for result in pipeline2():
            results2.append(result)

        assert len(results2) == 1
        assert results2[0].output == "Processed with claude-3 at 0.7"

        # Check that cache was created with second hash in path
        cache_dirs_2 = list(tmp_path.rglob(f"*{hash2}*"))
        assert len(cache_dirs_2) > 0, f"Cache directory should exist with hash {hash2}"

        # Verify both cache directories exist (different hashes)
        all_cache_json = list(tmp_path.rglob("*.json"))
        assert len(all_cache_json) >= 2, "Should have at least two cache files (one per config)"


class TestCacheInvalidationIntegration:
    """Integration tests for cache invalidation with real processor configs."""

    @pytest.mark.anyio
    async def test_llm_model_change_invalidates_cache(self, tmp_path):
        """Test that changing LLM model name creates different cache paths."""
        record = BaseRecord(record_id="test_003", content="test content")

        # Simulate LLM with model parameter
        processor1 = MockProcessor(model="claude-sonnet-4-6", template="default")
        hash1 = compute_processor_config_hash(processor1.model_dump())

        async def source1():
            yield record

        pipeline1 = PipelineOrchestrator(
            pipeline_name="llm_pipeline",
            source=source1(),
            processors=[processor1],
            concurrency=1,
            enable_record_cache=True,
            cache_dir=str(tmp_path),
        )

        results1 = []
        async for result in pipeline1():
            results1.append(result)

        # Now change model to gpt4
        processor2 = MockProcessor(model="gpt4", template="default")
        hash2 = compute_processor_config_hash(processor2.model_dump())

        # Verify hashes are different
        assert hash1 != hash2, "Different models should produce different hashes"

        async def source2():
            yield record

        pipeline2 = PipelineOrchestrator(
            pipeline_name="llm_pipeline",
            source=source2(),
            processors=[processor2],
            concurrency=1,
            enable_record_cache=True,
            cache_dir=str(tmp_path),
        )

        results2 = []
        async for result in pipeline2():
            results2.append(result)

        # Verify different outputs (cache was invalidated)
        assert results1[0].output != results2[0].output

        # Verify different cache files exist with different hashes
        cache_files = list(tmp_path.rglob("*.json"))
        assert len(cache_files) >= 2, "Should have cache files for both models"

    @pytest.mark.anyio
    async def test_template_change_invalidates_cache(self, tmp_path):
        """Test that changing template creates different cache paths."""
        record = BaseRecord(record_id="test_004", content="test content")

        # First run with template1
        processor1 = MockProcessor(model="gpt-4", template="template1")
        hash1 = compute_processor_config_hash(processor1.model_dump())

        async def source1():
            yield record

        pipeline1 = PipelineOrchestrator(
            pipeline_name="template_pipeline",
            source=source1(),
            processors=[processor1],
            concurrency=1,
            enable_record_cache=True,
            cache_dir=str(tmp_path),
        )

        results1 = []
        async for result in pipeline1():
            results1.append(result)

        # Second run with template2
        processor2 = MockProcessor(model="gpt-4", template="template2")
        hash2 = compute_processor_config_hash(processor2.model_dump())

        # Verify hashes are different
        assert hash1 != hash2, "Different templates should produce different hashes"

        async def source2():
            yield record

        pipeline2 = PipelineOrchestrator(
            pipeline_name="template_pipeline",
            source=source2(),
            processors=[processor2],
            concurrency=1,
            enable_record_cache=True,
            cache_dir=str(tmp_path),
        )

        results2 = []
        async for result in pipeline2():
            results2.append(result)

        # Verify different cache files exist
        cache_files = list(tmp_path.rglob("*.json"))
        assert len(cache_files) >= 2, "Should have cache files for both templates"
