"""Tests for GroupchatProcessor metadata merge functionality.

Part of RFC #311 Option A: Parameters in record.metadata are merged into
RunRequest.inputs for agent access.
"""

from buttermilk.processors.unified_processors import GroupchatProcessor


class TestGroupchatMetadataMerge:
    """Test metadata merge into RunRequest.inputs."""

    def test_merge_metadata_to_inputs_adds_metadata_keys(self):
        """Verify metadata keys are added to inputs dict."""
        # GroupchatProcessor needs minimal config
        processor = GroupchatProcessor(
            flow_name="test",
            flow_config={},  # Minimal config for test
        )

        base_inputs = {
            "record_id": "test-001",
            "record": {"content": "test"},
        }
        metadata = {
            "criteria": "accuracy",
            "model": "gpt-4",
            "score": 0.95,
        }

        result = processor._merge_metadata_to_inputs(base_inputs, metadata)

        # Metadata keys should be in result
        assert result["criteria"] == "accuracy"
        assert result["model"] == "gpt-4"
        assert result["score"] == 0.95
        # Base inputs preserved
        assert result["record_id"] == "test-001"
        assert result["record"] == {"content": "test"}

    def test_merge_metadata_reserved_keys_not_overwritten(self):
        """Verify record_id and record are not overwritten by metadata."""
        processor = GroupchatProcessor(flow_name="test", flow_config={})

        base_inputs = {
            "record_id": "original-id",
            "record": {"original": "data"},
        }
        metadata = {
            "record_id": "SHOULD_NOT_OVERRIDE",
            "record": {"SHOULD": "NOT_OVERRIDE"},
            "criteria": "valid_key",
        }

        result = processor._merge_metadata_to_inputs(base_inputs, metadata)

        # Reserved keys should NOT be overwritten
        assert result["record_id"] == "original-id"
        assert result["record"] == {"original": "data"}
        # Non-reserved key should be added
        assert result["criteria"] == "valid_key"

    def test_merge_metadata_skips_non_serializable(self):
        """Verify callables and classes are skipped."""
        processor = GroupchatProcessor(flow_name="test", flow_config={})

        base_inputs = {"record_id": "test"}
        metadata = {
            "criteria": "valid",
            "model_class": str,  # A class - should be skipped
            "callback": lambda x: x,  # Callable - should be skipped
        }

        result = processor._merge_metadata_to_inputs(base_inputs, metadata)

        assert result["criteria"] == "valid"
        assert "model_class" not in result
        assert "callback" not in result

    def test_merge_metadata_handles_none(self):
        """Verify None metadata returns base inputs unchanged."""
        processor = GroupchatProcessor(flow_name="test", flow_config={})

        base_inputs = {"record_id": "test", "record": {}}

        result = processor._merge_metadata_to_inputs(base_inputs, None)

        assert result == base_inputs
