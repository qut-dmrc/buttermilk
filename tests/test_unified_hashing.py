"""Tests for the unified hashing system."""

import tempfile
from pathlib import Path

import pytest

from autogen_core.models import SystemMessage, UserMessage

from buttermilk._core.hashing import (
    compute_flow_hash,
    compute_ground_truth_hash,
    compute_message_hashes,
    compute_record_hash,
    compute_sha256_hash,
    compute_template_hash,
    compute_template_hash_from_file,
    extract_system_hash,
    normalize_flow_config,
)


class TestCoreHashFunction:
    """Test the core SHA256 hash computation."""

    def test_compute_sha256_hash_basic(self):
        """Test basic SHA256 computation."""
        content = "test content"
        hash_value = compute_sha256_hash(content)

        # Should be 64-character hex string
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_compute_sha256_hash_consistency(self):
        """Test that same content produces same hash."""
        content = "test content"
        hash1 = compute_sha256_hash(content)
        hash2 = compute_sha256_hash(content)

        assert hash1 == hash2

    def test_compute_sha256_hash_uniqueness(self):
        """Test that different content produces different hashes."""
        hash1 = compute_sha256_hash("content A")
        hash2 = compute_sha256_hash("content B")

        assert hash1 != hash2


class TestRecordHashing:
    """Test record content hashing."""

    def test_compute_record_hash(self):
        """Test record markdown hashing."""
        markdown = "# Test Record\n\nThis is test content."
        hash_value = compute_record_hash(markdown)

        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_record_hash_consistency(self):
        """Test that same markdown produces same hash."""
        markdown = "# Test Record\n\nContent here."
        hash1 = compute_record_hash(markdown)
        hash2 = compute_record_hash(markdown)

        assert hash1 == hash2

    def test_record_hash_uniqueness(self):
        """Test that different markdown produces different hashes."""
        hash1 = compute_record_hash("# Record A")
        hash2 = compute_record_hash("# Record B")

        assert hash1 != hash2


class TestGroundTruthHashing:
    """Test ground truth data hashing."""

    def test_compute_ground_truth_hash_none(self):
        """Test that None ground truth returns None hash."""
        hash_value = compute_ground_truth_hash(None)
        assert hash_value is None

    def test_compute_ground_truth_hash_dict(self):
        """Test ground truth dict hashing."""
        gt_data = {"answer": "test", "score": 5}
        hash_value = compute_ground_truth_hash(gt_data)

        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_compute_ground_truth_hash_list(self):
        """Test ground truth list hashing."""
        gt_data = ["answer1", "answer2"]
        hash_value = compute_ground_truth_hash(gt_data)

        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_ground_truth_hash_key_order_independence(self):
        """Test that key order doesn't affect hash."""
        gt1 = {"b": 2, "a": 1, "c": 3}
        gt2 = {"a": 1, "c": 3, "b": 2}

        hash1 = compute_ground_truth_hash(gt1)
        hash2 = compute_ground_truth_hash(gt2)

        assert hash1 == hash2

    def test_ground_truth_hash_uniqueness(self):
        """Test that different data produces different hashes."""
        hash1 = compute_ground_truth_hash({"answer": "A"})
        hash2 = compute_ground_truth_hash({"answer": "B"})

        assert hash1 != hash2

    def test_ground_truth_hash_complex_data(self):
        """Test hashing of complex nested data."""
        complex_gt = {
            "answers": ["A", "B", "C"],
            "metadata": {"source": "test", "nested": {"deep": "value"}},
            "scores": [1, 2, 3],
        }

        hash_value = compute_ground_truth_hash(complex_gt)
        assert len(hash_value) == 64


class TestTemplateHashing:
    """Test template content hashing."""

    def test_compute_template_hash(self):
        """Test template content hashing."""
        template_content = "Hello {{ name }}! Your score is {{ score }}."
        hash_value = compute_template_hash(template_content)

        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_template_hash_consistency(self):
        """Test that same template produces same hash."""
        template_content = "Template: {{ variable }}"
        hash1 = compute_template_hash(template_content)
        hash2 = compute_template_hash(template_content)

        assert hash1 == hash2

    def test_template_hash_uniqueness(self):
        """Test that different templates produce different hashes."""
        hash1 = compute_template_hash("Template A: {{ var }}")
        hash2 = compute_template_hash("Template B: {{ var }}")

        assert hash1 != hash2

    def test_compute_template_hash_from_file(self):
        """Test template hashing from file."""
        template_content = "Test template with {{ variable }}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja2", delete=False) as f:
            f.write(template_content)
            temp_path = f.name

        try:
            hash_value = compute_template_hash_from_file(temp_path)
            expected_hash = compute_template_hash(template_content)

            assert hash_value == expected_hash
        finally:
            Path(temp_path).unlink()

    def test_template_hash_from_file_not_found(self):
        """Test error handling for missing template file."""
        with pytest.raises(FileNotFoundError):
            compute_template_hash_from_file("/nonexistent/path.jinja2")


class TestFlowHashing:
    """Test flow configuration hashing."""

    def test_normalize_flow_config_basic(self):
        """Test basic flow config normalization."""
        flow_config = {
            "name": "test_flow",
            "description": "Test flow",
            "orchestrator": "TestOrchestrator",
            "parameters": {"criteria": ["test"]},
            "session_id": "runtime-123",  # Should be excluded
            "timestamp": "2024-01-01T00:00:00Z",  # Should be excluded
        }

        normalized = normalize_flow_config(flow_config)

        assert "name" in normalized
        assert "description" in normalized
        assert "orchestrator" in normalized
        assert "parameters" in normalized
        assert "session_id" not in normalized
        assert "timestamp" not in normalized

    def test_normalize_flow_config_agents(self):
        """Test agent configuration normalization."""
        flow_config = {
            "name": "test_flow",
            "agents": {
                "fetch": {
                    "name": "Fetch Agent",
                    "role": "FETCHER",
                    "template": "fetch_template",
                    "runtime_param": "should_be_excluded",
                },
                "judge": {"name": "Judge Agent", "role": "JUDGE"},
            },
        }

        normalized = normalize_flow_config(flow_config)

        assert "agents" in normalized
        assert "fetch" in normalized["agents"]
        assert "judge" in normalized["agents"]

        fetch_agent = normalized["agents"]["fetch"]
        assert "name" in fetch_agent
        assert "role" in fetch_agent
        assert "template" in fetch_agent
        assert "runtime_param" not in fetch_agent

    def test_normalize_flow_config_storage(self):
        """Test storage configuration normalization."""
        flow_config = {
            "name": "test_flow",
            "storage": {
                "data_source": "test_db",
                "save_output": "/tmp/output",  # Save config should be excluded
                "cache_config": {"enabled": True},
            },
        }

        normalized = normalize_flow_config(flow_config)

        assert "storage" in normalized
        assert "data_source" in normalized["storage"]
        assert "cache_config" in normalized["storage"]
        assert "save_output" not in normalized["storage"]

    def test_compute_flow_hash(self):
        """Test flow configuration hashing."""
        flow_config = {
            "name": "test_flow",
            "orchestrator": "TestOrchestrator",
            "parameters": {"criteria": ["test1", "test2"]},
            "agents": {"agent1": {"name": "Agent 1", "role": "TESTER"}},
        }

        hash_value = compute_flow_hash(flow_config)

        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_flow_hash_consistency(self):
        """Test that same flow config produces same hash."""
        flow_config = {"name": "test_flow", "parameters": {"criteria": ["test"]}}

        hash1 = compute_flow_hash(flow_config)
        hash2 = compute_flow_hash(flow_config)

        assert hash1 == hash2

    def test_flow_hash_uniqueness(self):
        """Test that different flow configs produce different hashes."""
        config1 = {"name": "flow_a", "parameters": {"criteria": ["test1"]}}
        config2 = {"name": "flow_b", "parameters": {"criteria": ["test2"]}}

        hash1 = compute_flow_hash(config1)
        hash2 = compute_flow_hash(config2)

        assert hash1 != hash2

    def test_flow_hash_excludes_runtime_data(self):
        """Test that runtime data doesn't affect flow hash."""
        base_config = {"name": "test_flow", "parameters": {"criteria": ["test"]}}

        config_with_runtime = {
            **base_config,
            "session_id": "runtime-123",
            "timestamp": "2024-01-01T00:00:00Z",
            "user_id": "user-456",
        }

        hash1 = compute_flow_hash(base_config)
        hash2 = compute_flow_hash(config_with_runtime)

        assert hash1 == hash2


class TestHashFormatConsistency:
    """Test that all hash functions produce consistent formats."""

    def test_all_hashes_same_format(self):
        """Test that all hash functions return 64-character hex strings."""
        # Record hash
        record_hash = compute_record_hash("test markdown")
        assert len(record_hash) == 64
        assert all(c in "0123456789abcdef" for c in record_hash)

        # Ground truth hash
        gt_hash = compute_ground_truth_hash({"test": "data"})
        assert len(gt_hash) == 64
        assert all(c in "0123456789abcdef" for c in gt_hash)

        # Template hash
        template_hash = compute_template_hash("test template")
        assert len(template_hash) == 64
        assert all(c in "0123456789abcdef" for c in template_hash)

        # Flow hash
        flow_hash = compute_flow_hash({"name": "test"})
        assert len(flow_hash) == 64
        assert all(c in "0123456789abcdef" for c in flow_hash)

    def test_no_prefixes_in_hashes(self):
        """Test that hashes don't include prefixes like 'sha256:'."""
        # All hashes should be plain hex strings without prefixes
        hashes = [
            compute_record_hash("test"),
            compute_ground_truth_hash({"test": "data"}),
            compute_template_hash("test template"),
            compute_flow_hash({"name": "test"}),
        ]

        for hash_value in hashes:
            assert not hash_value.startswith("sha256:")
            assert not hash_value.startswith("md5:")
            assert not hash_value.startswith("hash:")
            # Should be pure hex
            assert all(c in "0123456789abcdef" for c in hash_value)


class TestMessageHashing:
    """Test per-message hashing for experiment identity."""

    def test_basic_message_hashing(self):
        """Test hashing SystemMessage and UserMessage produces correct structure."""
        messages = [
            SystemMessage(content="You are a judge. Criteria: fairness"),
            UserMessage(content="Evaluate this article.", source="test"),
        ]
        hashes = compute_message_hashes(messages)

        assert len(hashes) == 2
        assert hashes[0]["role"] == "system"
        assert hashes[0]["index"] == 0
        assert len(hashes[0]["hash"]) == 64
        assert hashes[1]["role"] == "user"
        assert hashes[1]["index"] == 1
        assert len(hashes[1]["hash"]) == 64

    def test_message_hash_determinism(self):
        """Same message content produces the same hash."""
        content = "You are a judge evaluating toxicity."
        messages = [SystemMessage(content=content)]

        hash1 = compute_message_hashes(messages)[0]["hash"]
        hash2 = compute_message_hashes(messages)[0]["hash"]

        assert hash1 == hash2

    def test_different_criteria_different_system_hash(self):
        """Different criteria content produces different system_hash — the core use case."""
        tja_messages = [SystemMessage(content="Criteria: TJA guidelines for trans coverage")]
        glaad_messages = [SystemMessage(content="Criteria: GLAAD media reference guide")]

        tja_hashes = compute_message_hashes(tja_messages)
        glaad_hashes = compute_message_hashes(glaad_messages)

        tja_system = extract_system_hash(tja_hashes)
        glaad_system = extract_system_hash(glaad_hashes)

        assert tja_system != glaad_system

    def test_extract_system_hash_present(self):
        """extract_system_hash returns hash of first system message."""
        messages = [
            SystemMessage(content="system prompt"),
            UserMessage(content="user input", source="test"),
        ]
        hashes = compute_message_hashes(messages)
        system_hash = extract_system_hash(hashes)

        assert system_hash is not None
        assert system_hash == hashes[0]["hash"]

    def test_extract_system_hash_missing(self):
        """extract_system_hash returns None when no system message exists."""
        messages = [UserMessage(content="just a user message", source="test")]
        hashes = compute_message_hashes(messages)

        assert extract_system_hash(hashes) is None

    def test_extract_system_hash_empty_list(self):
        """extract_system_hash handles empty list."""
        assert extract_system_hash([]) is None

    def test_message_hashes_empty_messages(self):
        """compute_message_hashes handles empty message list."""
        assert compute_message_hashes([]) == []
