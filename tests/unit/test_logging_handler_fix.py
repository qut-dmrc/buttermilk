"""Test for the StructlogRichHandler fix to prevent record modification.

This test validates that the fix for the logging issue where logger.info()
was outputting plain text to file while logger.debug() was outputting structured JSON.
"""

import json
import logging
import tempfile
import uuid
from pathlib import Path

import pytest

from buttermilk._core.log import (
    logger,
    setup_console_logging,
    setup_file_logging,
    reset_logging_configuration,
)


class TestStructlogRichHandlerFix:
    """Test that StructlogRichHandler doesn't modify the original record."""

    def setup_method(self):
        """Reset logging configuration for each test."""
        reset_logging_configuration()

    def teardown_method(self):
        """Clean up logging configuration after each test."""
        reset_logging_configuration()

    def test_logger_info_produces_structured_json_in_file(self):
        """Test that logger.info() produces structured JSON in file after fix."""
        # Set up logging
        setup_console_logging(verbose=False)
        execution_context_id = f"test_info_{uuid.uuid4().hex[:8]}"
        log_files = setup_file_logging(execution_context_id=execution_context_id, verbose=False)

        log_file_path = Path(log_files[0])

        # Clear any existing content
        log_file_path.write_text("")

        # Test logger.info() with structured data
        test_message = f"Test info message {uuid.uuid4()}"
        logger.info(test_message, flow="test_flow", record_id="test_record", job_id="test_job")

        # Read the log file content
        log_content = log_file_path.read_text().strip()

        # Verify it's valid JSON
        assert log_content, "Log file should not be empty"

        # Parse each line as JSON (JSONL format)
        log_lines = [line.strip() for line in log_content.split('\n') if line.strip()]
        assert len(log_lines) >= 1, "Should have at least one log entry"

        # Find our test message
        test_entry = None
        for line in log_lines:
            try:
                entry = json.loads(line)
                if entry.get("event") == test_message:
                    test_entry = entry
                    break
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON in log file: {repr(line)}")

        assert test_entry is not None, f"Test message not found in log. Content: {log_content}"

        # Verify structured data is present
        assert test_entry["event"] == test_message
        assert test_entry["flow"] == "test_flow"
        assert test_entry["record_id"] == "test_record"
        assert test_entry["job_id"] == "test_job"
        assert "timestamp" in test_entry
        assert test_entry["level"] == "info"

    def test_logger_debug_still_produces_structured_json_in_file(self):
        """Test that logger.debug() still works correctly after fix."""
        # Set up logging with verbose to capture debug messages
        setup_console_logging(verbose=True)
        execution_context_id = f"test_debug_{uuid.uuid4().hex[:8]}"
        log_files = setup_file_logging(execution_context_id=execution_context_id, verbose=True)

        log_file_path = Path(log_files[0])

        # Clear any existing content
        log_file_path.write_text("")

        # Test logger.debug() with structured data
        test_message = f"Test debug message {uuid.uuid4()}"
        logger.debug(test_message, flow="test_flow", record_id="test_record", job_id="test_job")

        # Read the log file content
        log_content = log_file_path.read_text().strip()

        # Verify it's valid JSON
        assert log_content, "Log file should not be empty"

        # Parse each line as JSON (JSONL format)
        log_lines = [line.strip() for line in log_content.split('\n') if line.strip()]
        assert len(log_lines) >= 1, "Should have at least one log entry"

        # Find our test message
        test_entry = None
        for line in log_lines:
            try:
                entry = json.loads(line)
                if entry.get("event") == test_message:
                    test_entry = entry
                    break
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON in log file: {repr(line)}")

        assert test_entry is not None, f"Test message not found in log. Content: {log_content}"

        # Verify structured data is present
        assert test_entry["event"] == test_message
        assert test_entry["flow"] == "test_flow"
        assert test_entry["record_id"] == "test_record"
        assert test_entry["job_id"] == "test_job"
        assert "timestamp" in test_entry
        assert test_entry["level"] == "debug"

    def test_both_info_and_debug_produce_consistent_json_format(self):
        """Test that both info and debug messages produce consistent JSON structure."""
        # Set up logging with verbose to capture both levels
        setup_console_logging(verbose=True)
        execution_context_id = f"test_both_{uuid.uuid4().hex[:8]}"
        log_files = setup_file_logging(execution_context_id=execution_context_id, verbose=True)

        log_file_path = Path(log_files[0])

        # Clear any existing content
        log_file_path.write_text("")

        # Test both logger.info() and logger.debug()
        info_message = f"Test info message {uuid.uuid4()}"
        debug_message = f"Test debug message {uuid.uuid4()}"

        logger.info(info_message, flow="test_flow", type="info_test")
        logger.debug(debug_message, flow="test_flow", type="debug_test")

        # Read the log file content
        log_content = log_file_path.read_text().strip()

        # Parse all log entries
        log_lines = [line for line in log_content.split('\n') if line.strip()]

        info_entry = None
        debug_entry = None

        for line in log_lines:
            try:
                entry = json.loads(line)
                if entry.get("event") == info_message:
                    info_entry = entry
                elif entry.get("event") == debug_message:
                    debug_entry = entry
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON in log file: {repr(line)}")

        # Verify both entries were found
        assert info_entry is not None, f"Info message not found in log. Content: {log_content}"
        assert debug_entry is not None, f"Debug message not found in log. Content: {log_content}"

        # Verify both have consistent structure
        for entry in [info_entry, debug_entry]:
            assert "event" in entry
            assert "flow" in entry
            assert "type" in entry
            assert "timestamp" in entry
            assert "level" in entry

        # Verify correct levels
        assert info_entry["level"] == "info"
        assert debug_entry["level"] == "debug"

        # Verify structured data is preserved
        assert info_entry["flow"] == "test_flow"
        assert debug_entry["flow"] == "test_flow"
        assert info_entry["type"] == "info_test"
        assert debug_entry["type"] == "debug_test"