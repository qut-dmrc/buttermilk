"""Integration tests for logging fail-fast protection system.

These tests validate that the fail-fast protection works correctly in real-world
scenarios where multiple components interact with the logging system.

The integration tests cover:
1. BM session creation with logging protection
2. Multi-session scenarios and protection
3. Cloud logging deduplication in practice
4. Verbose logging preservation across operations
"""

import logging
import uuid
from unittest.mock import MagicMock, patch

import pytest

from buttermilk._core.log import (
    ensure_logging_properly_initialized,
    logger,
    setup_console_logging,
    setup_file_logging,
    validate_logging_state,
)


class TestVerboseLoggingPreservation:
    """Test that verbose logging settings are preserved across operations."""

    def setup_method(self):
        """Reset global state before each test."""
        # Reset all logging state
        import buttermilk._core.log as log_module
        log_module._console_logging_configured = False
        log_module._file_logging_configured = False
        log_module._cloud_logging_sessions.clear()
        
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_verbose_logging_preserved_during_session_operations(self):
        """Test that verbose logging settings remain intact during normal operations."""
        # Set up verbose logging
        setup_console_logging(verbose=True)
        execution_context_id = f"verbose_preserve_{uuid.uuid4().hex[:8]}"
        setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        # Verify verbose logging is configured
        validation_initial = validate_logging_state(verbose_expected=True)
        assert validation_initial["valid"] is True
        
        # Verify DEBUG level is set
        buttermilk_logger = logging.getLogger("buttermilk")
        root_logger = logging.getLogger()
        assert buttermilk_logger.getEffectiveLevel() == logging.DEBUG
        assert root_logger.getEffectiveLevel() == logging.DEBUG
        
        # Simulate various operations that might interfere with logging
        logger.debug("Debug message during operation")
        logger.info("Info message during operation")
        logger.warning("Warning message during operation")
        
        # Attempt operations that should fail fast and not affect logging
        with pytest.raises(RuntimeError):
            setup_console_logging(verbose=False)  # Should fail fast
            
        with pytest.raises(RuntimeError):
            setup_file_logging(execution_context_id="different", verbose=False)  # Should fail fast
        
        # Verify verbose logging is still intact
        validation_final = validate_logging_state(verbose_expected=True)
        assert validation_final["valid"] is True
        assert buttermilk_logger.getEffectiveLevel() == logging.DEBUG
        assert root_logger.getEffectiveLevel() == logging.DEBUG
        
        # Verify we can still log at all levels
        logger.debug("Final debug message")
        logger.info("Final info message")

    def test_non_verbose_logging_preserved_during_operations(self):
        """Test that non-verbose logging settings remain intact during operations."""
        # Set up non-verbose logging
        setup_console_logging(verbose=False)
        execution_context_id = f"non_verbose_preserve_{uuid.uuid4().hex[:8]}"
        setup_file_logging(execution_context_id=execution_context_id, verbose=False)
        
        # Verify non-verbose logging is configured
        validation_initial = validate_logging_state(verbose_expected=False)
        assert validation_initial["valid"] is True
        
        # Verify INFO level is set for buttermilk logger
        buttermilk_logger = logging.getLogger("buttermilk")
        assert buttermilk_logger.getEffectiveLevel() == logging.INFO
        
        # Simulate operations
        logger.info("Info message during operation")
        logger.warning("Warning message during operation")
        
        # Attempt operations that should fail fast
        with pytest.raises(RuntimeError):
            setup_console_logging(verbose=True)  # Should fail fast
            
        with pytest.raises(RuntimeError):
            setup_file_logging(execution_context_id="different", verbose=True)  # Should fail fast
        
        # Verify non-verbose logging is still intact
        validation_final = validate_logging_state(verbose_expected=False)
        assert validation_final["valid"] is True
        assert buttermilk_logger.getEffectiveLevel() == logging.INFO


class TestCloudLoggingIntegration:
    """Test cloud logging deduplication in integration scenarios."""

    def setup_method(self):
        """Reset global state before each test."""
        # Reset cloud logging sessions
        import buttermilk._core.log as log_module
        log_module._cloud_logging_sessions.clear()
        
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    @patch("buttermilk._core.log.gcp_logging")
    @patch("buttermilk._core.log.CloudLoggingHandler")
    def test_cloud_logging_deduplication_across_sessions(self, mock_cloud_handler_cls, mock_gcp_logging):
        """Test that cloud logging is properly deduplicated across multiple sessions."""
        from buttermilk._core.log import _cloud_logging_sessions, setup_cloud_logging
        
        # Mock cloud logging components
        mock_logger_cfg = MagicMock()
        mock_logger_cfg.type = "gcp"
        mock_logger_cfg.project_id = "test-project"
        mock_logger_cfg.location = "us-central1"
        
        mock_cloud_manager = MagicMock()
        mock_cloud_handler = MagicMock()
        mock_cloud_handler_cls.return_value = mock_cloud_handler
        
        # Create first session with cloud logging
        mock_session_info_1 = MagicMock()
        mock_session_info_1.session_id = "session-123"
        mock_session_info_1.name = "test-session-1"
        mock_session_info_1.job = "test-job"
        mock_session_info_1.platform = "test"
        mock_session_info_1.batch_id = None
        mock_session_info_1.model_dump.return_value = {
            "session_id": "session-123",
            "name": "test-session-1",
            "job": "test-job",
            "platform": "test"
        }
        
        # First call should set up cloud logging
        setup_cloud_logging(mock_logger_cfg, mock_cloud_manager, mock_session_info_1)
        
        # Verify session was tracked
        session_key_1 = f"{mock_session_info_1.session_id}:{mock_logger_cfg.project_id}"
        assert session_key_1 in _cloud_logging_sessions
        assert mock_cloud_handler_cls.call_count == 1
        
        # Create second session with same session ID (should be deduplicated)
        mock_session_info_2 = MagicMock()
        mock_session_info_2.session_id = "session-123"  # Same session ID
        mock_session_info_2.name = "test-session-1"     # Same name
        mock_session_info_2.job = "test-job"
        mock_session_info_2.platform = "test"
        mock_session_info_2.batch_id = None
        mock_session_info_2.model_dump.return_value = {
            "session_id": "session-123",
            "name": "test-session-1",
            "job": "test-job",
            "platform": "test"
        }
        
        # Second call should be deduplicated (no new handler)
        setup_cloud_logging(mock_logger_cfg, mock_cloud_manager, mock_session_info_2)
        
        # Verify no additional handler was created
        assert mock_cloud_handler_cls.call_count == 1  # Still only 1
        
        # Create third session with different session ID (should create new handler)
        mock_session_info_3 = MagicMock()
        mock_session_info_3.session_id = "session-456"  # Different session ID
        mock_session_info_3.name = "test-session-2"     # Different name
        mock_session_info_3.job = "test-job"
        mock_session_info_3.platform = "test"
        mock_session_info_3.batch_id = None
        mock_session_info_3.model_dump.return_value = {
            "session_id": "session-456",
            "name": "test-session-2",
            "job": "test-job",
            "platform": "test"
        }
        
        # Third call should create new handler for different session
        setup_cloud_logging(mock_logger_cfg, mock_cloud_manager, mock_session_info_3)
        
        # Verify new handler was created for different session
        assert mock_cloud_handler_cls.call_count == 2
        
        # Verify both sessions are tracked
        session_key_3 = f"{mock_session_info_3.session_id}:{mock_logger_cfg.project_id}"
        assert session_key_1 in _cloud_logging_sessions
        assert session_key_3 in _cloud_logging_sessions
        assert session_key_1 != session_key_3


class TestErrorRecoveryAndValidation:
    """Test error recovery and validation in integration scenarios."""

    def setup_method(self):
        """Reset global state before each test."""
        # Reset all logging state
        import buttermilk._core.log as log_module
        log_module._console_logging_configured = False
        log_module._file_logging_configured = False
        log_module._cloud_logging_sessions.clear()
        
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_logging_validation_detects_broken_state(self):
        """Test that validation correctly detects broken logging states."""
        # Initially, logging should be unconfigured
        validation = validate_logging_state()
        assert validation["valid"] is False
        assert "Console logging not configured" in validation["issues"]
        assert "File logging not configured" in validation["issues"]
        
        # ensure_logging_properly_initialized should fail
        with pytest.raises(RuntimeError) as exc_info:
            ensure_logging_properly_initialized()
        
        error_msg = str(exc_info.value)
        assert "Logging system is not properly initialized" in error_msg
        assert "Console logging not configured" in error_msg

    def test_logging_validation_after_partial_setup(self):
        """Test validation when logging is only partially configured."""
        # Set up only console logging
        setup_console_logging(verbose=True)
        
        validation = validate_logging_state(verbose_expected=True)
        assert validation["valid"] is False  # Still invalid because file logging missing
        assert validation["console_configured"] is True
        assert validation["file_configured"] is False
        assert "File logging not configured" in validation["issues"]
        
        # Complete the setup
        execution_context_id = f"partial_setup_{uuid.uuid4().hex[:8]}"
        setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        # Now validation should pass
        validation_complete = validate_logging_state(verbose_expected=True)
        assert validation_complete["valid"] is True
        assert validation_complete["console_configured"] is True
        assert validation_complete["file_configured"] is True
        assert len(validation_complete["issues"]) == 0
        
        # ensure_logging_properly_initialized should now succeed
        ensure_logging_properly_initialized()

    def test_verbose_level_mismatch_detection(self):
        """Test that validation detects verbose level mismatches."""
        # Set up logging with verbose=False
        setup_console_logging(verbose=False)
        execution_context_id = f"level_mismatch_{uuid.uuid4().hex[:8]}"
        setup_file_logging(execution_context_id=execution_context_id, verbose=False)
        
        # Validate expecting verbose=True (should detect mismatch)
        validation = validate_logging_state(verbose_expected=True)
        assert validation["valid"] is False
        
        # Should detect both root logger and buttermilk logger level issues
        issues_text = " ".join(validation["issues"])
        assert "Verbose mode expected but root logger level is" in issues_text
        assert "Verbose mode expected but buttermilk logger level is" in issues_text
        
        # But validation without verbose expectation should pass
        validation_no_verbose = validate_logging_state(verbose_expected=False)
        assert validation_no_verbose["valid"] is True

    def test_integration_with_real_logging_operations(self):
        """Test integration with real logging operations."""
        # Set up verbose logging
        setup_console_logging(verbose=True)
        execution_context_id = f"real_logging_{uuid.uuid4().hex[:8]}"
        log_files = setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        # Verify setup is valid
        validation = validate_logging_state(verbose_expected=True)
        assert validation["valid"] is True
        
        # Perform real logging operations
        test_debug_msg = f"Debug message {uuid.uuid4()}"
        test_info_msg = f"Info message {uuid.uuid4()}"
        test_warning_msg = f"Warning message {uuid.uuid4()}"
        test_error_msg = f"Error message {uuid.uuid4()}"
        
        logger.debug(test_debug_msg)
        logger.info(test_info_msg)
        logger.warning(test_warning_msg)
        logger.error(test_error_msg)
        
        # Ensure all handlers are flushed
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if hasattr(handler, "flush"):
                handler.flush()
        
        # Validation should still pass after real logging
        validation_after = validate_logging_state(verbose_expected=True)
        assert validation_after["valid"] is True
        
        # ensure_logging_properly_initialized should still work
        ensure_logging_properly_initialized()
        
        # Verify log file exists and has content
        assert len(log_files) == 1
        from pathlib import Path
        log_file_path = Path(log_files[0])
        assert log_file_path.exists()
        assert log_file_path.stat().st_size > 0  # Has content


class TestFailFastIntegrationExamples:
    """Integration test examples demonstrating proper fail-fast usage patterns.
    
    These tests serve as documentation and examples for developers showing
    how to properly use the fail-fast protection system in real scenarios.
    """

    def setup_method(self):
        """Reset global state before each test."""
        # Reset all state
        import buttermilk._core.log as log_module
        log_module._console_logging_configured = False
        log_module._file_logging_configured = False
        log_module._cloud_logging_sessions.clear()
        
        import buttermilk._core.execution_context as ec_module
        ec_module._global_execution_context = None
        ec_module._execution_context_initialized = False
        ec_module._global_execution_context_id = ""
        
        # Clear handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_verbose_logging_workflow_example(self):
        """Example of proper verbose logging workflow."""
        # Step 1: Set up verbose logging at application start
        setup_console_logging(verbose=True)
        execution_context_id = f"verbose_workflow_{uuid.uuid4().hex[:8]}"
        setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        # Step 2: Validate logging is properly configured
        validation = validate_logging_state(verbose_expected=True)
        assert validation["valid"] is True
        
        # Step 3: Use logging throughout application lifecycle
        logger.debug("Application starting with verbose logging")
        logger.info("Processing data")
        logger.debug("Debug information for troubleshooting")
        logger.warning("Non-critical warning")
        logger.info("Operation completed successfully")
        
        # Step 4: Attempts to reconfigure should fail fast (protection working)
        with pytest.raises(RuntimeError, match="Console logging has already been configured"):
            setup_console_logging(verbose=False)
        
        with pytest.raises(RuntimeError, match="File logging has already been configured"):
            setup_file_logging(execution_context_id="different", verbose=False)
        
        # Step 5: Logging should still work perfectly after failed reconfiguration attempts
        logger.debug("Logging still works after protection triggered")
        
        # Step 6: Final validation confirms everything is still intact
        final_validation = validate_logging_state(verbose_expected=True)
        assert final_validation["valid"] is True
        
        buttermilk_logger = logging.getLogger("buttermilk")
        assert buttermilk_logger.getEffectiveLevel() == logging.DEBUG

    def test_error_handling_and_recovery_example(self):
        """Example of error handling and recovery workflow."""
        # Step 1: Check initial state (should be unconfigured)
        initial_validation = validate_logging_state()
        assert initial_validation["valid"] is False
        
        # Step 2: Detect the issue using validation
        issues = initial_validation["issues"]
        assert "Console logging not configured" in issues
        assert "File logging not configured" in issues
        
        # Step 3: Fix the issues by proper setup
        setup_console_logging(verbose=True)
        execution_context_id = f"error_recovery_{uuid.uuid4().hex[:8]}"
        setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        # Step 4: Verify the fixes worked
        fixed_validation = validate_logging_state(verbose_expected=True)
        assert fixed_validation["valid"] is True
        assert len(fixed_validation["issues"]) == 0
        
        # Step 5: Use ensure_logging_properly_initialized for ongoing checks
        ensure_logging_properly_initialized()  # Should not raise
        
        # Step 6: Demonstrate that protection prevents further issues
        with pytest.raises(RuntimeError):
            setup_console_logging(verbose=False)  # Would break verbose logging
        
        # Step 7: Verify system is still healthy after protection triggered
        final_validation = validate_logging_state(verbose_expected=True)
        assert final_validation["valid"] is True
        ensure_logging_properly_initialized()  # Should still not raise
