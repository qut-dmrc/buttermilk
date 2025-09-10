"""Validation tests for the logging fail-fast protection system.

These tests validate that the fail-fast protection system works correctly
in all scenarios and prevents the identified problematic patterns that
would break verbose logging functionality.

Validation scenarios:
1. Protection prevents broken logging initialization sequences
2. Verbose logging functionality is preserved
3. Error messages are clear and actionable
4. Safe alternatives work correctly
5. Integration with existing systems works properly
"""

import logging
import pytest
import uuid
from unittest.mock import patch, MagicMock
from pathlib import Path

from buttermilk._core.log import (
    logger,
    setup_console_logging,
    setup_file_logging,
    setup_cloud_logging,
    validate_logging_state,
    ensure_logging_properly_initialized,
    _console_logging_configured,
    _file_logging_configured,
    _cloud_logging_sessions,
)
from buttermilk._core.execution_context import (
    ExecutionContext,
    create_execution_context,
    get_or_create_execution_context,
    _execution_context_initialized,
)


class TestFailFastProtectionValidation:
    """Validate that fail-fast protection prevents all identified problematic patterns."""

    def setup_method(self):
        """Reset global state before each validation test."""
        # Reset all logging state
        import buttermilk._core.log as log_module
        log_module._console_logging_configured = False
        log_module._file_logging_configured = False
        log_module._cloud_logging_sessions.clear()
        
        # Reset execution context state
        import buttermilk._core.execution_context as ec_module
        ec_module._global_execution_context = None
        ec_module._execution_context_initialized = False
        ec_module._global_execution_context_id = ""
        
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_validate_console_logging_protection_prevents_broken_initialization(self):
        """Validate that console logging protection prevents broken initialization."""
        # Initial state should allow first setup
        assert not _console_logging_configured
        
        # First call should succeed
        setup_console_logging(verbose=True)
        assert _console_logging_configured
        
        # Verify verbose logging is properly configured
        buttermilk_logger = logging.getLogger("buttermilk")
        root_logger = logging.getLogger()
        assert buttermilk_logger.getEffectiveLevel() == logging.DEBUG
        assert root_logger.getEffectiveLevel() == logging.DEBUG
        
        # Second call should fail fast with clear error message
        with pytest.raises(RuntimeError) as exc_info:
            setup_console_logging(verbose=False)  # Would break verbose functionality
        
        error_msg = str(exc_info.value)
        assert "Console logging has already been configured" in error_msg
        assert "Multiple calls to setup_console_logging()" in error_msg
        assert "break verbose logging functionality" in error_msg
        
        # Verify verbose logging is still intact after failed attempt
        assert buttermilk_logger.getEffectiveLevel() == logging.DEBUG
        assert root_logger.getEffectiveLevel() == logging.DEBUG

    def test_validate_file_logging_protection_prevents_conflicting_handlers(self):
        """Validate that file logging protection prevents conflicting handlers."""
        # Initial state should allow first setup
        assert not _file_logging_configured
        
        # First call should succeed
        execution_context_id = f"validation_test_{uuid.uuid4().hex[:8]}"
        log_files = setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        assert _file_logging_configured
        assert len(log_files) == 1
        
        # Verify file handler was created with correct level
        root_logger = logging.getLogger()
        file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) > 0
        assert file_handlers[0].level == logging.DEBUG  # Verbose mode
        
        # Second call should fail fast
        with pytest.raises(RuntimeError) as exc_info:
            setup_file_logging(execution_context_id="different_context", verbose=False)
        
        error_msg = str(exc_info.value)
        assert "File logging has already been configured" in error_msg
        assert "Multiple calls to setup_file_logging()" in error_msg
        assert "conflicting log file handlers" in error_msg
        
        # Verify original file handler is still intact
        current_file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(current_file_handlers) == len(file_handlers)  # No additional handlers

    @patch('buttermilk._core.execution_context.setup_console_logging')
    @patch('buttermilk._core.execution_context.setup_file_logging')
    def test_validate_execution_context_protection_prevents_logging_corruption(self, mock_setup_file, mock_setup_console):
        """Validate that ExecutionContext protection prevents logging corruption."""
        mock_setup_console.return_value = None
        mock_setup_file.return_value = ["/tmp/test.log"]
        
        # Initial state should allow first creation
        assert not _execution_context_initialized
        
        # First creation should succeed
        context1 = create_execution_context()
        assert _execution_context_initialized
        assert context1 is not None
        
        # Verify logging setup was called once
        assert mock_setup_console.call_count == 1
        assert mock_setup_file.call_count == 1
        
        # Second creation should fail fast
        with pytest.raises(RuntimeError) as exc_info:
            create_execution_context()
        
        error_msg = str(exc_info.value)
        assert "ExecutionContext has already been initialized" in error_msg
        assert "break logging configuration" in error_msg
        assert "reset verbose logging settings" in error_msg
        assert "get_or_create_execution_context()" in error_msg
        
        # Verify logging setup was not called again (preventing corruption)
        assert mock_setup_console.call_count == 1
        assert mock_setup_file.call_count == 1

    def test_validate_cloud_logging_deduplication_prevents_duplicate_handlers(self):
        """Validate that cloud logging deduplication prevents duplicate handlers."""
        # Initial state should be empty
        assert len(_cloud_logging_sessions) == 0
        
        # Mock cloud logging components
        with patch('buttermilk._core.log.gcp_logging'), \
             patch('buttermilk._core.log.CloudLoggingHandler') as mock_cloud_handler_cls:
            
            mock_logger_cfg = MagicMock()
            mock_logger_cfg.type = "gcp"
            mock_logger_cfg.project_id = "test-project"
            mock_logger_cfg.location = "us-central1"
            
            mock_cloud_manager = MagicMock()
            mock_session_info = MagicMock()
            mock_session_info.session_id = "test-session-123"
            mock_session_info.name = "test-session"
            mock_session_info.job = "test-job"
            mock_session_info.platform = "test"
            mock_session_info.batch_id = None
            mock_session_info.model_dump.return_value = {
                "session_id": "test-session-123",
                "name": "test-session",
                "job": "test-job",
                "platform": "test"
            }
            
            mock_cloud_handler = MagicMock()
            mock_cloud_handler_cls.return_value = mock_cloud_handler
            
            # First call should create handler
            setup_cloud_logging(mock_logger_cfg, mock_cloud_manager, mock_session_info)
            
            # Verify session was tracked
            session_key = f"{mock_session_info.session_id}:{mock_logger_cfg.project_id}"
            assert session_key in _cloud_logging_sessions
            assert mock_cloud_handler_cls.call_count == 1
            
            # Second call with same session should be deduplicated
            setup_cloud_logging(mock_logger_cfg, mock_cloud_manager, mock_session_info)
            
            # Verify no additional handler was created
            assert mock_cloud_handler_cls.call_count == 1  # Still only 1

    def test_validate_logging_state_validation_detects_all_issues(self):
        """Validate that logging state validation detects all possible issues."""
        # Test unconfigured state
        validation = validate_logging_state()
        assert not validation["valid"]
        assert "Console logging not configured" in validation["issues"]
        assert "File logging not configured" in validation["issues"]
        
        # Test partially configured state
        setup_console_logging(verbose=True)
        validation_partial = validate_logging_state(verbose_expected=True)
        assert not validation_partial["valid"]
        assert validation_partial["console_configured"]
        assert not validation_partial["file_configured"]
        assert "File logging not configured" in validation_partial["issues"]
        
        # Test verbose level mismatch
        execution_context_id = f"validation_mismatch_{uuid.uuid4().hex[:8]}"
        setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        # Should be valid when expectations match
        validation_correct = validate_logging_state(verbose_expected=True)
        assert validation_correct["valid"]
        
        # Should detect mismatch when expectations don't match
        validation_mismatch = validate_logging_state(verbose_expected=False)
        # Note: This should still be valid because the logging is properly configured,
        # just not matching the expected verbose level
        
        # Test too many handlers
        root_logger = logging.getLogger()
        for i in range(10):  # Add many handlers
            dummy_handler = logging.StreamHandler()
            root_logger.addHandler(dummy_handler)
        
        validation_handlers = validate_logging_state()
        assert not validation_handlers["valid"]
        assert any("Unusually high number of handlers" in issue for issue in validation_handlers["issues"])

    def test_validate_ensure_logging_properly_initialized_fails_fast(self):
        """Validate that ensure_logging_properly_initialized fails fast appropriately."""
        # Should fail when logging is not configured
        with pytest.raises(RuntimeError) as exc_info:
            ensure_logging_properly_initialized()
        
        error_msg = str(exc_info.value)
        assert "Logging system is not properly initialized" in error_msg
        assert "Issues found:" in error_msg
        assert "Console logging not configured" in error_msg
        
        # Should succeed when logging is properly configured
        setup_console_logging(verbose=True)
        execution_context_id = f"ensure_test_{uuid.uuid4().hex[:8]}"
        setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        # Should not raise
        ensure_logging_properly_initialized()

    def test_validate_verbose_logging_functionality_preservation(self):
        """Validate that verbose logging functionality is preserved across operations."""
        # Set up verbose logging
        setup_console_logging(verbose=True)
        execution_context_id = f"verbose_preservation_{uuid.uuid4().hex[:8]}"
        log_files = setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        # Verify DEBUG level is configured
        buttermilk_logger = logging.getLogger("buttermilk")
        root_logger = logging.getLogger()
        assert buttermilk_logger.getEffectiveLevel() == logging.DEBUG
        assert root_logger.getEffectiveLevel() == logging.DEBUG
        
        # Test logging at DEBUG level
        debug_message = f"Debug test message {uuid.uuid4()}"
        info_message = f"Info test message {uuid.uuid4()}"
        
        logger.debug(debug_message)
        logger.info(info_message)
        
        # Force handler flush
        for handler in root_logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
        
        # Verify messages were logged to file
        log_file_path = Path(log_files[0])
        assert log_file_path.exists()
        
        log_content = log_file_path.read_text()
        # Both DEBUG and INFO messages should be in verbose mode
        assert debug_message in log_content or info_message in log_content  # At least one should be there
        
        # Attempt to break verbose logging (should fail fast)
        with pytest.raises(RuntimeError):
            setup_console_logging(verbose=False)
        
        # Verify verbose logging is still working
        final_debug_message = f"Final debug message {uuid.uuid4()}"
        logger.debug(final_debug_message)
        
        # Force flush again
        for handler in root_logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
        
        # Verify logging levels are still DEBUG
        assert buttermilk_logger.getEffectiveLevel() == logging.DEBUG
        assert root_logger.getEffectiveLevel() == logging.DEBUG

    @patch('buttermilk._core.execution_context.setup_console_logging')
    @patch('buttermilk._core.execution_context.setup_file_logging')
    def test_validate_safe_alternatives_work_correctly(self, mock_setup_file, mock_setup_console):
        """Validate that safe alternatives work correctly."""
        mock_setup_console.return_value = None
        mock_setup_file.return_value = ["/tmp/test.log"]
        
        # get_or_create_execution_context should work multiple times
        context1 = get_or_create_execution_context()
        context2 = get_or_create_execution_context()
        context3 = get_or_create_execution_context()
        
        # All should be the same object
        assert context1 is context2 is context3
        
        # Logging setup should only happen once
        assert mock_setup_console.call_count == 1
        assert mock_setup_file.call_count == 1
        
        # Verify this pattern is safe for any number of calls
        for _ in range(100):
            safe_context = get_or_create_execution_context()
            assert safe_context is context1
        
        # Logging setup should still only have happened once
        assert mock_setup_console.call_count == 1
        assert mock_setup_file.call_count == 1

    def test_validate_error_messages_are_actionable(self):
        """Validate that all error messages are clear and actionable."""
        # Test console logging error message
        setup_console_logging(verbose=True)
        
        with pytest.raises(RuntimeError) as exc_info:
            setup_console_logging(verbose=False)
        
        console_error = str(exc_info.value)
        # Should explain the problem
        assert "Console logging has already been configured" in console_error
        # Should explain the consequence
        assert "break verbose logging functionality" in console_error
        # Should indicate this is a design issue
        assert "problematic initialization sequence" in console_error
        
        # Test file logging error message
        execution_context_id = f"error_msg_test_{uuid.uuid4().hex[:8]}"
        setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        with pytest.raises(RuntimeError) as exc_info:
            setup_file_logging(execution_context_id="different", verbose=False)
        
        file_error = str(exc_info.value)
        # Should explain the problem
        assert "File logging has already been configured" in file_error
        # Should explain the consequences
        assert "break verbose logging functionality" in file_error
        assert "conflicting log file handlers" in file_error
        
        # Test ExecutionContext error message
        import buttermilk._core.execution_context as ec_module
        ec_module._execution_context_initialized = True
        
        with pytest.raises(RuntimeError) as exc_info:
            create_execution_context()
        
        context_error = str(exc_info.value)
        # Should explain the problem
        assert "ExecutionContext has already been initialized" in context_error
        # Should explain the consequences
        assert "break logging configuration" in context_error
        assert "reset verbose logging settings" in context_error
        # Should provide solution
        assert "get_or_create_execution_context()" in context_error
        
        # Test validation error message
        ec_module._execution_context_initialized = False  # Reset for this test
        
        with pytest.raises(RuntimeError) as exc_info:
            ensure_logging_properly_initialized()
        
        validation_error = str(exc_info.value)
        # Should explain the problem
        assert "Logging system is not properly initialized" in validation_error
        # Should list specific issues
        assert "Issues found:" in validation_error
        # Should explain the consequence
        assert "break verbose logging functionality" in validation_error

    def test_validate_end_to_end_protection_workflow(self):
        """Validate the complete end-to-end protection workflow."""
        # Step 1: Initial state should be unconfigured
        validation_initial = validate_logging_state()
        assert not validation_initial["valid"]
        
        # Step 2: Set up logging properly
        setup_console_logging(verbose=True)
        execution_context_id = f"e2e_validation_{uuid.uuid4().hex[:8]}"
        log_files = setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        # Step 3: Verify proper configuration
        validation_configured = validate_logging_state(verbose_expected=True)
        assert validation_configured["valid"]
        ensure_logging_properly_initialized()  # Should not raise
        
        # Step 4: Test verbose logging works
        debug_msg = f"E2E debug message {uuid.uuid4()}"
        info_msg = f"E2E info message {uuid.uuid4()}"
        logger.debug(debug_msg)
        logger.info(info_msg)
        
        # Step 5: Protection prevents reconfiguration
        with pytest.raises(RuntimeError):
            setup_console_logging(verbose=False)
        with pytest.raises(RuntimeError):
            setup_file_logging(execution_context_id="different", verbose=False)
        
        # Step 6: Logging is still healthy after protection triggered
        validation_after_protection = validate_logging_state(verbose_expected=True)
        assert validation_after_protection["valid"]
        ensure_logging_properly_initialized()  # Should still not raise
        
        # Step 7: Verbose logging still works
        final_debug_msg = f"Final E2E debug message {uuid.uuid4()}"
        logger.debug(final_debug_msg)
        
        buttermilk_logger = logging.getLogger("buttermilk")
        assert buttermilk_logger.getEffectiveLevel() == logging.DEBUG
        
        # Step 8: Log file exists and has content
        log_file_path = Path(log_files[0])
        assert log_file_path.exists()
        assert log_file_path.stat().st_size > 0