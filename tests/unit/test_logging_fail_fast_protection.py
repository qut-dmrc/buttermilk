"""Test fail-fast logging protection system.

These tests validate the fail-fast protection mechanisms that prevent broken
logging initialization, which could break verbose logging functionality.

The fail-fast system includes:
1. Global state tracking to prevent multiple calls to setup functions
2. ExecutionContext protection to prevent multiple initialization
3. Cloud logging deduplication per session
4. Validation functions that detect configuration issues
"""

import logging
import uuid
from unittest.mock import MagicMock, patch

import pytest

from buttermilk._core.execution_context import (
    ExecutionContext,
    create_execution_context,
    get_or_create_execution_context,
)
from buttermilk._core.log import (
    _cloud_logging_sessions,
    _console_logging_configured,
    _file_logging_configured,
    ensure_logging_properly_initialized,
    logger,
    setup_cloud_logging,
    setup_console_logging,
    setup_file_logging,
    validate_logging_state,
)


class TestSetupConsoleLoggingFailFast:
    """Test fail-fast protection for console logging setup."""

    def setup_method(self):
        """Reset global state before each test."""
        # Reset console logging state
        import buttermilk._core.log as log_module
        log_module._console_logging_configured = False
        
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_setup_console_logging_first_call_succeeds(self):
        """Test that the first call to setup_console_logging succeeds."""
        # Should work without error
        setup_console_logging(verbose=False)
        
        # Verify logging is configured
        assert _console_logging_configured is True
        
        # Verify handlers were added
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0

    def test_setup_console_logging_second_call_fails_fast(self):
        """Test that second call to setup_console_logging raises RuntimeError."""
        # First call should succeed
        setup_console_logging(verbose=False)
        
        # Second call should fail fast
        with pytest.raises(RuntimeError) as exc_info:
            setup_console_logging(verbose=True)
            
        error_msg = str(exc_info.value)
        assert "Console logging has already been configured" in error_msg
        assert "Multiple calls to setup_console_logging()" in error_msg
        assert "break verbose logging functionality" in error_msg
        assert "problematic initialization sequence" in error_msg

    def test_setup_console_logging_verbose_settings_preserved(self):
        """Test that verbose settings are properly configured on first call."""
        # Test verbose=True
        setup_console_logging(verbose=True)
        
        buttermilk_logger = logging.getLogger("buttermilk")
        root_logger = logging.getLogger()
        
        # Verify DEBUG level is set for verbose mode
        assert buttermilk_logger.getEffectiveLevel() == logging.DEBUG
        assert root_logger.getEffectiveLevel() == logging.DEBUG

    def test_setup_console_logging_non_verbose_settings(self):
        """Test that non-verbose settings are properly configured."""
        # Reset state
        self.setup_method()
        
        # Test verbose=False
        setup_console_logging(verbose=False)
        
        buttermilk_logger = logging.getLogger("buttermilk")
        
        # Verify INFO level is set for non-verbose mode
        assert buttermilk_logger.getEffectiveLevel() == logging.INFO


class TestSetupFileLoggingFailFast:
    """Test fail-fast protection for file logging setup."""

    def setup_method(self):
        """Reset global state before each test."""
        # Reset file logging state
        import buttermilk._core.log as log_module
        log_module._file_logging_configured = False
        
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_setup_file_logging_first_call_succeeds(self):
        """Test that the first call to setup_file_logging succeeds."""
        execution_context_id = f"test_first_call_{uuid.uuid4().hex[:8]}"
        
        # Should work without error
        log_files = setup_file_logging(execution_context_id=execution_context_id, verbose=False)
        
        # Verify logging is configured
        assert _file_logging_configured is True
        
        # Verify log file was created
        assert len(log_files) == 1
        assert execution_context_id in log_files[0]
        
        # Verify handlers were added
        root_logger = logging.getLogger()
        file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) > 0

    def test_setup_file_logging_second_call_fails_fast(self):
        """Test that second call to setup_file_logging raises RuntimeError."""
        execution_context_id = f"test_second_call_{uuid.uuid4().hex[:8]}"
        
        # First call should succeed
        setup_file_logging(execution_context_id=execution_context_id, verbose=False)
        
        # Second call should fail fast
        with pytest.raises(RuntimeError) as exc_info:
            setup_file_logging(execution_context_id=f"different_{execution_context_id}", verbose=True)
            
        error_msg = str(exc_info.value)
        assert "File logging has already been configured" in error_msg
        assert "Multiple calls to setup_file_logging()" in error_msg
        assert "break verbose logging functionality" in error_msg
        assert "conflicting log file handlers" in error_msg
        assert "problematic initialization sequence" in error_msg

    def test_setup_file_logging_verbose_creates_debug_file(self):
        """Test that verbose=True creates file with DEBUG level."""
        execution_context_id = f"test_verbose_debug_{uuid.uuid4().hex[:8]}"
        
        log_files = setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        # Verify file was created
        assert len(log_files) == 1
        
        # Verify DEBUG level is configured
        root_logger = logging.getLogger()
        buttermilk_logger = logging.getLogger("buttermilk")
        
        assert root_logger.getEffectiveLevel() == logging.DEBUG
        assert buttermilk_logger.getEffectiveLevel() == logging.DEBUG
        
        # Find file handler and verify its level
        file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) > 0
        assert file_handlers[0].level == logging.DEBUG

    def test_setup_file_logging_non_verbose_creates_info_file(self):
        """Test that verbose=False creates file with INFO level."""
        # Reset state
        self.setup_method()
        
        execution_context_id = f"test_non_verbose_info_{uuid.uuid4().hex[:8]}"
        
        log_files = setup_file_logging(execution_context_id=execution_context_id, verbose=False)
        
        # Verify file was created
        assert len(log_files) == 1
        
        # Find file handler and verify its level
        root_logger = logging.getLogger()
        file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) > 0
        assert file_handlers[0].level == logging.INFO


class TestCloudLoggingDeduplication:
    """Test cloud logging deduplication protection."""

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
    def test_setup_cloud_logging_first_call_succeeds(self, mock_cloud_handler_cls, mock_gcp_logging):
        """Test that first call to setup_cloud_logging succeeds."""
        # Mock objects
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
        
        # Should succeed without error
        setup_cloud_logging(mock_logger_cfg, mock_cloud_manager, mock_session_info)
        
        # Verify session was tracked
        session_key = f"{mock_session_info.session_id}:{mock_logger_cfg.project_id}"
        assert session_key in _cloud_logging_sessions
        
        # Verify cloud handler was created and added
        mock_cloud_handler_cls.assert_called_once()
        root_logger = logging.getLogger()
        root_logger.addHandler.assert_not_called()  # Mocked, but we can verify the call happened

    @patch("buttermilk._core.log.gcp_logging")
    @patch("buttermilk._core.log.CloudLoggingHandler")
    def test_setup_cloud_logging_duplicate_session_skipped(self, mock_cloud_handler_cls, mock_gcp_logging):
        """Test that duplicate cloud logging setup for same session is skipped."""
        # Mock objects
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
        
        # First call
        setup_cloud_logging(mock_logger_cfg, mock_cloud_manager, mock_session_info)
        
        # Reset mock to track second call
        mock_cloud_handler_cls.reset_mock()
        
        # Second call should be skipped
        setup_cloud_logging(mock_logger_cfg, mock_cloud_manager, mock_session_info)
        
        # Verify cloud handler was NOT created again
        mock_cloud_handler_cls.assert_not_called()

    def test_cloud_logging_session_tracking_different_sessions(self):
        """Test that different sessions can have cloud logging configured."""
        # Mock first session
        session1_key = "session1:project1"
        _cloud_logging_sessions.add(session1_key)
        
        # Mock second session with different session ID
        mock_logger_cfg = MagicMock()
        mock_logger_cfg.type = "gcp"
        mock_logger_cfg.project_id = "project1"
        
        mock_session_info = MagicMock()
        mock_session_info.session_id = "session2"  # Different session
        
        session2_key = f"{mock_session_info.session_id}:{mock_logger_cfg.project_id}"
        
        # Verify different sessions have different keys
        assert session1_key != session2_key
        assert session2_key not in _cloud_logging_sessions


class TestExecutionContextFailFast:
    """Test fail-fast protection for ExecutionContext creation."""

    def setup_method(self):
        """Reset global ExecutionContext state before each test."""
        # Reset execution context state
        import buttermilk._core.execution_context as ec_module
        ec_module._global_execution_context = None
        ec_module._execution_context_initialized = False
        
        # Reset logging state to avoid interference
        import buttermilk._core.log as log_module
        log_module._console_logging_configured = False
        log_module._file_logging_configured = False
        
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    @patch("buttermilk._core.execution_context.setup_console_logging")
    @patch("buttermilk._core.execution_context.setup_file_logging")
    def test_create_execution_context_first_call_succeeds(self, mock_setup_file, mock_setup_console):
        """Test that first call to create_execution_context succeeds."""
        mock_setup_console.return_value = None
        mock_setup_file.return_value = ["/tmp/test.log"]
        
        # Should work without error
        context = create_execution_context()
        
        # Verify context was created
        assert context is not None
        assert isinstance(context, ExecutionContext)
        
        # Verify logging setup was called
        mock_setup_console.assert_called_once()
        mock_setup_file.assert_called_once()

    @patch("buttermilk._core.execution_context.setup_console_logging")
    @patch("buttermilk._core.execution_context.setup_file_logging")
    def test_create_execution_context_second_call_fails_fast(self, mock_setup_file, mock_setup_console):
        """Test that second call to create_execution_context raises RuntimeError."""
        mock_setup_console.return_value = None
        mock_setup_file.return_value = ["/tmp/test.log"]
        
        # First call should succeed
        create_execution_context()
        
        # Second call should fail fast
        with pytest.raises(RuntimeError) as exc_info:
            create_execution_context()
            
        error_msg = str(exc_info.value)
        assert "ExecutionContext has already been initialized" in error_msg
        assert "Creating multiple ExecutionContext instances will break logging configuration" in error_msg
        assert "reset verbose logging settings" in error_msg
        assert "loss of execution context state" in error_msg
        assert "get_or_create_execution_context()" in error_msg

    @patch("buttermilk._core.execution_context.setup_console_logging")
    @patch("buttermilk._core.execution_context.setup_file_logging")
    def test_get_or_create_execution_context_safe_multiple_calls(self, mock_setup_file, mock_setup_console):
        """Test that get_or_create_execution_context is safe for multiple calls."""
        mock_setup_console.return_value = None
        mock_setup_file.return_value = ["/tmp/test.log"]
        
        # First call should create context
        context1 = get_or_create_execution_context()
        assert context1 is not None
        
        # Second call should return existing context
        context2 = get_or_create_execution_context()
        assert context2 is context1  # Same object
        
        # Verify logging setup was only called once
        mock_setup_console.assert_called_once()
        mock_setup_file.assert_called_once()

    @patch("buttermilk._core.execution_context.setup_console_logging")
    @patch("buttermilk._core.execution_context.setup_file_logging")
    def test_get_or_create_execution_context_multiple_calls_with_different_kwargs(self, mock_setup_file, mock_setup_console):
        """Test that get_or_create_execution_context ignores kwargs on subsequent calls."""
        mock_setup_console.return_value = None
        mock_setup_file.return_value = ["/tmp/test.log"]
        
        # First call with specific kwargs
        context1 = get_or_create_execution_context(clouds=[])
        assert context1 is not None
        
        # Second call with different kwargs should return same context
        context2 = get_or_create_execution_context(clouds=["different"])
        assert context2 is context1  # Same object, kwargs ignored


class TestLoggingValidation:
    """Test logging validation functions."""

    def setup_method(self):
        """Reset global state before each test."""
        # Reset logging state
        import buttermilk._core.log as log_module
        log_module._console_logging_configured = False
        log_module._file_logging_configured = False
        
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_validate_logging_state_unconfigured(self):
        """Test validation when logging is not configured."""
        validation = validate_logging_state()
        
        assert validation["console_configured"] is False
        assert validation["file_configured"] is False
        assert validation["valid"] is False
        assert "Console logging not configured" in validation["issues"]
        assert "File logging not configured" in validation["issues"]

    def test_validate_logging_state_properly_configured(self):
        """Test validation when logging is properly configured."""
        # Set up logging properly
        setup_console_logging(verbose=True)
        
        # Mock file logging configured
        import buttermilk._core.log as log_module
        log_module._file_logging_configured = True
        
        validation = validate_logging_state(verbose_expected=True)
        
        assert validation["console_configured"] is True
        assert validation["file_configured"] is True
        assert validation["valid"] is True
        assert len(validation["issues"]) == 0

    def test_validate_logging_state_verbose_level_mismatch(self):
        """Test validation detects verbose level mismatches."""
        # Set up console logging with verbose=False
        setup_console_logging(verbose=False)
        
        # Mock file logging configured
        import buttermilk._core.log as log_module
        log_module._file_logging_configured = True
        
        # Validate expecting verbose=True (should find mismatch)
        validation = validate_logging_state(verbose_expected=True)
        
        assert validation["valid"] is False
        assert any("Verbose mode expected but root logger level is" in issue for issue in validation["issues"])
        assert any("Verbose mode expected but buttermilk logger level is" in issue for issue in validation["issues"])

    def test_validate_logging_state_too_many_handlers(self):
        """Test validation detects excessive number of handlers."""
        # Set up normal logging
        setup_console_logging(verbose=False)
        
        # Add many dummy handlers to trigger the threshold
        root_logger = logging.getLogger()
        for i in range(6):  # Add 6 more handlers (threshold is >5)
            dummy_handler = logging.StreamHandler()
            root_logger.addHandler(dummy_handler)
        
        # Mock file logging configured
        import buttermilk._core.log as log_module
        log_module._file_logging_configured = True
        
        validation = validate_logging_state()
        
        assert validation["valid"] is False
        assert any("Unusually high number of handlers" in issue for issue in validation["issues"])
        assert any("duplicate handler registration" in issue for issue in validation["issues"])

    def test_ensure_logging_properly_initialized_valid_state(self):
        """Test ensure_logging_properly_initialized with valid state."""
        # Set up logging properly
        setup_console_logging(verbose=False)
        
        # Mock file logging configured
        import buttermilk._core.log as log_module
        log_module._file_logging_configured = True
        
        # Should not raise any exception
        ensure_logging_properly_initialized()

    def test_ensure_logging_properly_initialized_invalid_state_fails_fast(self):
        """Test ensure_logging_properly_initialized fails fast with invalid state."""
        # Leave logging unconfigured
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            ensure_logging_properly_initialized()
            
        error_msg = str(exc_info.value)
        assert "Logging system is not properly initialized" in error_msg
        assert "Issues found:" in error_msg
        assert "logging initialization sequence" in error_msg
        assert "break verbose logging functionality" in error_msg


class TestIntegrationScenarios:
    """Test integration scenarios that combine multiple protection mechanisms."""

    def setup_method(self):
        """Reset all global state before each test."""
        # Reset logging state
        import buttermilk._core.log as log_module
        log_module._console_logging_configured = False
        log_module._file_logging_configured = False
        log_module._cloud_logging_sessions.clear()
        
        # Reset execution context state
        import buttermilk._core.execution_context as ec_module
        ec_module._global_execution_context = None
        ec_module._execution_context_initialized = False
        
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_multiple_session_creation_protection(self):
        """Test that creating multiple sessions doesn't break logging."""
        # This test simulates the scenario that was causing issues:
        # Multiple BM sessions trying to set up logging
        
        # First session setup (should succeed)
        setup_console_logging(verbose=True)
        execution_context_id_1 = f"session1_{uuid.uuid4().hex[:8]}"
        log_files_1 = setup_file_logging(execution_context_id=execution_context_id_1, verbose=True)
        
        # Verify first session setup worked
        assert len(log_files_1) == 1
        validation_1 = validate_logging_state(verbose_expected=True)
        assert validation_1["valid"] is True
        
        # Second session setup (should fail fast on logging)
        with pytest.raises(RuntimeError) as exc_info:
            setup_console_logging(verbose=True)
        assert "Console logging has already been configured" in str(exc_info.value)
        
        with pytest.raises(RuntimeError) as exc_info:
            setup_file_logging(execution_context_id="session2", verbose=True)
        assert "File logging has already been configured" in str(exc_info.value)
        
        # But validation should still pass for the original setup
        validation_2 = validate_logging_state(verbose_expected=True)
        assert validation_2["valid"] is True

    @patch("buttermilk._core.execution_context.setup_console_logging")
    @patch("buttermilk._core.execution_context.setup_file_logging")
    def test_execution_context_prevents_logging_corruption(self, mock_setup_file, mock_setup_console):
        """Test that ExecutionContext protection prevents logging corruption."""
        mock_setup_console.return_value = None
        mock_setup_file.return_value = ["/tmp/test.log"]
        
        # First ExecutionContext creation should succeed
        context1 = create_execution_context()
        
        # Verify logging was set up once
        assert mock_setup_console.call_count == 1
        assert mock_setup_file.call_count == 1
        
        # Second ExecutionContext creation should fail fast
        with pytest.raises(RuntimeError) as exc_info:
            create_execution_context()
            
        # Verify logging setup wasn't called again (preventing corruption)
        assert mock_setup_console.call_count == 1
        assert mock_setup_file.call_count == 1
        
        error_msg = str(exc_info.value)
        assert "break logging configuration" in error_msg

    def test_verbose_logging_protection_end_to_end(self):
        """Test end-to-end verbose logging protection."""
        # Set up verbose logging
        setup_console_logging(verbose=True)
        execution_context_id = f"verbose_test_{uuid.uuid4().hex[:8]}"
        log_files = setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        # Verify verbose logging is properly configured
        validation = validate_logging_state(verbose_expected=True)
        assert validation["valid"] is True
        assert validation["console_configured"] is True
        assert validation["file_configured"] is True
        
        # Verify DEBUG level is set
        buttermilk_logger = logging.getLogger("buttermilk")
        root_logger = logging.getLogger()
        assert buttermilk_logger.getEffectiveLevel() == logging.DEBUG
        assert root_logger.getEffectiveLevel() == logging.DEBUG
        
        # Verify any attempt to reconfigure fails fast
        with pytest.raises(RuntimeError):
            setup_console_logging(verbose=False)  # Would break verbose mode
            
        with pytest.raises(RuntimeError):
            setup_file_logging(execution_context_id="different", verbose=False)  # Would break verbose mode
        
        # Verify verbose logging is still intact after failed attempts
        final_validation = validate_logging_state(verbose_expected=True)
        assert final_validation["valid"] is True
        assert buttermilk_logger.getEffectiveLevel() == logging.DEBUG

    def test_error_messages_are_helpful_and_actionable(self):
        """Test that error messages provide clear guidance."""
        # Test console logging error message
        setup_console_logging(verbose=False)
        
        with pytest.raises(RuntimeError) as exc_info:
            setup_console_logging(verbose=True)
            
        console_error = str(exc_info.value)
        assert "Multiple calls to setup_console_logging()" in console_error
        assert "break verbose logging functionality" in console_error
        assert "problematic initialization sequence" in console_error
        
        # Test file logging error message
        execution_context_id = f"error_msg_test_{uuid.uuid4().hex[:8]}"
        setup_file_logging(execution_context_id=execution_context_id, verbose=False)
        
        with pytest.raises(RuntimeError) as exc_info:
            setup_file_logging(execution_context_id="different", verbose=True)
            
        file_error = str(exc_info.value)
        assert "Multiple calls to setup_file_logging()" in file_error
        assert "break verbose logging functionality" in file_error
        assert "conflicting log file handlers" in file_error
        assert "problematic initialization sequence" in file_error
        
        # Test ExecutionContext error message
        import buttermilk._core.execution_context as ec_module
        ec_module._execution_context_initialized = True
        
        with pytest.raises(RuntimeError) as exc_info:
            create_execution_context()
            
        context_error = str(exc_info.value)
        assert "ExecutionContext has already been initialized" in context_error
        assert "break logging configuration" in context_error
        assert "reset verbose logging settings" in context_error
        assert "get_or_create_execution_context()" in context_error
        
        # Test validation error message
        ec_module._execution_context_initialized = False  # Reset for validation test
        
        with pytest.raises(RuntimeError) as exc_info:
            ensure_logging_properly_initialized()
            
        validation_error = str(exc_info.value)
        assert "Logging system is not properly initialized" in validation_error
        assert "Issues found:" in validation_error
        assert "logging initialization sequence" in validation_error
        assert "break verbose logging functionality" in validation_error


class TestFailFastProtectionExamples:
    """Test examples that demonstrate the fail-fast protection working correctly.
    
    These tests serve as documentation for how the protection system should work
    and provide examples for developers to understand the correct usage patterns.
    """

    def setup_method(self):
        """Reset all global state before each test."""
        # Reset logging state
        import buttermilk._core.log as log_module
        log_module._console_logging_configured = False
        log_module._file_logging_configured = False
        log_module._cloud_logging_sessions.clear()
        
        # Reset execution context state
        import buttermilk._core.execution_context as ec_module
        ec_module._global_execution_context = None
        ec_module._execution_context_initialized = False
        
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_correct_usage_pattern_example(self):
        """Example of correct usage pattern that should work."""
        # Step 1: Set up console logging once
        setup_console_logging(verbose=True)
        
        # Step 2: Set up file logging once
        execution_context_id = f"correct_usage_{uuid.uuid4().hex[:8]}"
        log_files = setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        # Step 3: Validate the setup is correct
        validation = validate_logging_state(verbose_expected=True)
        assert validation["valid"] is True
        
        # Step 4: Use logging normally
        logger.debug("This is a debug message")
        logger.info("This is an info message")
        
        # Step 5: Validation continues to pass
        ensure_logging_properly_initialized()

    @patch("buttermilk._core.execution_context.setup_console_logging")
    @patch("buttermilk._core.execution_context.setup_file_logging")
    def test_safe_execution_context_pattern_example(self, mock_setup_file, mock_setup_console):
        """Example of safe ExecutionContext usage pattern."""
        mock_setup_console.return_value = None
        mock_setup_file.return_value = ["/tmp/test.log"]
        
        # Safe pattern: Use get_or_create_execution_context
        context1 = get_or_create_execution_context()
        
        # This is safe to call multiple times
        context2 = get_or_create_execution_context()
        context3 = get_or_create_execution_context()
        
        # All should return the same context
        assert context1 is context2 is context3
        
        # Logging setup should only happen once
        assert mock_setup_console.call_count == 1
        assert mock_setup_file.call_count == 1

    def test_problematic_pattern_example_that_fails_fast(self):
        """Example of problematic patterns that the protection prevents."""
        # Set up initial logging
        setup_console_logging(verbose=True)
        execution_context_id = f"problematic_{uuid.uuid4().hex[:8]}"
        setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        # These problematic patterns should all fail fast:
        
        # 1. Trying to reconfigure console logging
        with pytest.raises(RuntimeError, match="Console logging has already been configured"):
            setup_console_logging(verbose=False)
        
        # 2. Trying to reconfigure file logging
        with pytest.raises(RuntimeError, match="File logging has already been configured"):
            setup_file_logging(execution_context_id="different", verbose=False)
        
        # 3. Trying to create multiple ExecutionContexts
        import buttermilk._core.execution_context as ec_module
        ec_module._execution_context_initialized = True
        
        with pytest.raises(RuntimeError, match="ExecutionContext has already been initialized"):
            create_execution_context()
        
        # 4. Validation should still pass for the original setup
        ec_module._execution_context_initialized = False  # Reset for validation
        validation = validate_logging_state(verbose_expected=True)
        assert validation["valid"] is True

    def test_debugging_broken_logging_example(self):
        """Example of how to debug broken logging using validation functions."""
        # Simulate broken logging state (no setup)
        validation = validate_logging_state()
        
        # Validation should reveal the issues
        assert validation["valid"] is False
        assert "Console logging not configured" in validation["issues"]
        assert "File logging not configured" in validation["issues"]
        
        # ensure_logging_properly_initialized should fail fast with helpful message
        with pytest.raises(RuntimeError) as exc_info:
            ensure_logging_properly_initialized()
        
        error_msg = str(exc_info.value)
        assert "Logging system is not properly initialized" in error_msg
        assert "Console logging not configured" in error_msg
        assert "File logging not configured" in error_msg
        
        # After fixing the logging setup
        setup_console_logging(verbose=True)
        execution_context_id = f"debug_fixed_{uuid.uuid4().hex[:8]}"
        setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        # Validation should now pass
        validation_fixed = validate_logging_state(verbose_expected=True)
        assert validation_fixed["valid"] is True
        
        # ensure_logging_properly_initialized should now succeed
        ensure_logging_properly_initialized()  # Should not raise
