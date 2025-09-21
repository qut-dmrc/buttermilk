"""Example tests for logging fail-fast protection that serve as documentation.

These tests demonstrate how to use the fail-fast logging protection system
and ensure the examples in our documentation remain accurate.

The fail-fast protection system prevents:
1. Multiple calls to setup_console_logging() that would break verbose logging
2. Multiple calls to setup_file_logging() that would create conflicting handlers
3. Multiple ExecutionContext creation that would reset logging configuration
4. Cloud logging duplication that would create duplicate handlers

Usage examples cover:
- Correct initialization patterns
- Safe multiple session creation
- Proper error handling
- Validation and debugging techniques
"""

import logging
import uuid
from unittest.mock import patch

import pytest

from buttermilk._core.execution_context import (
    create_execution_context,
    get_or_create_execution_context,
)
from buttermilk._core.log import (
    ensure_logging_properly_initialized,
    logger,
    setup_console_logging,
    setup_file_logging,
    validate_logging_state,
)


class TestBasicUsageExamples:
    """Basic usage examples for the fail-fast logging protection system."""

    def setup_method(self):
        """Reset global state before each test example."""
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

    def test_example_basic_logging_setup(self):
        """
        Basic usage of logging setup with fail-fast protection.
        
        This example demonstrates:
        - How to initialize console and file logging once
        - How the protection prevents multiple initialization
        - Basic validation of logging state
        """
        # Step 1: Set up console logging (first call succeeds)
        setup_console_logging(verbose=True)
        
        # Step 2: Set up file logging (first call succeeds)
        execution_context_id = f"basic_example_{uuid.uuid4().hex[:8]}"
        log_files = setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        # Step 3: Verify logging is working
        logger.debug("This debug message will be logged")
        logger.info("This info message will be logged")
        
        # Step 4: Validate the logging state
        validation = validate_logging_state(verbose_expected=True)
        assert validation["valid"] is True
        assert validation["console_configured"] is True
        assert validation["file_configured"] is True
        
        # Step 5: Demonstrate protection (these will fail fast)
        with pytest.raises(RuntimeError, match="Console logging has already been configured"):
            setup_console_logging(verbose=False)  # Would break verbose mode
            
        with pytest.raises(RuntimeError, match="File logging has already been configured"):
            setup_file_logging(execution_context_id="different", verbose=False)  # Would conflict
        
        # Step 6: Verify logging still works after failed attempts
        logger.debug("Logging still works perfectly")
        final_validation = validate_logging_state(verbose_expected=True)
        assert final_validation["valid"] is True

    def test_example_verbose_vs_non_verbose_logging(self):
        """
        Example showing the difference between verbose and non-verbose logging.
        
        This example covers:
        - Setting up verbose logging (DEBUG level)
        - Setting up non-verbose logging (INFO level)
        - How protection preserves the original setting
        """
        # Example A: Verbose logging setup
        setup_console_logging(verbose=True)
        execution_context_id = f"verbose_example_{uuid.uuid4().hex[:8]}"
        log_files = setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        # Verify DEBUG level is configured
        buttermilk_logger = logging.getLogger("buttermilk")
        root_logger = logging.getLogger()
        assert buttermilk_logger.getEffectiveLevel() == logging.DEBUG
        assert root_logger.getEffectiveLevel() == logging.DEBUG
        
        # Validate verbose configuration
        validation = validate_logging_state(verbose_expected=True)
        assert validation["valid"] is True
        
        # Test that DEBUG messages are captured
        logger.debug("This DEBUG message will be logged in verbose mode")
        logger.info("This INFO message will also be logged")
        
        # Verify protection prevents changing to non-verbose
        with pytest.raises(RuntimeError):
            setup_console_logging(verbose=False)  # Would break verbose functionality

    def test_example_non_verbose_logging_setup(self):
        """
        Example of non-verbose logging setup.
        
        This demonstrates:
        - Setting up INFO-level logging
        - How protection preserves non-verbose settings
        - Validation for non-verbose mode
        """
        # Set up non-verbose logging
        setup_console_logging(verbose=False)
        execution_context_id = f"non_verbose_example_{uuid.uuid4().hex[:8]}"
        log_files = setup_file_logging(execution_context_id=execution_context_id, verbose=False)
        
        # Verify INFO level is configured
        buttermilk_logger = logging.getLogger("buttermilk")
        assert buttermilk_logger.getEffectiveLevel() == logging.INFO
        
        # Validate non-verbose configuration
        validation = validate_logging_state(verbose_expected=False)
        assert validation["valid"] is True
        
        # Test logging at different levels
        logger.debug("This DEBUG message may not be visible in non-verbose mode")
        logger.info("This INFO message will be logged")
        logger.warning("This WARNING message will be logged")
        
        # Verify protection prevents changing to verbose
        with pytest.raises(RuntimeError):
            setup_console_logging(verbose=True)  # Would conflict with existing setup


class TestExecutionContextExamples:
    """Examples for ExecutionContext fail-fast protection."""

    def setup_method(self):
        """Reset global state before each test example."""
        # Reset execution context state
        import buttermilk._core.execution_context as ec_module
        ec_module._global_execution_context = None
        ec_module._execution_context_initialized = False
        ec_module._global_execution_context_id = ""
        
        # Reset logging state
        import buttermilk._core.log as log_module
        log_module._console_logging_configured = False
        log_module._file_logging_configured = False
        
        # Clear handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    @patch("buttermilk._core.execution_context.setup_console_logging")
    @patch("buttermilk._core.execution_context.setup_file_logging")
    def test_example_safe_execution_context_pattern(self, mock_setup_file, mock_setup_console):
        """
        Example of the safe ExecutionContext usage pattern.
        
        This example demonstrates:
        - Using get_or_create_execution_context() safely
        - How multiple calls return the same context
        - Why this pattern prevents logging corruption
        """
        mock_setup_console.return_value = None
        mock_setup_file.return_value = ["/tmp/test.log"]
        
        # Safe pattern: Always use get_or_create_execution_context()
        context1 = get_or_create_execution_context()
        
        # Multiple calls are safe and return the same context
        context2 = get_or_create_execution_context()
        context3 = get_or_create_execution_context()
        
        # All contexts are the same object
        assert context1 is context2 is context3
        
        # Logging setup only happened once (prevents corruption)
        assert mock_setup_console.call_count == 1
        assert mock_setup_file.call_count == 1
        
        # This pattern is safe to use anywhere in your application
        for _ in range(10):
            safe_context = get_or_create_execution_context()
            assert safe_context is context1

    @patch("buttermilk._core.execution_context.setup_console_logging")
    @patch("buttermilk._core.execution_context.setup_file_logging")
    def test_example_dangerous_execution_context_pattern(self, mock_setup_file, mock_setup_console):
        """
        Example of the dangerous ExecutionContext pattern that fails fast.
        
        This example shows:
        - Why create_execution_context() should only be called once
        - How the protection prevents multiple creation
        - The error message that guides developers to the safe pattern
        """
        mock_setup_console.return_value = None
        mock_setup_file.return_value = ["/tmp/test.log"]
        
        # First call succeeds
        context1 = create_execution_context()
        assert context1 is not None
        
        # Second call fails fast with helpful error message
        with pytest.raises(RuntimeError) as exc_info:
            create_execution_context()
        
        error_msg = str(exc_info.value)
        assert "ExecutionContext has already been initialized" in error_msg
        assert "break logging configuration" in error_msg
        assert "get_or_create_execution_context()" in error_msg
        
        # The error message guides developers to the safe alternative
        safe_context = get_or_create_execution_context()
        assert safe_context is context1  # Same context, safely accessed


class TestValidationAndDebuggingExamples:
    """Examples for validation and debugging with the fail-fast system."""

    def setup_method(self):
        """Reset global state before each test example."""
        # Reset logging state
        import buttermilk._core.log as log_module
        log_module._console_logging_configured = False
        log_module._file_logging_configured = False
        
        # Clear handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_example_debugging_broken_logging(self):
        """
        Example of how to debug broken logging using validation functions.
        
        This example demonstrates:
        - How to detect logging configuration issues
        - Using validate_logging_state() for diagnosis
        - Using ensure_logging_properly_initialized() for fail-fast checks
        - How to fix detected issues
        """
        # Step 1: Check current logging state (initially broken)
        validation = validate_logging_state()
        assert validation["valid"] is False
        
        # Step 2: Examine the specific issues
        issues = validation["issues"]
        print(f"Logging issues detected: {issues}")  # For demonstration
        
        # Common issues you'll see:
        assert "Console logging not configured" in issues
        assert "File logging not configured" in issues
        
        # Step 3: Use ensure_logging_properly_initialized() for fail-fast detection
        with pytest.raises(RuntimeError) as exc_info:
            ensure_logging_properly_initialized()
        
        error_msg = str(exc_info.value)
        assert "Logging system is not properly initialized" in error_msg
        assert "Console logging not configured" in error_msg
        
        # Step 4: Fix the detected issues
        setup_console_logging(verbose=True)
        execution_context_id = f"debugging_example_{uuid.uuid4().hex[:8]}"
        setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        # Step 5: Verify the fixes worked
        fixed_validation = validate_logging_state(verbose_expected=True)
        assert fixed_validation["valid"] is True
        assert len(fixed_validation["issues"]) == 0
        
        # Step 6: ensure_logging_properly_initialized() now succeeds
        ensure_logging_properly_initialized()  # Should not raise

    def test_example_verbose_level_mismatch_debugging(self):
        """
        Example of debugging verbose level mismatches.
        
        This example shows:
        - How to detect when verbose settings don't match expectations
        - Understanding validation error messages
        - How the protection prevents accidental level changes
        """
        # Step 1: Set up logging with verbose=False
        setup_console_logging(verbose=False)
        execution_context_id = f"mismatch_example_{uuid.uuid4().hex[:8]}"
        setup_file_logging(execution_context_id=execution_context_id, verbose=False)
        
        # Step 2: Validate expecting verbose=True (will detect mismatch)
        validation = validate_logging_state(verbose_expected=True)
        assert validation["valid"] is False
        
        # Step 3: Examine the mismatch issues
        issues_text = " ".join(validation["issues"])
        assert "Verbose mode expected but root logger level is" in issues_text
        assert "Verbose mode expected but buttermilk logger level is" in issues_text
        
        # Step 4: The protection prevents "fixing" this by reconfiguring
        with pytest.raises(RuntimeError, match="Console logging has already been configured"):
            setup_console_logging(verbose=True)  # Would like to fix, but protection prevents it
        
        # Step 5: Validate with correct expectation (verbose=False)
        correct_validation = validate_logging_state(verbose_expected=False)
        assert correct_validation["valid"] is True
        
        # This demonstrates that the initial setup was correct for non-verbose mode
        # The "issue" was in our expectation, not the actual configuration

    def test_example_monitoring_logging_health(self):
        """
        Example of ongoing logging health monitoring.
        
        This example demonstrates:
        - Regular health checks using validation
        - Detecting handler proliferation issues
        - Monitoring for configuration drift
        """
        # Step 1: Set up proper logging
        setup_console_logging(verbose=True)
        execution_context_id = f"monitoring_example_{uuid.uuid4().hex[:8]}"
        setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        # Step 2: Initial health check
        health_check_1 = validate_logging_state(verbose_expected=True)
        assert health_check_1["valid"] is True
        initial_handler_count = health_check_1["handler_count"]
        
        # Step 3: Simulate some application operations
        logger.debug("Operation 1")
        logger.info("Operation 2")
        logger.warning("Operation 3")
        
        # Step 4: Regular health check during operations
        health_check_2 = validate_logging_state(verbose_expected=True)
        assert health_check_2["valid"] is True
        assert health_check_2["handler_count"] == initial_handler_count  # No handler proliferation
        
        # Step 5: Simulate attempt to add duplicate handlers (protection prevents this)
        with pytest.raises(RuntimeError):
            setup_console_logging(verbose=True)  # Protection prevents duplicate
        
        # Step 6: Verify health after protection triggered
        health_check_3 = validate_logging_state(verbose_expected=True)
        assert health_check_3["valid"] is True
        assert health_check_3["handler_count"] == initial_handler_count  # Still same count
        
        # Step 7: Use ensure_logging_properly_initialized() for critical operations
        ensure_logging_properly_initialized()  # Fail-fast if logging is broken


class TestRealWorldScenarioExamples:
    """Real-world scenario examples for the fail-fast protection system."""

    def setup_method(self):
        """Reset global state before each test example."""
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

    def test_example_application_startup_pattern(self):
        """
        Example of recommended application startup pattern.
        
        This example demonstrates:
        - Proper initialization order
        - Single-point logging setup
        - Validation and error handling
        - Protection against reinitialization
        """
        # Application startup sequence:
        
        # Step 1: Initialize logging early in application startup
        def initialize_logging():
            """Initialize application logging once during startup."""
            setup_console_logging(verbose=True)
            execution_context_id = f"app_startup_{uuid.uuid4().hex[:8]}"
            log_files = setup_file_logging(execution_context_id=execution_context_id, verbose=True)
            
            # Validate the setup
            validation = validate_logging_state(verbose_expected=True)
            if not validation["valid"]:
                raise RuntimeError(f"Logging initialization failed: {validation['issues']}")
            
            logger.info("Application logging initialized successfully")
            return log_files
        
        # Step 2: Call initialization once
        log_files = initialize_logging()
        assert len(log_files) == 1
        
        # Step 3: Subsequent calls to initialize_logging() will fail fast
        with pytest.raises(RuntimeError, match="Console logging has already been configured"):
            initialize_logging()
        
        # Step 4: But logging continues to work normally
        logger.debug("Application startup complete")
        logger.info("Ready to process requests")
        
        # Step 5: Ongoing health checks
        ensure_logging_properly_initialized()  # Critical operations can verify logging health

    def test_example_multi_session_workflow(self):
        """
        Example of multi-session workflow with logging protection.
        
        This example demonstrates:
        - Creating multiple sessions safely
        - Avoiding logging reconfiguration
        - Maintaining consistent logging across sessions
        """
        # Scenario: Multiple processing sessions in the same application
        
        # Step 1: Initialize logging once for the application
        setup_console_logging(verbose=True)
        execution_context_id = f"multi_session_{uuid.uuid4().hex[:8]}"
        setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        # Step 2: Create multiple processing sessions
        session_results = []
        
        for session_id in range(3):
            # Each session does its own work
            logger.info(f"Starting session {session_id}")
            
            # Session attempts to set up logging (protection prevents issues)
            try:
                setup_console_logging(verbose=True)  # Would normally cause issues
            except RuntimeError as e:
                # Protection worked - logging already configured
                assert "already been configured" in str(e)
                logger.debug(f"Session {session_id}: Logging already configured (protection working)")
            
            # Session continues with its work
            logger.debug(f"Session {session_id}: Processing data")
            logger.info(f"Session {session_id}: Completed successfully")
            session_results.append(f"session_{session_id}_complete")
        
        # Step 3: Verify all sessions completed and logging is still healthy
        assert len(session_results) == 3
        validation = validate_logging_state(verbose_expected=True)
        assert validation["valid"] is True
        
        logger.info(f"All sessions completed: {session_results}")

    def test_example_error_recovery_workflow(self):
        """
        Example of error recovery workflow using validation.
        
        This example demonstrates:
        - Detecting logging issues during application runtime
        - Using validation for health checks
        - Recovery strategies when logging is broken
        """
        # Scenario: Application needs to verify logging health before critical operations
        
        def critical_operation_with_logging_check():
            """Perform critical operation with logging health verification."""
            try:
                # Check logging health before proceeding
                ensure_logging_properly_initialized()
                
                # Proceed with critical operation
                logger.info("Starting critical operation")
                logger.debug("Critical operation details...")
                
                # Simulate operation
                result = "operation_successful"
                
                logger.info(f"Critical operation completed: {result}")
                return result
                
            except RuntimeError as e:
                # Logging is broken - cannot proceed safely
                print(f"Critical operation aborted due to logging issue: {e}")
                raise
        
        # Step 1: Initially, logging is not configured (broken state)
        with pytest.raises(RuntimeError, match="Logging system is not properly initialized"):
            critical_operation_with_logging_check()
        
        # Step 2: Fix the logging configuration
        setup_console_logging(verbose=True)
        execution_context_id = f"error_recovery_{uuid.uuid4().hex[:8]}"
        setup_file_logging(execution_context_id=execution_context_id, verbose=True)
        
        # Step 3: Now critical operation can proceed
        result = critical_operation_with_logging_check()
        assert result == "operation_successful"
        
        # Step 4: Verify ongoing health
        validation = validate_logging_state(verbose_expected=True)
        assert validation["valid"] is True
        
        # Step 5: Critical operation continues to work
        result_2 = critical_operation_with_logging_check()
        assert result_2 == "operation_successful"
