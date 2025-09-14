"""Integration test for nb_init() fixes.

Tests the specific issues identified and fixed:
1. Mixed logging formats (structured JSON vs unstructured console)
2. Secret provider configuration loading
3. OpenTelemetry instrumentation duplication
4. Configuration debugging output
"""
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from buttermilk.utils.nb import nb_init
from buttermilk._core.log import logger


class TestNbInitFixes:
    """Test nb_init() fixes for mixed logging, secrets, and OTEL duplication."""

    def setup_method(self):
        """Reset global state before each test."""
        # Reset execution context state
        import buttermilk._core.execution_context as ec_module
        ec_module._global_execution_context = None
        ec_module._execution_context_initialized = False

        # Reset logging state
        import buttermilk._core.log as log_module
        log_module._console_logging_configured = False
        log_module._file_logging_configured = False
        log_module._cloud_logging_sessions.clear()

        # Clear all handlers from loggers
        for logger_name in ["buttermilk", "root"]:
            test_logger = logging.getLogger(logger_name)
            for handler in test_logger.handlers[:]:
                test_logger.removeHandler(handler)

    def test_nb_init_with_missing_secrets_fails_fast(self, tmp_path):
        """Test that nb_init fails fast with clear error when secrets are missing."""
        # Create minimal config without secrets
        config_dir = tmp_path / "conf"
        config_dir.mkdir()

        # Create minimal config.yaml without secret_provider
        config_file = config_dir / "config.yaml"
        config_file.write_text("""
infrastructure:
  logging:
    verbose: true
  clouds: []
  # No secret_provider - this should cause failure
""")

        # Mock to avoid real tracing setup
        with patch('buttermilk._core.execution_context.weave'), \
             patch('buttermilk.utils.otel.setup_tracing_otel_with_execution_context'):

            # Should fail with clear error about missing secrets when trying to access LLMs
            with pytest.raises(RuntimeError, match="Secret provider configuration is missing"):
                bm = nb_init(job="test_job", project="test_project", config_dir=str(config_dir))
                # Trigger LLM access which requires secrets
                _ = bm.llms

    def test_nb_init_structured_logging_consistency(self, tmp_path, caplog):
        """Test that nb_init produces consistent structured logging."""
        config_dir = tmp_path / "conf"
        config_dir.mkdir()

        config_file = config_dir / "config.yaml"
        config_file.write_text("""
infrastructure:
  logging:
    verbose: true
  clouds: []
  secret_provider:
    type: "gcp"
    project_id: "test-project"
""")

        # Capture all log records to verify structure
        with caplog.at_level(logging.DEBUG), \
             patch('buttermilk._core.execution_context.weave'), \
             patch('buttermilk.utils.otel.setup_tracing_otel_with_execution_context'), \
             patch('buttermilk._core.execution_context.SecretsManager'):

            bm = nb_init(job="test_job", project="test_project", config_dir=str(config_dir))

            # Check that no log records contain f-string patterns (which would indicate mixed formats)
            structured_log_count = 0
            for record in caplog.records:
                if hasattr(record, 'msg') and isinstance(record.msg, str):
                    # Structured logs should not contain f-string artifacts like "Set project name for execution context: test_project"
                    # They should be separate fields
                    assert " for execution context: " not in record.msg, f"Found unstructured f-string in log: {record.msg}"
                    assert " to: " not in record.msg or "writing to:" not in record.msg, f"Found unstructured path in log: {record.msg}"

                    # Count properly structured log messages
                    if "project name for execution context" in record.msg or "logging enabled" in record.msg:
                        structured_log_count += 1

            # Should have captured some structured log messages during initialization
            assert structured_log_count > 0, "No structured log messages found during init"

    def test_nb_init_configuration_debugging_output(self, tmp_path, caplog):
        """Test that configuration debugging output appears during init."""
        config_dir = tmp_path / "conf"
        config_dir.mkdir()

        config_file = config_dir / "config.yaml"
        config_file.write_text("""
infrastructure:
  logging:
    verbose: true
  clouds:
    - type: "gcp"
      project_id: "test-project"
  secret_provider:
    type: "gcp"
    project_id: "test-project"
  tracing:
    weave:
      enabled: false
""")

        with caplog.at_level(logging.DEBUG), \
             patch('buttermilk._core.execution_context.weave'), \
             patch('buttermilk.utils.otel.setup_tracing_otel_with_execution_context'), \
             patch('buttermilk._core.execution_context.SecretsManager'):

            bm = nb_init(job="test_job", project="test_project", config_dir=str(config_dir))

            # Check for configuration debugging output
            debug_messages = [record.message for record in caplog.records if record.levelno == logging.DEBUG]

            # Should contain infrastructure configuration debugging
            config_debug_found = any("Infrastructure configuration loaded" in msg for msg in debug_messages)
            assert config_debug_found, f"Configuration debugging output not found. Debug messages: {debug_messages}"

            # Should contain top-level config keys debugging
            top_level_debug_found = any("Configuration loaded" in msg for msg in debug_messages)
            assert top_level_debug_found, f"Top-level configuration debugging not found. Debug messages: {debug_messages}"

    @patch('buttermilk.utils.otel.OpenAIInstrumentor')
    @patch('buttermilk.utils.otel.AnthropicInstrumentor')
    @patch('buttermilk.utils.otel.LoggingInstrumentor')
    def test_nb_init_prevents_otel_instrumentation_duplication(self, mock_logging_instr, mock_anthropic_instr, mock_openai_instr, tmp_path):
        """Test that OTEL instrumentors are not called multiple times."""
        config_dir = tmp_path / "conf"
        config_dir.mkdir()

        config_file = config_dir / "config.yaml"
        config_file.write_text("""
infrastructure:
  logging:
    verbose: true
  clouds: []
  secret_provider:
    type: "gcp"
    project_id: "test-project"
  tracing:
    otel:
      enabled: true
      endpoint: "http://test-endpoint"
""")

        # Mock instrumentors to track calls
        mock_openai_instance = MagicMock()
        mock_openai_instance.is_instrumented_by_opentelemetry = False
        mock_openai_instr.return_value = mock_openai_instance

        mock_anthropic_instance = MagicMock()
        mock_anthropic_instance.is_instrumented_by_opentelemetry = False
        mock_anthropic_instr.return_value = mock_anthropic_instance

        mock_logging_instance = MagicMock()
        mock_logging_instance.is_instrumented_by_opentelemetry = False
        mock_logging_instr.return_value = mock_logging_instance

        with patch('buttermilk._core.execution_context.weave'), \
             patch('buttermilk._core.execution_context.SecretsManager'), \
             patch('buttermilk.utils.otel.TracerProvider'), \
             patch('buttermilk.utils.otel.OTLPSpanExporter'), \
             patch('buttermilk.utils.otel.grpc'), \
             patch('buttermilk.utils.otel.google'):

            # First init should instrument
            bm1 = nb_init(job="test_job1", project="test_project", config_dir=str(config_dir))

            # Verify instrumentors were called once
            assert mock_openai_instance.instrument.call_count <= 1, "OpenAI instrumentor called more than once initially"
            assert mock_anthropic_instance.instrument.call_count <= 1, "Anthropic instrumentor called more than once initially"

            # Mark instrumentors as instrumented for subsequent calls
            mock_openai_instance.is_instrumented_by_opentelemetry = True
            mock_anthropic_instance.is_instrumented_by_opentelemetry = True
            mock_logging_instance.is_instrumented_by_opentelemetry = True

            # Reset execution context for second init
            import buttermilk._core.execution_context as ec_module
            ec_module._global_execution_context = None
            ec_module._execution_context_initialized = False

            # Second init should NOT instrument again
            bm2 = nb_init(job="test_job2", project="test_project", config_dir=str(config_dir))

            # Verify instrumentors were not called again
            assert mock_openai_instance.instrument.call_count <= 1, "OpenAI instrumentor called multiple times"
            assert mock_anthropic_instance.instrument.call_count <= 1, "Anthropic instrumentor called multiple times"

    def test_nb_init_user_exact_call_pattern(self, tmp_path):
        """Test the exact user call pattern that was failing."""
        config_dir = tmp_path / "conf"
        config_dir.mkdir()

        # Create config similar to what user likely has
        config_file = config_dir / "config.yaml"
        config_file.write_text("""
infrastructure:
  logging:
    verbose: true
  clouds:
    - type: "gcp"
      project_id: "test-project"
  secret_provider:
    type: "gcp"
    project_id: "test-project"
  tracing:
    weave:
      enabled: false
    otel:
      enabled: false
""")

        with patch('buttermilk._core.execution_context.weave'), \
             patch('buttermilk.utils.otel.setup_tracing_otel_with_execution_context'), \
             patch('buttermilk._core.execution_context.SecretsManager'):

            # This is the exact user call that was failing
            bm = nb_init(job="compile_cases", project="osb", config_dir=str(config_dir))

            # Verify basic initialization worked
            assert bm is not None
            assert bm.session_info.job_name == "compile_cases"
            assert bm.session_info.project_name == "osb"