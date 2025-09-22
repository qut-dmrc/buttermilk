# Test Health Dashboard

## Summary Statistics

- **Total Test Files**: 140
- **Total Tests**: 1057
- **Passed**: 391 (37.0%)
- **Failed**: 568 (53.7%)
- **Errors**: 70
- **Skipped**: 28
- **Collection Errors**: 1

## Failures by Error Type

### Other (57 issues)
- `tests/agents/test_agentoutput.py`
  - Test: `test_actual_agent_trace_full_dump_with_default_outputs
FAILED tests/api/test_osb_criteria_mapping.py::test_osb_flow_with_direct_query
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_with_text
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_no_media_no_text
FAILED tests/agents/test_markdown.py::TestFramedStatementMarkdown::test_framed_statement_as_markdown
FAILED tests/data/test_records_openaimessages.py::test_from_path_valid`
  - Error: `Attr......`
- `tests/endtoend/test_imagegen.py`
  - Test: `test_model[SD35Large-stereotype]`
  - Error: `Ru......`
- `tests/api/test_score_endpoints.py`
  - Test: `TestScoreEndpointsIntegration::test_data_service_error_handling
FAILED tests/endtoend/test_imagegen.py::test_model[VertexImagegenModels-cat]
FAILED tests/api/test_score_endpoints.py::TestScoreAPIEndpoints::test_endpoint_registration
FAILED tests/endtoend/test_imagegen.py::test_model[VertexImagegenModels-stereotype]
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_should_persist_message
FAILED tests/endtoend/test_imagegen.py::test_model[SD3-cat]`
  - Error: `RuntimeError: B......`
  - ... and 54 more

### Assertion Error (3 issues)
- `tests/00initial/test_bm_singleton.py`
  - Test: `test_conf`
  - Error: `AssertionError: BM i......`
- `tests/tools/test_uploader.py`
  - Test: `TestAsyncDataUploader::test_uploader_initialization_with_real_storage
FAILED tests/tools/test_templating.py::test_template_synth`
  - Error: `AssertionError: ......`
- `tests/tools/test_uploader.py`
  - Test: `TestAsyncDataUploader::test_uploader_handles_storage_interface
FAILED tests/tools/test_uploader.py::TestAsyncDataUploaderErrorHandling::test_uploader_handles_invalid_buffer_size
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[criteria-criteria_hatefb_factorised-9000]
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[criteria-criteria_ordinary-5240]
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[synth-synthesise-1000]
FAILED tests/endtoend/test_text2image_apis.py::test_generated_image_is_valid_and_nontrivial[DALLE]
FAILED tests/tools/test_utils.py::test_b64_str_validator`
  - Error: `AssertionError: as......`
### Type Error (2 issues)
- `tests/00initial/test_bm_singleton.py`
  - Test: `test_import_singleton_from_different_modules
FAILED tests/api/test_session_persistence.py::TestSessionGCSArchival::test_finalize_session_always_attempts_archival
FAILED tests/endtoend/flows/test_rag_zotero_llm.py::test_rag_zotero_with_structured_output[gpt5mini]
FAILED tests/endtoend/flows/test_describe.py::test_run_flow_describe_only[gemini25pro]
FAILED tests/endtoend/flows/test_rag_zotero_llm.py::test_rag_zotero_with_structured_output[gpt5nano]
FAILED tests/endtoend/flows/test_rag_zotero_llm.py::test_rag_zotero_with_structured_output[llama4maverick]
FAILED tests/endtoend/test_imagegen.py::test_batch`
  - Error: `TypeError: BatchImageGen......`
- `tests/api/test_osb_websocket_lifecycle.py`
  - Test: `TestOSBWebSocketMessageFlow::test_osb_query_message_flow
FAILED tests/api/test_session_persistence.py::TestSessionGCSArchival::test_archive_to_gcs_success
FAILED tests/endtoend/flows/test_describe.py::test_run_flow_describe_only[llama32_90b]
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_record_by_id_found
FAILED tests/endtoend/flows/test_fastapi.py::test_run_flow`
  - Error: `TypeError: Async......`
### Import Error (1 issues)
- `tests/data/test_tox_examples.py _______________`
  - Error: `ImportError while importing test module '/home/nic/src/writing/projects/buttermilk/tests/data/test_t...`
### Value Error (1 issues)
- `tests/data/test_records_openaimessages.py`
  - Test: `test_as_openai_message_with_media
FAILED tests/endtoend/test_llm_tool_calling.py::test_structured_output_without_tools[gemini25flash]
FAILED tests/endtoend/test_llm_tool_calling.py::test_no_tool_needed[gemini25flash]
FAILED tests/endtoend/test_llms.py::TestPromptStyles::test_usertext_and_placeholder[lady macbeth-gemini25flash]
FAILED tests/integration/test_osb_websocket_integration.py::TestOSBWebSocketErrorHandling::test_osb_websocket_connection_recovery
FAILED tests/integration/test_osb_websocket_integration.py::TestOSBWebSocketErrorHandling::test_osb_agent_failure_handling
FAILED tests/endtoend/test_llm_tool_calling.py::test_single_tool_call[gemini25pro]
FAILED tests/endtoend/test_llms.py::TestPromptStyles::test_usertext_and_placeholder[osb blm-gemini25flash]
FAILED tests/integration/test_logging_fail_fast_integration.py::TestErrorRecoveryAndValidation::test_logging_validation_after_partial_setup
FAILED tests/integration/test_logging_fail_fast_integration.py::TestVerboseLoggingPreservation::test_verbose_logging_preserved_during_session_operations
FAILED tests/integration/20core/test_bm_session_isolation.py::TestBackwardCompatibility::test_flowrunner_without_session_bm
FAILED tests/integration/20core/test_bm_session_isolation.py::TestBackwardCompatibility::test_orchestrator_without_session_bm
FAILED tests/integration/20core/test_bm_session_isolation.py::TestBackwardCompatibility::test_agent_without_session_bm
FAILED tests/integration/test_logging_fail_fast_integration.py::TestVerboseLoggingPreservation::test_non_verbose_logging_preserved_during_operations
FAILED tests/integration/test_logging_fail_fast_integration.py::TestErrorRecoveryAndValidation::test_verbose_level_mismatch_detection
FAILED tests/integration/test_logging_fail_fast_integration.py::TestCloudLoggingIntegration::test_cloud_logging_deduplication_across_sessions
FAILED tests/api/test_osb_websocket_lifecycle.py::TestOSBWebSocketConnection::test_websocket_osb_message_routing
FAILED tests/integration/test_safetysettings.py::test_safetysettings[gemini-selfharm blockeds]
FAILED tests/integration/test_safetysettings.py::test_safetysettings[gemini-dangerous allowed]
FAILED tests/integration/test_safetysettings.py::test_safetysettings[gemini-selfharm allowed]
FAILED tests/integration/20core/test_cloud_logging_live.py::test_multiple_sessions_isolated_logging
FAILED tests/integration/20core/test_cloud_logging_live.py::test_structured_json_format_consistency
FAILED tests/integration/20core/test_cloud_logging_live.py::test_cloud_logging_error_handling
FAILED tests/integration/test_safetysettings.py::test_safetysettings[llama32-ok]
FAILED tests/integration/20core/test_cloud_logging_live.py::test_async_cloud_logging_performance
FAILED tests/integration/test_safetysettings.py::test_safetysettings[llama32-refused]
FAILED tests/integration/20core/test_config_reload.py::TestConfigurationReload::test_config_snapshot_saving
FAILED tests/integration/20core/test_config_reload.py::TestConfigurationReload::test_reload_configurations_success
FAILED tests/integration/test_safetysettings.py::test_safetysettings[llama32-also refused]
FAILED tests/integration/test_safetysettings.py::test_safetysettings[llama31-also refused]
FAILED tests/integration/test_safetysettings.py::test_safetysettings[llama31-dangerous blocked]
FAILED tests/api/test_osb_websocket_lifecycle.py::TestOSBWebSocketErrorHandling::test_websocket_large_message_handling
FAILED tests/integration/test_safetysettings.py::test_safetysettings[llama31-selfharm blockeds]
FAILED tests/integration/test_safetysettings.py::test_safetysettings[llama31-dangerous allowed]
FAILED tests/integration/test_logging_fail_fast_integration.py::TestErrorRecoveryAndValidation::test_integration_with_real_logging_operations
FAILED tests/integration/test_safetysettings.py::test_safetysettings[llama31-selfharm allowed]
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_cli_init_with_real_config_first_session
FAILED tests/api/test_osb_websocket_lifecycle.py::TestOSBWebSocketConnection::test_websocket_connection_state_management
FAILED tests/endtoend/test_llmagents.py::test_llm_agent_template_metadata[gemini25flash]
FAILED tests/integration/test_logging_fail_fast_integration.py::TestFailFastIntegrationExamples::test_recommended_application_startup_pattern
FAILED tests/endtoend/test_llmagents.py::test_llm_agent_template_metadata[gpt5nano]
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_cli_init_with_real_config_subsequent_session_inheritance
FAILED tests/endtoend/test_llmagents.py::test_llm_agent_template_metadata[haiku]
FAILED tests/api/test_osb_websocket_lifecycle.py::TestOSBWebSocketErrorHandling::test_websocket_rapid_message_sending
FAILED tests/endtoend/test_llm_tool_calling.py::test_structured_output_without_tools[gemini25pro]
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_cli_init_with_real_config_project_mismatch_error
FAILED tests/api/test_osb_websocket_lifecycle.py::TestOSBWebSocketConnection::test_websocket_message_validation
FAILED tests/api/test_osb_websocket_lifecycle.py::TestOSBWebSocketErrorHandling::test_websocket_session_not_found_handling
FAILED tests/integration/test_logging_fail_fast_integration.py::TestFailFastIntegrationExamples::test_verbose_logging_workflow_example
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_bootstrap_session_function_with_real_config
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_weave_missing_project_id_fails_fast
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_weave_missing_api_key_fails_fast
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_weave_trace_creation_and_submission
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_session_isolation_with_real_config
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_traceloop_missing_api_key_validation
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_bootstrap_session_with_config_function_with_real_config
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_otel_gcp_trace_setup
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_cli_init_with_overrides_real_config
FAILED tests/api/test_osb_websocket_lifecycle.py::TestOSBWebSocketSessionIsolation::test_concurrent_osb_sessions
FAILED tests/api/test_osb_websocket_lifecycle.py::TestOSBWebSocketPerformance::test_websocket_connection_performance
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_multiple_providers_configuration
FAILED tests/integration/test_unified_bootstrap_integration.py::TestMainScriptRealExecution::test_main_script_scenario_with_real_config
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_disabled_providers_not_initialized
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_execution_context_fix_for_nonetype_error
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_weave_config_based_initialization
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_config_based_credentials_override_environment
FAILED tests/runner/test_cli.py::test_main_console_mode`
  - Error: `ValueError: Invalid......`

## Failures by Test Category

- **root**: 24 failures
- **integration**: 7 failures
- **tools**: 7 failures
- **data**: 5 failures
- **initial**: 5 failures
- **agents**: 4 failures
- **api**: 3 failures
- **groupchat**: 3 failures
- **unit**: 3 failures
- **storage**: 2 failures
- **utils**: 1 failures

## Priority Fixes (Blocking Test Execution)

These issues prevent tests from even running:

1. `tests/data/test_tox_examples.py _______________`
   - Type: import_error
   - Error: `ImportError while importing test module '/home/nic/src/writing/projects/buttermilk/tests/data/test_tox_examples.py'.
Hint: make sure your test modules...`