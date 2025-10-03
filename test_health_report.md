# Test Health Dashboard

## Summary Statistics

- **Total Test Files**: 156
- **Total Tests**: 929
- **Passed**: 494 (53.2%)
- **Failed**: 342 (36.8%)
- **Errors**: 67
- **Skipped**: 26
- **Collection Errors**: 0

## Failures by Error Type

### Other (31 issues)
- `tests/api/test_score_endpoints.py`
  - Test: `TestDataService::test_get_record_by_id_found
FAILED tests/api/test_message_service_tokens.py::TestMessageServiceTokenExtraction::test_format_message_extracts_tokens_from_agent_trace
FAILED tests/agents/test_agentoutput.py::test_actual_judge_reasons_direct_dump
FAILED tests/groupchat/test_agenttrace_runinfo.py::test_agenttrace_serializes_runinfo_correctly
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_record_by_id_not_found
FAILED tests/api/test_score_endpoints.py::TestScoreAPIEndpoints::test_endpoint_registration
FAILED tests/groupchat/test_agenttrace_runinfo.py::test_agenttrace_runinfo_is_json_serializable
FAILED tests/agents/test_agentoutput.py::test_actual_agent_trace_full_dump_includes_nested_outputs
FAILED tests/data/test_chromadb_sync_fix.py::TestChromaDBSyncFix::test_path_preservation_logic
FAILED tests/agents/test_agentoutput.py::test_actual_agent_trace_full_dump_with_default_outputs
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announces_on_cleanup
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_records_for_flow_without_scores
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_host_listens_for_announcements
FAILED tests/groupchat/test_agenttrace_runinfo.py::test_agenttrace_handles_missing_bm_gracefully
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_records_for_flow_with_scores
FAILED tests/agents/test_markdown.py::TestFramedStatementMarkdown::test_framed_statement_as_markdown
FAILED tests/agents/test_fetch.py::TestFetch::test_load_data`
  - Error: `AttributeError......`
- `tests/examples/test_standalone_trace_examples.py`
  - Test: `TestStandaloneTraceExamples::test_agent_integration_example
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_save_message
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announcement_includes_tools_and_message_types
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_scores_for_record_no_data
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_host_announces_itself_and_receives_responses
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_get_session_messages
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_with_media
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_handles_announcement_errors_gracefully
FAILED tests/api/test_osb_criteria_mapping.py::test_osb_criteria_mapped_to_query
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_scores_for_record_with_data
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_with_text
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_session_exists
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announcement_in_invoke_lifecycle
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_calculation_in_wait_method
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_no_media_no_text
FAILED tests/api/test_osb_criteria_mapping.py::test_osb_flow_with_direct_query
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_batch_processing_pattern
FAILED tests/groupchat/test_agent_centric_tools.py::test_agent_tool_definition
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_responses_for_record
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_with_zero_tasks
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_should_persist_message
FAILED tests/groupchat/test_agent_centric_tools.py::test_agent_announcement
FAILED tests/api/test_score_endpoints.py::TestScoreEndpointsIntegration::test_data_service_error_handling
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_with_varying_task_counts
FAILED tests/groupchat/test_agent_centric_tools.py::test_host_tool_collection
FAILED tests/agents/test_markdown.py::TestExecutionTraceMarkdown::test_agent_trace_with_judge_output
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_corrupted_session_file_handling
FAILED tests/integration/test_osb_websocket_integration.py::TestOSBWebSocketErrorHandling::test_osb_agent_failure_handling
FAILED tests/integration/test_llm_agent_refactoring.py::TestLLMAgentRefactoring::test_llmagent_process_with_context
FAILED tests/groupchat/test_agent_centric_tools.py::test_tool_calling_flow`
  - Error: `......`
- `tests/integration/20core/test_config_reload.py`
  - Test: `TestConfigurationReload::test_config_snapshot_saving
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_respects_limits
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_inject_parent_trace_example
FAILED tests/integration/20core/test_config_reload.py::TestConfigurationReload::test_reload_configurations_success
FAILED tests/integration/20core/test_bm_session_isolation.py::TestBackwardCompatibility::test_orchestrator_without_session_bm
FAILED tests/integration/20core/test_config_reload.py::TestConfigurationReload::test_reload_configurations_failure
FAILED tests/groupchat/test_agent_centric_tools.py::test_registry_update_rebuilds_tools
FAILED tests/integration/test_llm_agent_refactoring.py::TestLLMAgentRefactoring::test_llmagent_process_with_records
FAILED tests/integration/20core/test_config_reload.py::TestConfigurationReload::test_config_snapshot_handles_missing_attributes
FAILED tests/api/test_session_persistence.py::TestSessionStorageHelperMethods::test_get_or_create_session_data_existing_session
FAILED tests/integration/20core/test_config_reload.py::TestConfigReloadIntegration::test_startup_script_functionality
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_error_handling_example
FAILED tests/integration/test_llm_agent_refactoring.py::TestLLMAgentRefactoring::test_llmagent_parameter_merging
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_traceloop_missing_api_key_validation
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_manual_context_management
FAILED tests/00initial/test_init.py::test_short_form_cli`
  - Error: `RuntimeError: Proj......`
  - ... and 28 more

### Value Error (1 issues)
- `tests/integration/test_unified_tracing_config.py`
  - Test: `TestUnifiedTracingConfig::test_execution_context_fix_for_nonetype_error
FAILED tests/integration/test_llm_agent_refactoring.py::TestLLMAgentRefactoring::test_llmagent_handles_unexpected_error
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceEdgeCases::test_nested_traces
FAILED tests/examples/test_logging_fail_fast_examples.py::TestBasicUsageExamples::test_example_non_verbose_logging_setup
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announces_on_initialize
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_otel_gcp_trace_setup
FAILED tests/integration/test_llm_agent_refactoring.py::TestLLMAgentRefactoring::test_llmagent_template_metadata_preserved
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_detects_available_tools
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_responds_to_host_announcement
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_does_not_respond_to_non_host_announcement
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_cli_init_with_overrides_real_config
FAILED tests/runner/test_cli.py::test_main_console_mode`
  - Error: `ValueError: Invalid......`
### Assertion Error (1 issues)
- `tests/tools/test_uploader.py`
  - Test: `TestAsyncDataUploader::test_uploader_initialization_with_real_storage
FAILED tests/examples/test_logging_fail_fast_examples.py::TestRealWorldScenarioExamples::test_example_error_recovery_workflow
FAILED tests/tools/test_uploader.py::TestAsyncDataUploader::test_uploader_handles_storage_interface
FAILED tests/tools/test_uploader.py::TestAsyncDataUploaderErrorHandling::test_uploader_handles_invalid_buffer_size
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[criteria-criteria_ordinary-5240]
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[criteria-criteria_hatefb_factorised-9000]
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[synth-synthesise-1000]
FAILED tests/tools/test_utils.py::test_b64_str_validator`
  - Error: `AssertionError: as......`

## Failures by Test Category

- **root**: 9 failures
- **tools**: 7 failures
- **integration**: 6 failures
- **data**: 5 failures
- **api**: 1 failures
- **examples**: 1 failures
- **storage**: 1 failures
- **groupchat**: 1 failures
- **unit**: 1 failures
- **validation**: 1 failures

## Priority Fixes (Blocking Test Execution)

No critical blocking issues found.
