# Test Health Dashboard

## Summary Statistics

- **Total Test Files**: 156
- **Total Tests**: 929
- **Passed**: 472 (50.8%)
- **Failed**: 366 (39.4%)
- **Errors**: 65
- **Skipped**: 26
- **Collection Errors**: 0

## Failures by Error Type

### Other (36 issues)
- `tests/api/test_score_endpoints.py`
  - Test: `TestDataService::test_get_record_by_id_found
FAILED tests/api/test_score_endpoints.py::TestScoreAPIEndpoints::test_endpoint_registration
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_host_listens_for_announcements
FAILED tests/agents/test_agentoutput.py::test_actual_judge_reasons_direct_dump
FAILED tests/api/test_message_service_tokens.py::TestMessageServiceTokenExtraction::test_format_message_extracts_tokens_from_agent_trace
FAILED tests/groupchat/test_agenttrace_runinfo.py::test_agenttrace_serializes_runinfo_correctly
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_record_by_id_not_found
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_save_message
FAILED tests/agents/test_agentoutput.py::test_actual_agent_trace_full_dump_includes_nested_outputs
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announces_on_cleanup
FAILED tests/00initial/test_bm_singleton.py::test_singleton_instance`
  - Error: `Assert......`
- `tests/groupchat/test_agenttrace_runinfo.py`
  - Test: `test_agenttrace_runinfo_is_json_serializable
FAILED tests/agents/test_agentoutput.py::test_actual_agent_trace_full_dump_with_default_outputs
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_host_announces_itself_and_receives_responses
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_records_for_flow_without_scores
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_get_session_messages
FAILED tests/data/test_chromadb_sync_fix.py::TestChromaDBSyncFix::test_path_preservation_logic
FAILED tests/00initial/test_bm_singleton.py::test_singleton_between_modules
FAILED tests/groupchat/test_agenttrace_runinfo.py::test_agenttrace_handles_missing_bm_gracefully
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announcement_includes_tools_and_message_types
FAILED tests/agents/test_markdown.py::TestFramedStatementMarkdown::test_framed_statement_as_markdown
FAILED tests/agents/test_fetch.py::TestFetch::test_load_data`
  - Error: `AttributeError......`
- `tests/api/test_score_endpoints.py`
  - Test: `TestDataService::test_get_records_for_flow_with_scores
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_agent_integration_example
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_session_exists
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_calculation_in_wait_method
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_with_media
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_scores_for_record_no_data
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_handles_announcement_errors_gracefully
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_with_text
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_should_persist_message
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_with_zero_tasks
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_no_media_no_text
FAILED tests/api/test_osb_criteria_mapping.py::test_osb_criteria_mapped_to_query
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announcement_in_invoke_lifecycle
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_scores_for_record_with_data
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_batch_processing_pattern
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_corrupted_session_file_handling
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_with_varying_task_counts
FAILED tests/api/test_osb_criteria_mapping.py::test_osb_flow_with_direct_query
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_responses_for_record
FAILED tests/groupchat/test_agent_centric_tools.py::test_agent_tool_definition
FAILED tests/groupchat/test_agent_centric_tools.py::test_agent_announcement
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_respects_limits
FAILED tests/api/test_score_endpoints.py::TestScoreEndpointsIntegration::test_data_service_error_handling
FAILED tests/groupchat/test_agent_centric_tools.py::test_host_tool_collection
FAILED tests/groupchat/test_agent_centric_tools.py::test_tool_calling_flow`
  - Error: `......`
  - ... and 33 more

### Assertion Error (3 issues)
- `tests/test_configuration_unit.py`
  - Test: `TestConfigurationCaching::test_config_model_creation_performance
FAILED tests/tools/test_uploader.py::TestAsyncDataUploader::test_uploader_initialization_with_real_storage
FAILED tests/test_configuration_unit.py::TestConfigurationCaching::test_config_validation_caching
FAILED tests/groupchat/test_groupchat_orchestrator.py::test_agent_registry_population_and_get_config
FAILED tests/examples/test_logging_fail_fast_examples.py::TestRealWorldScenarioExamples::test_example_error_recovery_workflow
FAILED tests/tools/test_uploader.py::TestAsyncDataUploader::test_uploader_handles_storage_interface
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_update_agent_registry_joining
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[criteria-criteria_ordinary-5240]
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_update_agent_registry_leaving
FAILED tests/tools/test_uploader.py::TestAsyncDataUploaderErrorHandling::test_uploader_handles_invalid_buffer_size
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[criteria-criteria_hatefb_factorised-9000]
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_update_agent_registry_multiple_agents_same_tool
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[synth-synthesise-1000]
FAILED tests/tools/test_utils.py::test_b64_str_validator`
  - Error: `AssertionError: as......`
- `tests/tools/test_zotero_incremental_sync.py`
  - Test: `TestZoteroIncrementalSync::test_get_all_records_first_run
FAILED tests/integration/20core/test_config_reload.py::TestConfigurationReload::test_config_snapshot_saving
FAILED tests/integration/20core/test_config_reload.py::TestConfigurationReload::test_reload_configurations_success
FAILED tests/integration/20core/test_config_reload.py::TestConfigurationReload::test_reload_configurations_failure
FAILED tests/integration/20core/test_config_reload.py::TestConfigurationReload::test_config_snapshot_handles_missing_attributes
FAILED tests/integration/20core/test_config_reload.py::TestConfigReloadIntegration::test_startup_script_functionality
FAILED tests/examples/test_main_script_examples.py::TestMainScriptRealExecution::test_session_initialization_sequence
FAILED tests/tools/test_zotero_incremental_sync.py::TestZoteroIncrementalSync::test_get_all_records_with_force_full_sync
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_basic_usage_example
FAILED tests/groupchat/test_host_error_handling.py::TestHostAgentErrorHandling::test_wait_check_completions_clears_tracking
FAILED tests/groupchat/test_host_error_handling.py::TestHostAgentErrorHandling::test_custom_error_threshold
FAILED tests/unit/test_bm_weave_client_delegation.py::TestBMWeaveClientDelegation::test_delegation_to_execution_context_success
FAILED tests/unit/test_bm_weave_client_delegation.py::TestBMWeaveClientDelegation::test_execution_context_get_weave_client_failure_propagates
FAILED tests/groupchat/test_host_timeout_resilience.py::test_timeout_resilience_logging
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_step_request_routes_to_role_topic
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_flow_event_sent_before_step_request
FAILED tests/unit/test_bm_weave_client_delegation.py::TestBMWeaveClientDelegation::test_both_paths_return_valid_weave_client_interface
FAILED tests/unit/test_bm_weave_client_delegation.py::TestBMWeaveClientBehaviorDocumentation::test_documented_delegation_behavior
FAILED tests/unit/test_bm_weave_client_delegation.py::TestBMWeaveClientDelegation::test_import_isolation_in_fallback
FAILED tests/unit/test_bm_weave_client_delegation.py::TestBMWeaveClientBehaviorDocumentation::test_documented_fallback_behavior
FAILED tests/00initial/test_config.py::test_has_test_info`
  - Error: `AssertionError: a......`
- `tests/00initial/test_config.py`
  - Test: `test_save_dir`
  - Error: `AssertionError: assert......`
### Value Error (1 issues)
- `tests/groupchat/test_agent_announcement_behavior.py`
  - Test: `TestAgentAnnouncementBehavior::test_agent_responds_to_host_announcement
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_otel_gcp_trace_setup
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_base_agent_publish_defaults_to_agent_topic
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_does_not_respond_to_non_host_announcement
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_cli_init_with_overrides_real_config
FAILED tests/groupchat/test_llm_wrapper.py::test_call_chat_returns_text_without_tools
FAILED tests/groupchat/test_llm_wrapper.py::test_call_chat_tool_then_text_happy_path
FAILED tests/groupchat/test_llm_wrapper.py::test_call_chat_tool_loop_fails_when_exceeded
FAILED tests/runner/test_cli.py::test_main_console_mode`
  - Error: `ValueError: Invalid......`

## Failures by Test Category

- **root**: 9 failures
- **integration**: 8 failures
- **groupchat**: 5 failures
- **unit**: 5 failures
- **data**: 4 failures
- **api**: 3 failures
- **tools**: 3 failures
- **agents**: 1 failures
- **initial**: 1 failures
- **validation**: 1 failures

## Priority Fixes (Blocking Test Execution)

No critical blocking issues found.
