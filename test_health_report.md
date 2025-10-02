# Test Health Dashboard

## Summary Statistics

- **Total Test Files**: 156
- **Total Tests**: 929
- **Passed**: 468 (50.4%)
- **Failed**: 368 (39.6%)
- **Errors**: 67
- **Skipped**: 26
- **Collection Errors**: 0

## Failures by Error Type

### Other (36 issues)
- `tests/groupchat/test_host_agent_registry.py`
  - Test: `TestHostAgentRegistry::test_host_listens_for_announcements
FAILED tests/00initial/test_bm_singleton.py::test_singleton_instance`
  - Error: `Assert......`
- `tests/api/test_message_service_tokens.py`
  - Test: `TestMessageServiceTokenExtraction::test_format_message_extracts_tokens_from_agent_trace
FAILED tests/00initial/test_bm_singleton.py::test_singleton_between_modules
FAILED tests/agents/test_agentoutput.py::test_actual_judge_reasons_direct_dump
FAILED tests/api/test_score_endpoints.py::TestScoreAPIEndpoints::test_endpoint_registration
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_host_announces_itself_and_receives_responses
FAILED tests/agents/test_agentoutput.py::test_actual_agent_trace_full_dump_includes_nested_outputs
FAILED tests/groupchat/test_agenttrace_runinfo.py::test_agenttrace_serializes_runinfo_correctly
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announces_on_cleanup
FAILED tests/agents/test_agentoutput.py::test_actual_agent_trace_full_dump_with_default_outputs
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_record_by_id_found
FAILED tests/groupchat/test_agenttrace_runinfo.py::test_agenttrace_runinfo_is_json_serializable
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_save_message
FAILED tests/groupchat/test_agenttrace_runinfo.py::test_agenttrace_handles_missing_bm_gracefully
FAILED tests/agents/test_markdown.py::TestFramedStatementMarkdown::test_framed_statement_as_markdown
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_record_by_id_not_found
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_with_media
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_calculation_in_wait_method
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announcement_includes_tools_and_message_types
FAILED tests/data/test_chromadb_sync_fix.py::TestChromaDBSyncFix::test_path_preservation_logic
FAILED tests/agents/test_fetch.py::TestFetch::test_load_data`
  - Error: `AttributeError......`
- `tests/api/test_session_persistence.py`
  - Test: `TestSessionStorageService::test_get_session_messages
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_with_text
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_records_for_flow_without_scores
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_with_zero_tasks
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_no_media_no_text
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_session_exists
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_records_for_flow_with_scores
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_agent_integration_example
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_handles_announcement_errors_gracefully
FAILED tests/api/test_osb_criteria_mapping.py::test_osb_criteria_mapped_to_query
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_should_persist_message
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_with_varying_task_counts
FAILED tests/api/test_osb_criteria_mapping.py::test_osb_flow_with_direct_query
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_scores_for_record_no_data
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announcement_in_invoke_lifecycle
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_respects_limits
FAILED tests/00initial/test_bm_singleton.py::test_import_singleton_from_different_modules
FAILED tests/groupchat/test_agent_centric_tools.py::test_agent_tool_definition
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_corrupted_session_file_handling
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_scores_for_record_with_data
FAILED tests/groupchat/test_agent_centric_tools.py::test_agent_announcement
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_responses_for_record
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_batch_processing_pattern
FAILED tests/groupchat/test_agent_centric_tools.py::test_host_tool_collection
FAILED tests/agents/test_markdown.py::TestExecutionTraceMarkdown::test_agent_trace_with_judge_output
FAILED tests/groupchat/test_agent_centric_tools.py::test_tool_calling_flow`
  - Error: `......`
  - ... and 33 more

### Assertion Error (3 issues)
- `tests/tools/test_uploader.py`
  - Test: `TestAsyncDataUploader::test_uploader_initialization_with_real_storage
FAILED tests/groupchat/test_groupchat_orchestrator.py::test_agent_registry_population_and_get_config
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_update_agent_registry_joining
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_update_agent_registry_leaving
FAILED tests/tools/test_uploader.py::TestAsyncDataUploader::test_uploader_handles_storage_interface
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_update_agent_registry_multiple_agents_same_tool
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_create_registry_summary
FAILED tests/tools/test_uploader.py::TestAsyncDataUploaderErrorHandling::test_uploader_handles_invalid_buffer_size
FAILED tests/examples/test_logging_fail_fast_examples.py::TestRealWorldScenarioExamples::test_example_error_recovery_workflow
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[criteria-criteria_ordinary-5240]
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[criteria-criteria_hatefb_factorised-9000]
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[synth-synthesise-1000]
FAILED tests/tools/test_utils.py::test_b64_str_validator`
  - Error: `AssertionError: as......`
- `tests/tools/test_zotero_incremental_sync.py`
  - Test: `TestZoteroIncrementalSync::test_get_all_records_first_run
FAILED tests/integration/test_logging_fail_fast_integration.py::TestFailFastIntegrationExamples::test_verbose_logging_workflow_example
FAILED tests/examples/test_main_script_examples.py::TestMainScriptRealExecution::test_session_initialization_sequence
FAILED tests/tools/test_zotero_incremental_sync.py::TestZoteroIncrementalSync::test_get_all_records_incremental
FAILED tests/unit/test_async_uploader_timestamp.py::TestAsyncDataUploaderTimestamp::test_no_timestamp_suffix_when_file_not_exists
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_basic_usage_example
FAILED tests/unit/test_async_uploader_timestamp.py::TestAsyncDataUploaderTimestamp::test_explicit_timestamp_suffix_override
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_manager_step_sends_ui_message_only
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_end_step_routes_to_main_topic
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_unknown_role_still_routes_to_role_topic
FAILED tests/unit/test_bm_injection.py::TestBMInjectionMechanism::test_flowrunner_fallback_to_global
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_route_tool_calls_uses_role_topics
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_base_agent_publish_with_topic_parameter
FAILED tests/groupchat/test_llm_wrapper.py::test_create_schema_with_base_model_content_normalizes_and_parses
FAILED tests/unit/test_bm_weave_client_delegation.py::TestBMWeaveClientDelegation::test_delegation_to_execution_context_success
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_base_agent_publish_defaults_to_agent_topic
FAILED tests/integration/test_logging_fail_fast_integration.py::TestFailFastIntegrationExamples::test_error_handling_and_recovery_example
FAILED tests/integration/test_osb_websocket_integration.py::TestOSBWebSocketIntegration::test_osb_websocket_message_validation
FAILED tests/groupchat/test_llm_wrapper.py::test_call_chat_returns_text_without_tools
FAILED tests/integration/20core/test_bm_session_isolation.py::TestBMInjectionSystem::test_orchestrator_bm_injection
FAILED tests/groupchat/test_llm_wrapper.py::test_call_chat_tool_then_text_happy_path
FAILED tests/groupchat/test_llm_wrapper.py::test_call_chat_tool_loop_fails_when_exceeded
FAILED tests/unit/test_bm_weave_client_delegation.py::TestBMWeaveClientDelegation::test_execution_context_get_weave_client_failure_propagates
FAILED tests/integration/test_osb_websocket_integration.py::TestOSBWebSocketIntegration::test_osb_websocket_response_streaming
FAILED tests/integration/test_osb_websocket_integration.py::TestOSBWebSocketIntegration::test_osb_session_isolation
FAILED tests/integration/test_osb_websocket_integration.py::TestOSBWebSocketErrorHandling::test_osb_websocket_connection_recovery
FAILED tests/integration/20core/test_bm_session_isolation.py::TestBMInjectionSystem::test_end_to_end_bm_flow
FAILED tests/unit/test_bm_weave_client_delegation.py::TestBMWeaveClientDelegation::test_both_paths_return_valid_weave_client_interface
FAILED tests/unit/test_bootstrap_query_runner_injection.py::TestBootstrapQueryRunnerInjection::test_bootstrap_query_runner_error_handling
FAILED tests/integration/20core/test_bm_session_isolation.py::TestBMInjectionSystem::test_session_isolation_between_runners
FAILED tests/unit/test_bootstrap_query_runner_injection.py::TestBootstrapQueryRunnerInjection::test_bootstrap_session_bm_has_run_query_convenience_method
FAILED tests/unit/test_bm_weave_client_delegation.py::TestBMWeaveClientDelegation::test_import_isolation_in_fallback
FAILED tests/00initial/test_config.py::test_has_test_info`
  - Error: `AssertionError: a......`
- `tests/00initial/test_config.py`
  - Test: `test_save_dir`
  - Error: `AssertionError: assert......`
### Value Error (1 issues)
- `tests/integration/20core/test_bm_session_isolation.py`
  - Test: `TestBackwardCompatibility::test_orchestrator_without_session_bm
FAILED tests/groupchat/test_host_error_handling.py::TestHostAgentErrorHandling::test_wait_check_completions_clears_tracking
FAILED tests/groupchat/test_agent_centric_tools.py::test_registry_update_rebuilds_tools
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_inject_parent_trace_example
FAILED tests/groupchat/test_host_error_handling.py::TestHostAgentErrorHandling::test_custom_error_threshold
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_traceloop_missing_api_key_validation
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_error_handling_example
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_manual_context_management
FAILED tests/runner/test_cli.py::test_main_console_mode`
  - Error: `ValueError: Invalid......`

## Failures by Test Category

- **root**: 9 failures
- **integration**: 7 failures
- **unit**: 7 failures
- **tools**: 6 failures
- **api**: 3 failures
- **examples**: 2 failures
- **initial**: 2 failures
- **groupchat**: 1 failures
- **storage**: 1 failures
- **utils**: 1 failures
- **data**: 1 failures

## Priority Fixes (Blocking Test Execution)

No critical blocking issues found.
