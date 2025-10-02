# Test Health Dashboard

## Summary Statistics

- **Total Test Files**: 156
- **Total Tests**: 929
- **Passed**: 463 (49.8%)
- **Failed**: 373 (40.2%)
- **Errors**: 67
- **Skipped**: 26
- **Collection Errors**: 0

## Failures by Error Type

### Other (36 issues)
- `tests/api/test_message_service_tokens.py`
  - Test: `TestMessageServiceTokenExtraction::test_format_message_extracts_tokens_from_agent_trace
FAILED tests/agents/test_agentoutput.py::test_actual_judge_reasons_direct_dump
FAILED tests/00initial/test_bm_singleton.py::test_singleton_instance`
  - Error: `Assert......`
- `tests/agents/test_agentoutput.py`
  - Test: `test_actual_agent_trace_full_dump_includes_nested_outputs
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announces_on_cleanup
FAILED tests/api/test_score_endpoints.py::TestScoreAPIEndpoints::test_endpoint_registration
FAILED tests/agents/test_agentoutput.py::test_actual_agent_trace_full_dump_with_default_outputs
FAILED tests/00initial/test_bm_singleton.py::test_singleton_between_modules
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_host_listens_for_announcements
FAILED tests/agents/test_markdown.py::TestFramedStatementMarkdown::test_framed_statement_as_markdown
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announcement_includes_tools_and_message_types
FAILED tests/agents/test_fetch.py::TestFetch::test_load_data`
  - Error: `AttributeError......`
- `tests/data/test_records_openaimessages.py`
  - Test: `test_as_openai_message_with_media
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_save_message
FAILED tests/groupchat/test_agenttrace_runinfo.py::test_agenttrace_serializes_runinfo_correctly
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_host_announces_itself_and_receives_responses
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_handles_announcement_errors_gracefully
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_record_by_id_found
FAILED tests/groupchat/test_agenttrace_runinfo.py::test_agenttrace_runinfo_is_json_serializable
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_with_text
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_get_session_messages
FAILED tests/groupchat/test_agenttrace_runinfo.py::test_agenttrace_handles_missing_bm_gracefully
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_record_by_id_not_found
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_no_media_no_text
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_calculation_in_wait_method
FAILED tests/data/test_chromadb_sync_fix.py::TestChromaDBSyncFix::test_path_preservation_logic
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announcement_in_invoke_lifecycle
FAILED tests/api/test_osb_criteria_mapping.py::test_osb_criteria_mapped_to_query
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_session_exists
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_records_for_flow_without_scores
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_with_zero_tasks
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_should_persist_message
FAILED tests/groupchat/test_agent_centric_tools.py::test_agent_tool_definition
FAILED tests/api/test_osb_criteria_mapping.py::test_osb_flow_with_direct_query
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_records_for_flow_with_scores
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_agent_integration_example
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_with_varying_task_counts
FAILED tests/groupchat/test_agent_centric_tools.py::test_agent_announcement
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_corrupted_session_file_handling
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_scores_for_record_no_data
FAILED tests/groupchat/test_agent_centric_tools.py::test_host_tool_collection
FAILED tests/agents/test_markdown.py::TestExecutionTraceMarkdown::test_agent_trace_with_judge_output
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_respects_limits
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_scores_for_record_with_data
FAILED tests/groupchat/test_agent_centric_tools.py::test_tool_calling_flow`
  - Error: `......`
  - ... and 33 more

### Assertion Error (3 issues)
- `tests/tools/test_tmdb.py`
  - Test: `TestTMDBGetAvailability::test_get_availability_api_error_handling
FAILED tests/tools/test_uploader.py::TestAsyncDataUploader::test_uploader_initialization_with_real_storage
FAILED tests/tools/test_uploader.py::TestAsyncDataUploader::test_uploader_handles_storage_interface
FAILED tests/agents/test_fetch.py::TestFetch::test_ingest_news[semaphor web]
FAILED tests/tools/test_uploader.py::TestAsyncDataUploaderErrorHandling::test_uploader_handles_invalid_buffer_size
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[criteria-criteria_ordinary-5240]
FAILED tests/tools/test_utils.py::test_b64_str_validator`
  - Error: `AssertionError: as......`
- `tests/integration/20core/test_bm_session_isolation.py`
  - Test: `TestBackwardCompatibility::test_orchestrator_without_session_bm
FAILED tests/tools/test_zotero_incremental_sync.py::TestZoteroIncrementalSync::test_get_all_records_first_run
FAILED tests/examples/test_main_script_examples.py::TestMainScriptRealExecution::test_session_initialization_sequence
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_manager_step_sends_ui_message_only
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_basic_usage_example
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_end_step_routes_to_main_topic
FAILED tests/tools/test_zotero_incremental_sync.py::TestZoteroIncrementalSync::test_get_all_records_incremental
FAILED tests/integration/test_logging_fail_fast_integration.py::TestFailFastIntegrationExamples::test_verbose_logging_workflow_example
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_unknown_role_still_routes_to_role_topic
FAILED tests/unit/test_async_uploader_timestamp.py::TestAsyncDataUploaderTimestamp::test_no_timestamp_suffix_when_file_not_exists
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_route_tool_calls_uses_role_topics
FAILED tests/unit/test_async_uploader_timestamp.py::TestAsyncDataUploaderTimestamp::test_explicit_timestamp_suffix_override
FAILED tests/unit/test_bm_injection.py::TestBMInjectionMechanism::test_flowrunner_fallback_to_global
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_base_agent_publish_with_topic_parameter
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_base_agent_publish_defaults_to_agent_topic
FAILED tests/unit/test_bm_weave_client_delegation.py::TestBMWeaveClientDelegation::test_delegation_to_execution_context_success
FAILED tests/groupchat/test_llm_wrapper.py::test_call_chat_returns_text_without_tools
FAILED tests/groupchat/test_llm_wrapper.py::test_call_chat_tool_then_text_happy_path
FAILED tests/groupchat/test_llm_wrapper.py::test_call_chat_tool_loop_fails_when_exceeded
FAILED tests/integration/test_logging_fail_fast_integration.py::TestFailFastIntegrationExamples::test_error_handling_and_recovery_example
FAILED tests/00initial/test_config.py::test_has_test_info`
  - Error: `AssertionError: a......`
- `tests/unit/test_bm_weave_client_delegation.py`
  - Test: `TestBMWeaveClientDelegation::test_execution_context_get_weave_client_failure_propagates
FAILED tests/00initial/test_config.py::test_save_dir`
  - Error: `AssertionError: assert......`
### Value Error (1 issues)
- `tests/groupchat/test_host_timeout_resilience.py`
  - Test: `test_timeout_resilience_logging
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_step_request_routes_to_role_topic
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_flow_event_sent_before_step_request
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_manual_context_management
FAILED tests/examples/test_logging_fail_fast_examples.py::TestBasicUsageExamples::test_example_non_verbose_logging_setup
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_otel_gcp_trace_setup
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceEdgeCases::test_nested_traces
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announces_on_initialize
FAILED tests/runner/test_cli.py::test_main_console_mode`
  - Error: `ValueError: Invalid......`

## Failures by Test Category

- **root**: 8 failures
- **unit**: 7 failures
- **data**: 6 failures
- **integration**: 5 failures
- **tools**: 4 failures
- **groupchat**: 3 failures
- **agents**: 2 failures
- **storage**: 2 failures
- **api**: 1 failures
- **examples**: 1 failures
- **validation**: 1 failures

## Priority Fixes (Blocking Test Execution)

No critical blocking issues found.
