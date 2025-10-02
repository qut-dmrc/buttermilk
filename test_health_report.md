# Test Health Dashboard

## Summary Statistics

- **Total Test Files**: 156
- **Total Tests**: 929
- **Passed**: 473 (50.9%)
- **Failed**: 360 (38.8%)
- **Errors**: 70
- **Skipped**: 26
- **Collection Errors**: 0

## Failures by Error Type

### Other (34 issues)
- `tests/api/test_message_service_tokens.py`
  - Test: `TestMessageServiceTokenExtraction::test_format_message_extracts_tokens_from_agent_trace
FAILED tests/groupchat/test_agenttrace_runinfo.py::test_agenttrace_serializes_runinfo_correctly
FAILED tests/00initial/test_bm_singleton.py::test_singleton_instance`
  - Error: `Assert......`
- `tests/groupchat/test_agenttrace_runinfo.py`
  - Test: `test_agenttrace_runinfo_is_json_serializable
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announces_on_cleanup
FAILED tests/agents/test_agentoutput.py::test_actual_judge_reasons_direct_dump
FAILED tests/groupchat/test_agenttrace_runinfo.py::test_agenttrace_handles_missing_bm_gracefully
FAILED tests/00initial/test_bm_singleton.py::test_singleton_between_modules
FAILED tests/agents/test_agentoutput.py::test_actual_agent_trace_full_dump_includes_nested_outputs
FAILED tests/agents/test_agentoutput.py::test_actual_agent_trace_full_dump_with_default_outputs
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announcement_includes_tools_and_message_types
FAILED tests/data/test_chromadb_sync_fix.py::TestChromaDBSyncFix::test_path_preservation_logic
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_record_by_id_found
FAILED tests/agents/test_fetch.py::TestFetch::test_load_data`
  - Error: `AttributeError......`
- `tests/api/test_score_endpoints.py`
  - Test: `TestScoreAPIEndpoints::test_endpoint_registration
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_handles_announcement_errors_gracefully
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_host_listens_for_announcements
FAILED tests/agents/test_markdown.py::TestFramedStatementMarkdown::test_framed_statement_as_markdown
FAILED tests/api/test_osb_criteria_mapping.py::test_osb_criteria_mapped_to_query
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_save_message
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_record_by_id_not_found
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_records_for_flow_without_scores
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announcement_in_invoke_lifecycle
FAILED tests/api/test_osb_criteria_mapping.py::test_osb_flow_with_direct_query
FAILED tests/groupchat/test_agent_centric_tools.py::test_agent_tool_definition
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_host_announces_itself_and_receives_responses
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_get_session_messages
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_records_for_flow_with_scores
FAILED tests/groupchat/test_agent_centric_tools.py::test_agent_announcement
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_with_media
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_calculation_in_wait_method
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_session_exists
FAILED tests/groupchat/test_agent_centric_tools.py::test_host_tool_collection
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_agent_integration_example
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_with_text
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_scores_for_record_no_data
FAILED tests/00initial/test_bm_singleton.py::test_import_singleton_from_different_modules
FAILED tests/groupchat/test_agent_centric_tools.py::test_tool_calling_flow`
  - Error: `......`
  - ... and 31 more

### Assertion Error (3 issues)
- `tests/tools/test_uploader.py`
  - Test: `TestAsyncDataUploader::test_uploader_initialization_with_real_storage
FAILED tests/tools/test_uploader.py::TestAsyncDataUploader::test_uploader_handles_storage_interface
FAILED tests/tools/test_uploader.py::TestAsyncDataUploaderErrorHandling::test_uploader_handles_invalid_buffer_size
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[criteria-criteria_ordinary-5240]
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[criteria-criteria_hatefb_factorised-9000]
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[synth-synthesise-1000]
FAILED tests/tools/test_utils.py::test_b64_str_validator`
  - Error: `AssertionError: as......`
- `tests/examples/test_main_script_examples.py`
  - Test: `TestMainScriptRealExecution::test_session_initialization_sequence
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_basic_usage_example
FAILED tests/tools/test_zotero_incremental_sync.py::TestZoteroIncrementalSync::test_get_all_records_first_run
FAILED tests/tools/test_zotero_incremental_sync.py::TestZoteroIncrementalSync::test_get_all_records_incremental
FAILED tests/unit/test_async_uploader_timestamp.py::TestAsyncDataUploaderTimestamp::test_no_timestamp_suffix_when_file_not_exists
FAILED tests/unit/test_async_uploader_timestamp.py::TestAsyncDataUploaderTimestamp::test_explicit_timestamp_suffix_override
FAILED tests/groupchat/test_groupchat_orchestrator.py::test_agent_registry_population_and_get_config
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_update_agent_registry_joining
FAILED tests/unit/test_bm_injection.py::TestBMInjectionMechanism::test_flowrunner_fallback_to_global
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_update_agent_registry_leaving
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_update_agent_registry_multiple_agents_same_tool
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_create_registry_summary
FAILED tests/00initial/test_config.py::test_has_test_info`
  - Error: `AssertionError: a......`
- `tests/00initial/test_config.py`
  - Test: `test_save_dir`
  - Error: `AssertionError: assert......`
### Value Error (1 issues)
- `tests/integration/test_llm_agent_refactoring.py`
  - Test: `TestLLMAgentRefactoring::test_llmagent_template_metadata_preserved
FAILED tests/integration/20core/test_bm_session_isolation.py::TestBMInjectionSystem::test_end_to_end_bm_flow
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceEdgeCases::test_nested_traces
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_otel_gcp_trace_setup
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announces_on_initialize
FAILED tests/integration/20core/test_bm_session_isolation.py::TestBMInjectionSystem::test_session_isolation_between_runners
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_detects_available_tools
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_cli_init_with_real_config_first_session
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_responds_to_host_announcement
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_does_not_respond_to_non_host_announcement
FAILED tests/runner/test_cli.py::test_main_console_mode`
  - Error: `ValueError: Invalid......`

## Failures by Test Category

- **root**: 10 failures
- **integration**: 8 failures
- **tools**: 4 failures
- **data**: 3 failures
- **unit**: 3 failures
- **api**: 2 failures
- **groupchat**: 2 failures
- **examples**: 2 failures
- **storage**: 1 failures
- **agents**: 1 failures
- **initial**: 1 failures
- **validation**: 1 failures

## Priority Fixes (Blocking Test Execution)

No critical blocking issues found.
