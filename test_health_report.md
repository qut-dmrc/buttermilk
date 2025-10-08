# Test Health Dashboard

## Summary Statistics

- **Total Test Files**: 158
- **Total Tests**: 928
- **Passed**: 479 (51.6%)
- **Failed**: 357 (38.5%)
- **Errors**: 65
- **Skipped**: 27
- **Collection Errors**: 0

## Failures by Error Type

### Other (29 issues)
- `tests/api/test_score_endpoints.py`
  - Test: `TestDataService::test_get_records_for_flow_without_scores
FAILED tests/00initial/test_bm_singleton.py::test_singleton_between_modules
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_should_persist_message
FAILED tests/agents/test_markdown.py::TestFramedStatementMarkdown::test_framed_statement_as_markdown
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_records_for_flow_with_scores
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_corrupted_session_file_handling
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_scores_for_record_no_data
FAILED tests/api/test_osb_criteria_mapping.py::test_osb_criteria_mapped_to_query
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_manual_context_management
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_scores_for_record_with_data
FAILED tests/groupchat/test_agent_centric_tools.py::test_agent_tool_definition
FAILED tests/api/test_osb_criteria_mapping.py::test_osb_flow_with_direct_query
FAILED tests/groupchat/test_agenttrace_runinfo.py::test_agenttrace_handles_missing_bm_gracefully
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_with_varying_task_counts
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_record_by_id_found
FAILED tests/data/test_records_openaimessages.py::test_from_path_valid`
  - Error: `Attr......`
- `tests/groupchat/test_agent_centric_tools.py`
  - Test: `test_agent_announcement
FAILED tests/api/test_session_persistence.py::TestSessionGCSArchival::test_finalize_session
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_responses_for_record
FAILED tests/groupchat/test_agent_centric_tools.py::test_host_tool_collection
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_respects_limits
FAILED tests/api/test_session_persistence.py::TestSessionStorageHelperMethods::test_get_or_create_session_data_existing_session
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceEdgeCases::test_nested_traces
FAILED tests/api/test_score_endpoints.py::TestScoreEndpointsIntegration::test_data_service_error_handling
FAILED tests/groupchat/test_agent_centric_tools.py::test_tool_calling_flow`
  - Error: `......`
- `tests/api/test_score_endpoints.py`
  - Test: `TestScoreAPIEndpoints::test_endpoint_registration
FAILED tests/api/test_session_persistence.py::TestSessionGCSArchival::test_archive_to_gcs_success
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announces_on_initialize
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_save_message
FAILED tests/groupchat/test_agent_centric_tools.py::test_registry_update_rebuilds_tools
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_detects_available_tools
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_get_session_messages
FAILED tests/groupchat/test_host_error_handling.py::TestHostAgentErrorHandling::test_wait_check_completions_clears_tracking
FAILED tests/api/test_session_persistence.py::TestSessionGCSArchival::test_finalize_session_always_attempts_archival
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_responds_to_host_announcement
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_session_exists
FAILED tests/groupchat/test_agenttrace_runinfo.py::test_agenttrace_serializes_runinfo_correctly
FAILED tests/groupchat/test_host_error_handling.py::TestHostAgentErrorHandling::test_custom_error_threshold
FAILED tests/api/test_session_persistence.py::TestConfigurableSessionsDirectory::test_get_sessions_dir_with_bm_config
FAILED tests/groupchat/test_agenttrace_runinfo.py::test_agenttrace_runinfo_is_json_serializable
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_does_not_respond_to_non_host_announcement
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announces_on_cleanup
FAILED tests/integration/20core/test_config_reload.py::TestConfigReloadIntegration::test_startup_script_functionality
FAILED tests/integration/20core/test_bm_session_isolation.py::TestBMInjectionSystem::test_orchestrator_bm_injection
FAILED tests/api/test_session_persistence.py::TestConfigurableSessionsDirectory::test_end_to_end_configurable_sessions_dir
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announcement_includes_tools_and_message_types
FAILED tests/data/test_chromadb_sync_fix.py::TestChromaDBSyncFix::test_path_preservation_logic
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_handles_announcement_errors_gracefully
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announcement_in_invoke_lifecycle
FAILED tests/integration/test_osb_websocket_integration.py::TestOSBWebSocketErrorHandling::test_osb_websocket_connection_recovery
FAILED tests/integration/test_osb_websocket_integration.py::TestOSBWebSocketErrorHandling::test_osb_agent_failure_handling
FAILED tests/integration/20core/test_setup_infrastructure.py::test_database[prosocial-443205.testing.flow-schemas/flow.json]
FAILED tests/integration/20core/test_bm_session_isolation.py::TestBMInjectionSystem::test_end_to_end_bm_flow
FAILED tests/00initial/test_init.py::test_short_form_cli`
  - Error: `hydra.errors.Confi......`
  - ... and 26 more

### Type Error (3 issues)
- `tests/test_async_init.py`
  - Test: `test_bm_ensure_initialized`
  - Error: `TypeError: crea......`
- `tests/test_bigquery_storage.py`
  - Test: `TestBigQueryStorage::test_clustering_fields_with_both_fields
FAILED tests/storage/test_file_storage_column_mapping.py::TestFileStorageIntegration::test_cloud_path_simulation
FAILED tests/test_bm_async_init.py::TestBMAsyncInitialization::test_initialization_error_handling
FAILED tests/test_async_init.py::test_create_session_bm_async`
  - Error: `TypeError: cr......`
- `tests/test_startup_performance_unit.py`
  - Test: `TestBMInitialization::test_cloud_manager_is_lazy
FAILED tests/00initial/test_init.py::test_short_form_nb`
  - Error: `TypeError: nb_init(......`
### Value Error (1 issues)
- `tests/agents/test_agent_name.py`
  - Test: `test_agent_name_generation_empty_components
FAILED tests/storage/test_file_storage_column_mapping.py::TestFileStorageColumnMapping::test_empty_metadata_mapping
FAILED tests/integration/test_logging_fail_fast_integration.py::TestFailFastIntegrationExamples::test_error_handling_and_recovery_example
FAILED tests/agents/test_agent_name.py::test_agent_name_generation_jmespath_failure
FAILED tests/runner/test_cli.py::test_main_console_mode`
  - Error: `ValueError: Invalid......`
### Assertion Error (1 issues)
- `tests/test_startup_performance_unit.py`
  - Test: `TestBMInitialization::test_weave_import_is_cached
FAILED tests/api/test_message_service_tokens.py::TestMessageServiceTokenExtraction::test_format_message_extracts_tokens_from_agent_trace
FAILED tests/groupchat/test_groupchat_orchestrator.py::test_agent_registry_population_and_get_config
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_update_agent_registry_joining
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_update_agent_registry_leaving
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_update_agent_registry_multiple_agents_same_tool
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_create_registry_summary
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_host_listens_for_announcements
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_host_announces_itself_and_receives_responses
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_calculation_in_wait_method
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_with_zero_tasks
FAILED tests/tools/test_uploader.py::TestAsyncDataUploader::test_uploader_initialization_with_real_storage
FAILED tests/tools/test_uploader.py::TestAsyncDataUploader::test_uploader_handles_storage_interface
FAILED tests/tools/test_uploader.py::TestAsyncDataUploaderErrorHandling::test_uploader_handles_invalid_buffer_size
FAILED tests/groupchat/test_host_timeout_resilience.py::test_timeout_resilience_logging
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_step_request_routes_to_role_topic
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[criteria-criteria_ordinary-5240]
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_flow_event_sent_before_step_request
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[criteria-criteria_hatefb_factorised-9000]
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_manager_step_sends_ui_message_only
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[synth-synthesise-1000]
FAILED tests/tools/test_utils.py::test_b64_str_validator`
  - Error: `AssertionError: as......`

## Failures by Test Category

- **root**: 14 failures
- **integration**: 5 failures
- **data**: 3 failures
- **api**: 2 failures
- **groupchat**: 2 failures
- **storage**: 2 failures
- **tools**: 2 failures
- **unit**: 2 failures
- **agents**: 1 failures
- **validation**: 1 failures

## Priority Fixes (Blocking Test Execution)

No critical blocking issues found.
