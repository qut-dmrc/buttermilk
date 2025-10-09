# Test Health Dashboard

## Summary Statistics

- **Total Test Files**: 158
- **Total Tests**: 928
- **Passed**: 487 (52.5%)
- **Failed**: 346 (37.3%)
- **Errors**: 68
- **Skipped**: 27
- **Collection Errors**: 0

## Failures by Error Type

### Other (28 issues)
- `tests/api/test_score_endpoints.py`
  - Test: `TestDataService::test_get_records_for_flow_without_scores
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_should_persist_message
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_records_for_flow_with_scores
FAILED tests/00initial/test_bm_singleton.py::test_singleton_between_modules
FAILED tests/agents/test_markdown.py::TestFramedStatementMarkdown::test_framed_statement_as_markdown
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_scores_for_record_no_data
FAILED tests/api/test_osb_criteria_mapping.py::test_osb_criteria_mapped_to_query
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_corrupted_session_file_handling
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_scores_for_record_with_data
FAILED tests/api/test_osb_criteria_mapping.py::test_osb_flow_with_direct_query
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_responses_for_record
FAILED tests/groupchat/test_agenttrace_runinfo.py::test_agenttrace_handles_missing_bm_gracefully
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_with_varying_task_counts
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_manual_context_management
FAILED tests/data/test_records_openaimessages.py::test_from_path_valid`
  - Error: `Attr......`
- `tests/api/test_score_endpoints.py`
  - Test: `TestDataService::test_get_record_by_id_found
FAILED tests/api/test_session_persistence.py::TestSessionStorageHelperMethods::test_get_or_create_session_data_existing_session
FAILED tests/groupchat/test_agent_centric_tools.py::test_agent_tool_definition
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_respects_limits
FAILED tests/api/test_score_endpoints.py::TestScoreEndpointsIntegration::test_data_service_error_handling
FAILED tests/api/test_score_endpoints.py::TestScoreAPIEndpoints::test_endpoint_registration
FAILED tests/api/test_session_persistence.py::TestSessionGCSArchival::test_finalize_session
FAILED tests/groupchat/test_agent_centric_tools.py::test_agent_announcement
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceEdgeCases::test_nested_traces
FAILED tests/api/test_session_persistence.py::TestSessionGCSArchival::test_archive_to_gcs_success
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_save_message
FAILED tests/groupchat/test_agent_centric_tools.py::test_host_tool_collection
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announces_on_initialize
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_get_session_messages
FAILED tests/groupchat/test_host_error_handling.py::TestHostAgentErrorHandling::test_wait_check_completions_clears_tracking
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_detects_available_tools
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_session_exists
FAILED tests/groupchat/test_agent_centric_tools.py::test_tool_calling_flow`
  - Error: `......`
- `tests/data/test_chromadb_sync_fix.py`
  - Test: `TestChromaDBSyncFix::test_path_preservation_logic
FAILED tests/groupchat/test_host_error_handling.py::TestHostAgentErrorHandling::test_custom_error_threshold
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_responds_to_host_announcement
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_does_not_respond_to_non_host_announcement
FAILED tests/groupchat/test_agent_centric_tools.py::test_registry_update_rebuilds_tools
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announces_on_cleanup
FAILED tests/api/test_session_persistence.py::TestSessionGCSArchival::test_finalize_session_always_attempts_archival
FAILED tests/integration/20core/test_config_reload.py::TestConfigReloadIntegration::test_startup_script_functionality
FAILED tests/groupchat/test_agenttrace_runinfo.py::test_agenttrace_serializes_runinfo_correctly
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announcement_includes_tools_and_message_types
FAILED tests/groupchat/test_agenttrace_runinfo.py::test_agenttrace_runinfo_is_json_serializable
FAILED tests/api/test_session_persistence.py::TestConfigurableSessionsDirectory::test_get_sessions_dir_with_bm_config
FAILED tests/groupchat/test_host_timeout_resilience.py::test_timeout_resilience_logging
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_handles_announcement_errors_gracefully
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_step_request_routes_to_role_topic
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announcement_in_invoke_lifecycle
FAILED tests/integration/20core/test_setup_infrastructure.py::test_database[prosocial-443205.testing.flow-schemas/flow.json]
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_flow_event_sent_before_step_request
FAILED tests/api/test_session_persistence.py::TestConfigurableSessionsDirectory::test_end_to_end_configurable_sessions_dir
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_manager_step_sends_ui_message_only
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_end_step_routes_to_main_topic
FAILED tests/integration/20core/test_setup_infrastructure.py::test_hf_login
FAILED tests/integration/test_llm_agent_refactoring.py::TestLLMAgentRefactoring::test_llmagent_handles_unexpected_error
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_unknown_role_still_routes_to_role_topic
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_route_tool_calls_uses_role_topics
FAILED tests/00initial/test_init.py::test_short_form_cli`
  - Error: `hydra.errors.Confi......`
  - ... and 25 more

### Value Error (1 issues)
- `tests/integration/test_llm_agent_refactoring.py`
  - Test: `TestLLMAgentRefactoring::test_llmagent_template_metadata_preserved
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_cli_init_with_real_config_first_session
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_cli_init_with_real_config_subsequent_session_inheritance
FAILED tests/integration/test_templating.py::TestMakeMessages::test_make_messages_with_record_placeholder
FAILED tests/integration/test_bootstrap_sequence_e2e.py::TestBootstrapSequenceErrorRecovery::test_bootstrap_with_invalid_configuration
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_cli_init_with_real_config_project_mismatch_error
FAILED tests/examples/test_logging_fail_fast_examples.py::TestValidationAndDebuggingExamples::test_example_verbose_level_mismatch_debugging
FAILED tests/runner/test_cli.py::test_main_console_mode`
  - Error: `ValueError: Invalid......`
### Assertion Error (1 issues)
- `tests/groupchat/test_host_topic_routing.py`
  - Test: `TestHostTopicRouting::test_base_agent_publish_with_topic_parameter
FAILED tests/tools/test_uploader.py::TestAsyncDataUploader::test_uploader_initialization_with_real_storage
FAILED tests/tools/test_uploader.py::TestAsyncDataUploader::test_uploader_handles_storage_interface
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[criteria-criteria_ordinary-5240]
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[criteria-criteria_hatefb_factorised-9000]
FAILED tests/tools/test_uploader.py::TestAsyncDataUploaderErrorHandling::test_uploader_handles_invalid_buffer_size
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_base_agent_publish_defaults_to_agent_topic
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[synth-synthesise-1000]
FAILED tests/tools/test_utils.py::test_b64_str_validator`
  - Error: `AssertionError: as......`

## Failures by Test Category

- **root**: 6 failures
- **data**: 5 failures
- **integration**: 5 failures
- **tools**: 3 failures
- **unit**: 3 failures
- **api**: 2 failures
- **runner**: 2 failures
- **examples**: 2 failures
- **groupchat**: 1 failures
- **validation**: 1 failures

## Priority Fixes (Blocking Test Execution)

No critical blocking issues found.
