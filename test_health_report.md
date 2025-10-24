# Test Health Dashboard

## Summary Statistics

- **Total Test Files**: 146
- **Total Tests**: 848
- **Passed**: 422 (49.8%)
- **Failed**: 253 (29.8%)
- **Errors**: 155
- **Skipped**: 18
- **Collection Errors**: 9

## Failures by Error Type

### Other (25 issues)
- `tests/api/test_score_endpoints.py`
  - Test: `TestScoreAPIEndpoints::test_endpoint_registration
FAILED tests/agents/test_markdown.py::TestFramedStatementMarkdown::test_framed_statement_as_markdown
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_manual_context_management
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announcement_in_invoke_lifecycle
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_host_announces_itself_and_receives_responses
FAILED tests/groupchat/test_agent_centric_tools.py::test_agent_tool_definition
FAILED tests/agents/test_agentoutput.py::test_actual_agent_trace_full_dump_includes_nested_outputs
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_should_persist_message
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceEdgeCases::test_nested_traces
FAILED tests/api/test_osb_criteria_mapping.py::test_osb_criteria_mapped_to_query
FAILED tests/groupchat/test_agent_centric_tools.py::test_agent_announcement
FAILED tests/api/test_osb_criteria_mapping.py::test_osb_flow_with_direct_query
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announces_on_initialize
FAILED tests/agents/test_agentoutput.py::test_actual_agent_trace_full_dump_with_default_outputs
FAILED tests/groupchat/test_agent_centric_tools.py::test_host_tool_collection
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_detects_available_tools
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_responds_to_host_announcement
FAILED tests/groupchat/test_agent_centric_tools.py::test_tool_calling_flow`
  - Error: `......`
- `tests/runner/test_cli.py`
  - Test: `test_main_slack_mode_event_loop`
  - Error: `ValueError......`
- `tests/runner/test_cli.py`
  - Test: `test_main_slack_mode_environment_vars`
  - Error: `Valu......`
  - ... and 22 more

### Import Error (9 issues)
- `tests/endtoend/flows/test_embeddings.py ___________`
  - Error: `ImportError while importing test module '/home/nic/src/buttermilk/tests/endtoend/flows/test_embeddings.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../.local/share/uv/python/cpython-3.12.0-linux-x86_64-gnu/lib/python3.12/importlib/__init__.py:90: in import_m...`
- `tests/endtoend/test_pipeline_live.py _____________`
  - Error: `ImportError while importing test module '/home/nic/src/buttermilk/tests/endtoend/test_pipeline_live.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../.local/share/uv/python/cpython-3.12.0-linux-x86_64-gnu/lib/python3.12/importlib/__init__.py:90: in import_modu...`
- `tests/endtoend/test_tmdb_live.py _______________`
  - Error: `ImportError while importing test module '/home/nic/src/buttermilk/tests/endtoend/test_tmdb_live.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../.local/share/uv/python/cpython-3.12.0-linux-x86_64-gnu/lib/python3.12/importlib/__init__.py:90: in import_module
 ...`
  - ... and 6 more

### Assertion Error (2 issues)
- `tests/test_lazy_loading_unit.py`
  - Test: `TestAsyncBackgroundOperations::test_background_config_saving
FAILED tests/test_record_content_validation.py::TestRecordContentValidation::test_osb_field_mapping_scenario
FAILED tests/test_lazy_loading_unit.py::TestCloudManagerLazyLoading::test_cloud_manager_clients_are_lazy
FAILED tests/test_lazy_loading_unit.py::TestCloudManagerLazyLoading::test_cloud_manager_credentials_cached
FAILED tests/test_record_content_validation.py::TestRecordContentValidation::test_content_validation_prevents_silent_failures
FAILED tests/test_lazy_loading_unit.py::TestLLMManagerLazyLoading::test_llm_connections_cache_loading
FAILED tests/test_record_content_validation.py::TestRecordFieldCounts::test_record_has_reasonable_field_count
FAILED tests/test_record_content_validation.py::TestRecordContentValidation::test_record_requires_content
FAILED tests/test_record_content_validation.py::TestRecordContentValidation::test_record_rejects_none_content
FAILED tests/test_otel_gcp_integration.py::TestOtelGCPIntegration::test_otel_gcp_and_wandb_exporters_configured
FAILED tests/test_simple_rag_agent.py::TestSimpleRagAgent::test_research_result_structure
FAILED tests/test_lazy_loading_unit.py::TestAsyncBackgroundOperations::test_ip_fetching_is_background
FAILED tests/test_simple_rag_agent.py::TestRagZotero::test_rag_zotero_forces_zotero_output
FAILED tests/test_record_content_validation.py::TestStructuredDataHandling::test_record_preserves_structured_metadata
FAILED tests/test_simple_rag_agent.py::TestRagZotero::test_zotero_result_structure
FAILED tests/test_simple_rag_agent.py::TestSimpleRagAgent::test_rag_agent_forces_structured_output
FAILED tests/test_simple_rag_agent.py::TestSimpleRagAgent::test_rag_agent_custom_template
FAILED tests/test_startup_performance_unit.py::TestConfigurationValidation::test_storage_config_validation
FAILED tests/test_lazy_loading_unit.py::TestMemoryEfficiency::test_large_objects_not_created_unnecessarily
FAILED tests/examples/test_logging_fail_fast_examples.py::TestBasicUsageExamples::test_example_non_verbose_logging_setup
FAILED tests/test_startup_performance_unit.py::TestBMInitialization::test_bm_creation_is_fast
FAILED tests/test_startup_performance_unit.py::TestSecretsManagerOptimizations::test_secrets_manager_client_is_lazy
FAILED tests/test_startup_performance_unit.py::TestBMInitialization::test_llm_property_is_lazy
FAILED tests/test_otel_gcp_integration.py::TestOtelGCPIntegration::test_otel_gcp_from_env_variable
FAILED tests/test_otel_gcp_integration.py::TestOtelGCPIntegration::test_otel_no_exporters_configured
FAILED tests/test_otel_gcp_integration.py::TestCloudPyGCPTracing::test_setup_google_tracing_with_correct_imports
FAILED tests/groupchat/test_groupchat_orchestrator.py::test_agent_registry_population_and_get_config
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_update_agent_registry_joining
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_update_agent_registry_leaving
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_update_agent_registry_multiple_agents_same_tool
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_create_registry_summary
FAILED tests/test_startup_performance_unit.py::TestAsyncCacheOperations::test_llm_cache_writing_is_async
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_weave_missing_api_key_fails_fast
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_traceloop_missing_api_key_validation
FAILED tests/test_startup_performance_unit.py::TestBMInitialization::test_secret_manager_is_lazy
FAILED tests/test_startup_performance_unit.py::TestBMInitialization::test_cloud_manager_is_lazy
FAILED tests/test_startup_performance_unit.py::TestBMInitialization::test_weave_import_is_cached
FAILED tests/examples/test_logging_fail_fast_examples.py::TestValidationAndDebuggingExamples::test_example_debugging_broken_logging
FAILED tests/examples/test_logging_fail_fast_examples.py::TestValidationAndDebuggingExamples::test_example_verbose_level_mismatch_debugging
FAILED tests/api/test_message_service_tokens.py::TestMessageServiceTokenExtraction::test_format_message_extracts_tokens_from_agent_trace
FAILED tests/agents/test_fetch.py::TestFetch::test_ingest_news[abc news web]
FAILED tests/agents/test_fetch.py::TestFetch::test_ingest_news[semaphor web]
FAILED tests/groupchat/test_host_agent_registry.py::TestHostAgentRegistry::test_host_listens_for_announcements
FAILED tests/test_storage_config_validation.py::TestStorageConfigValidation::test_invalid_storage_type_fallback
FAILED tests/test_structured_tool_handling.py::TestStructuredToolHandling::test_tool_invocation_with_role_prefix
FAILED tests/test_storage_config_validation.py::TestStorageConfigValidation::test_legacy_datasource_config_compatibility
FAILED tests/test_structured_tool_handling.py::TestStructuredToolHandling::test_tool_parameters_passed_correctly
FAILED tests/test_structured_llmhost_unit.py::TestStructuredLLMHostInitialization::test_agent_registry_with_tools
FAILED tests/test_ui_agent_display.py::TestUIAgentDisplay::test_announcement_display_in_listen
FAILED tests/test_configuration_unit.py::TestStorageConfigValidation::test_bigquery_storage_config_valid
FAILED tests/test_structured_tool_handling.py::TestStructuredToolHandling::test_error_handling_for_missing_participants
FAILED tests/tools/test_id_generator.py::test_sexuality_iv`
  - Error: `AssertionError: ......`
- `tests/tools/test_utils.py`
  - Test: `test_b64_str_validator`
  - Error: `AssertionError: as......`
### Value Error (1 issues)
- `tests/groupchat/test_agent_announcement_behavior.py`
  - Test: `TestAgentAnnouncementBehavior::test_agent_does_not_respond_to_non_host_announcement
FAILED tests/api/test_session_persistence.py::TestSessionGCSArchival::test_archive_to_gcs_no_bm_instance
FAILED tests/groupchat/test_agent_centric_tools.py::test_registry_update_rebuilds_tools
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announces_on_cleanup
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announcement_includes_tools_and_message_types
FAILED tests/integration/20core/test_bm_session_isolation.py::TestBMInjectionSystem::test_orchestrator_bm_injection
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_handles_announcement_errors_gracefully
FAILED tests/integration/20core/test_bm_session_isolation.py::TestBMInjectionSystem::test_agent_bm_injection
FAILED tests/integration/20core/test_config_reload.py::TestConfigReloadIntegration::test_startup_script_functionality
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_calculation_in_wait_method
FAILED tests/groupchat/test_host_timeout_resilience.py::test_timeout_resilience_logging
FAILED tests/integration/20core/test_bm_session_isolation.py::TestBMInjectionSystem::test_end_to_end_bm_flow
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_step_request_routes_to_role_topic
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_flow_event_sent_before_step_request
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_manager_step_sends_ui_message_only
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_end_step_routes_to_main_topic
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_with_zero_tasks
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_unknown_role_still_routes_to_role_topic
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_route_tool_calls_uses_role_topics
FAILED tests/examples/test_logging_fail_fast_examples.py::TestValidationAndDebuggingExamples::test_example_monitoring_logging_health
FAILED tests/integration/20core/test_bm_session_isolation.py::TestBMInjectionSystem::test_session_isolation_between_runners
FAILED tests/integration/20core/test_bm_session_isolation.py::TestBackwardCompatibility::test_flowrunner_without_session_bm
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_base_agent_publish_with_topic_parameter
FAILED tests/integration/20core/test_bm_session_isolation.py::TestBackwardCompatibility::test_orchestrator_without_session_bm
FAILED tests/integration/20core/test_bm_session_isolation.py::TestBackwardCompatibility::test_agent_without_session_bm
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_with_varying_task_counts
FAILED tests/groupchat/test_host_dynamic_timeout.py::test_dynamic_timeout_respects_limits
FAILED tests/groupchat/test_host_topic_routing.py::TestHostTopicRouting::test_base_agent_publish_defaults_to_agent_topic
FAILED tests/integration/test_llm_agent_refactoring.py::TestLLMAgentRefactoring::test_llmagent_process_structured_output
FAILED tests/integration/test_llm_agent_refactoring.py::TestLLMAgentRefactoring::test_llmagent_parameter_merging
FAILED tests/integration/test_llm_agent_refactoring.py::TestLLMAgentRefactoring::test_llmagent_handles_processing_error
FAILED tests/integration/test_llm_agent_refactoring.py::TestLLMAgentRefactoring::test_llmagent_handles_unexpected_error
FAILED tests/integration/test_llm_agent_refactoring.py::TestLLMAgentRefactoring::test_llmagent_preserves_parent_trace_id
FAILED tests/integration/test_llm_agent_refactoring.py::TestLLMAgentRefactoring::test_llmagent_template_metadata_preserved
FAILED tests/integration/test_logging_fail_fast_integration.py::TestCloudLoggingIntegration::test_cloud_logging_deduplication_across_sessions
FAILED tests/examples/test_logging_fail_fast_examples.py::TestRealWorldScenarioExamples::test_example_application_startup_pattern
FAILED tests/groupchat/test_agenttrace_runinfo.py::test_agenttrace_handles_missing_bm_gracefully
FAILED tests/integration/test_agent_variable_extraction.py::TestAgentVariableExtraction::test_extract_ground_truth_from_real_fetch_output
FAILED tests/integration/20core/test_setup_infrastructure.py::test_hf_login
FAILED tests/integration/test_logging_fail_fast_integration.py::TestVerboseLoggingPreservation::test_verbose_logging_preserved_during_session_operations
FAILED tests/integration/test_logging_fail_fast_integration.py::TestErrorRecoveryAndValidation::test_logging_validation_after_partial_setup
FAILED tests/integration/test_agent_variable_extraction.py::TestAgentVariableExtraction::test_template_rendering_fails_loudly_with_real_config
FAILED tests/examples/test_logging_fail_fast_examples.py::TestRealWorldScenarioExamples::test_example_multi_session_workflow
FAILED tests/integration/00initial/test_bm_singleton.py::test_init_async_without_config_dir_uses_default
FAILED tests/integration/00initial/test_bm_singleton.py::test_init_async_with_relative_config_dir
FAILED tests/integration/00initial/test_bm_singleton.py::test_init_async_with_absolute_config_dir
FAILED tests/integration/00initial/test_bm_singleton.py::test_multiple_sessions_different_config_dirs
FAILED tests/integration/test_agent_variable_extraction.py::TestAgentVariableExtraction::test_template_rendering_succeeds_with_proper_extraction
FAILED tests/integration/test_agent_variable_extraction.py::TestAgentVariableExtraction::test_real_scorer_config_loads
FAILED tests/integration/test_agent_variable_extraction.py::TestAgentVariableExtraction::test_real_scorer_inputs_mapping
FAILED tests/integration/test_templating.py::TestMakeMessages::test_make_messages_with_record_placeholder
FAILED tests/integration/test_logging_fail_fast_integration.py::TestVerboseLoggingPreservation::test_non_verbose_logging_preserved_during_operations
FAILED tests/runner/test_batch_cli.py::TestBatchCLI::test_batch_with_max_records
FAILED tests/integration/test_logging_fail_fast_integration.py::TestErrorRecoveryAndValidation::test_verbose_level_mismatch_detection
FAILED tests/runner/test_batch_cli.py::TestBatchCLI::test_batch_with_max_jobs
FAILED tests/examples/test_logging_fail_fast_examples.py::TestRealWorldScenarioExamples::test_example_error_recovery_workflow
FAILED tests/runner/test_cli.py::test_main_console_mode`
  - Error: `ValueError: Invalid......`

## Failures by Test Category

- **root**: 14 failures
- **tools**: 7 failures
- **endtoend**: 3 failures
- **integration**: 3 failures
- **data**: 3 failures
- **runner**: 2 failures
- **api**: 1 failures
- **groupchat**: 1 failures
- **storage**: 1 failures
- **examples**: 1 failures
- **unit**: 1 failures

## Priority Fixes (Blocking Test Execution)

These issues prevent tests from even running:

1. `tests/endtoend/flows/test_embeddings.py ___________`
   - Type: import_error
   - Error: `ImportError while importing test module '/home/nic/src/buttermilk/tests/endtoend/flows/test_embeddings.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../.local/share/uv/python/cpython-3.12.0-linux-x86_64-gnu/lib/python3.12/importlib/__init__.py:90: in import_m...`
1. `tests/endtoend/test_pipeline_live.py _____________`
   - Type: import_error
   - Error: `ImportError while importing test module '/home/nic/src/buttermilk/tests/endtoend/test_pipeline_live.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../.local/share/uv/python/cpython-3.12.0-linux-x86_64-gnu/lib/python3.12/importlib/__init__.py:90: in import_modu...`
1. `tests/endtoend/test_tmdb_live.py _______________`
   - Type: import_error
   - Error: `ImportError while importing test module '/home/nic/src/buttermilk/tests/endtoend/test_tmdb_live.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../.local/share/uv/python/cpython-3.12.0-linux-x86_64-gnu/lib/python3.12/importlib/__init__.py:90: in import_module
 ...`
1. `tests/integration/test_tmdb_agent_integration.py _______`
   - Type: import_error
   - Error: `ImportError while importing test module '/home/nic/src/buttermilk/tests/integration/test_tmdb_agent_integration.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../.local/share/uv/python/cpython-3.12.0-linux-x86_64-gnu/lib/python3.12/importlib/__init__.py:90: in...`
1. `tests/test_catalog_observations.py ______________`
   - Type: import_error
   - Error: `ImportError while importing test module '/home/nic/src/buttermilk/tests/test_catalog_observations.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../.local/share/uv/python/cpython-3.12.0-linux-x86_64-gnu/lib/python3.12/importlib/__init__.py:90: in import_module...`
1. `tests/test_pipeline.py ____________________`
   - Type: import_error
   - Error: `ImportError while importing test module '/home/nic/src/buttermilk/tests/test_pipeline.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../.local/share/uv/python/cpython-3.12.0-linux-x86_64-gnu/lib/python3.12/importlib/__init__.py:90: in import_module
    return ...`
1. `tests/test_storage_record_class.py ______________`
   - Type: import_error
   - Error: `ImportError while importing test module '/home/nic/src/buttermilk/tests/test_storage_record_class.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../.local/share/uv/python/cpython-3.12.0-linux-x86_64-gnu/lib/python3.12/importlib/__init__.py:90: in import_module...`
1. `tests/test_tmdb_multiprocessor.py ______________`
   - Type: import_error
   - Error: `ImportError while importing test module '/home/nic/src/buttermilk/tests/test_tmdb_multiprocessor.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../.local/share/uv/python/cpython-3.12.0-linux-x86_64-gnu/lib/python3.12/importlib/__init__.py:90: in import_module
...`
1. `tests/tools/test_tmdb.py ___________________`
   - Type: import_error
   - Error: `ImportError while importing test module '/home/nic/src/buttermilk/tests/tools/test_tmdb.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../.local/share/uv/python/cpython-3.12.0-linux-x86_64-gnu/lib/python3.12/importlib/__init__.py:90: in import_module
    retur...`
