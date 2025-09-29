# Test Health Dashboard

## Summary Statistics

- **Total Test Files**: 152
- **Total Tests**: 1072
- **Passed**: 458 (42.7%)
- **Failed**: 507 (47.3%)
- **Errors**: 77
- **Skipped**: 30
- **Collection Errors**: 1

## Failures by Error Type

### Other (51 issues)
- `tests/agents/test_agent_name.py`
  - Test: `test_agent_name_generation_jmespath_failure
FAILED tests/agents/test_markdown.py::TestFramedStatementMarkdown::test_framed_statement_as_markdown
FAILED tests/00initial/test_bm_singleton.py::test_singleton_instance`
  - Error: `Assert......`
- `tests/api/test_session_persistence.py`
  - Test: `TestSessionStorageService::test_session_exists
FAILED tests/00initial/test_bm_singleton.py::test_session_scoped_instances`
  - Error: `......`
- `tests/agents/test_agentoutput.py`
  - Test: `test_actual_judge_reasons_direct_dump
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_should_persist_message
FAILED tests/00initial/test_bm_singleton.py::test_singleton_between_modules
FAILED tests/agents/test_agentoutput.py::test_actual_agent_trace_full_dump_includes_nested_outputs
FAILED tests/data/test_chromadb_sync_fix.py::TestChromaDBSyncFix::test_path_preservation_logic
FAILED tests/agents/test_agentoutput.py::test_actual_agent_trace_full_dump_with_default_outputs
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_with_media
FAILED tests/endtoend/flows/test_describe.py::test_run_flow_describe_only[gpt41]
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_corrupted_session_file_handling
FAILED tests/endtoend/flows/test_rag_zotero_llm.py::test_rag_zotero_with_structured_output[llama4maverick]
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_with_text
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_record_by_id_found
FAILED tests/agents/test_fetch.py::TestFetch::test_load_data`
  - Error: `AttributeError......`
  - ... and 48 more

### Assertion Error (3 issues)
- `tests/api/test_session_persistence.py`
  - Test: `TestSessionStorageService::test_get_session_messages
FAILED tests/00initial/test_bm_singleton.py::test_conf`
  - Error: `AssertionError: BM i......`
- `tests/test_storage_config_unit.py`
  - Test: `test_storage_config_full_table_id
FAILED tests/tools/test_id_generator.py::test_sexuality_iv`
  - Error: `AssertionError: ......`
- `tests/tools/test_tmdb.py`
  - Test: `TestTMDBGetAvailability::test_get_availability_no_providers_returns_null_observations
FAILED tests/test_configuration_unit.py::TestStorageConfigValidation::test_storage_config_merge_defaults
FAILED tests/tools/test_tmdb.py::TestTMDBGetAvailability::test_get_availability_api_error_handling
FAILED tests/test_configuration_unit.py::TestStorageConfigValidation::test_storage_config_serialization
FAILED tests/endtoend/test_text2image_apis.py::test_generated_image_is_valid_and_nontrivial[SDXLReplicate]
FAILED tests/test_lazy_loading_unit.py::TestCloudManagerLazyLoading::test_cloud_manager_gcp_credentials_lazy
FAILED tests/test_lazy_loading_unit.py::TestCloudManagerLazyLoading::test_cloud_manager_clients_are_lazy
FAILED tests/test_lazy_loading_unit.py::TestCloudManagerLazyLoading::test_cloud_manager_credentials_cached
FAILED tests/tools/test_tmdb.py::TestTMDBGetAvailability::test_get_availability_with_providers
FAILED tests/tools/test_uploader.py::TestAsyncDataUploader::test_uploader_initialization_with_real_storage
FAILED tests/tools/test_uploader.py::TestAsyncDataUploader::test_uploader_handles_storage_interface
FAILED tests/tools/test_uploader.py::TestAsyncDataUploaderErrorHandling::test_uploader_handles_invalid_buffer_size
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[criteria-criteria_ordinary-5240]
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[criteria-criteria_hatefb_factorised-9000]
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[synth-synthesise-1000]
FAILED tests/tools/test_utils.py::test_b64_str_validator`
  - Error: `AssertionError: as......`
### Type Error (2 issues)
- `tests/endtoend/flows/test_describe.py`
  - Test: `test_run_flow_describe_only[llama32_90b]
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_no_media_no_text
FAILED tests/endtoend/flows/test_fastapi.py::test_run_flow`
  - Error: `TypeError: Async......`
- `tests/endtoend/test_imagegen.py`
  - Test: `test_model[SDXLReplicate-stereotype]
FAILED tests/endtoend/test_imagegen.py::test_batch`
  - Error: `TypeError: BatchImageGen......`
### Attribute Error (2 issues)
- `tests/examples/test_logging_fail_fast_examples.py`
  - Test: `TestValidationAndDebuggingExamples::test_example_debugging_broken_logging
FAILED tests/endtoend/flows/test_framing.py::test_framing_video[llama32_90b]
FAILED tests/examples/test_logging_fail_fast_examples.py::TestValidationAndDebuggingExamples::test_example_verbose_level_mismatch_debugging
FAILED tests/endtoend/flows/test_describe.py::test_run_flow_describe_only[gemini25pro]
FAILED tests/endtoend/flows/test_describe.py::test_run_flow_describe_only[llama4maverick]
FAILED tests/endtoend/flows/test_describe.py::test_run_flow_describe_only[gemini25flash]
FAILED tests/00initial/test_init.py::test_short_form_nb`
  - Error: `AttributeError: 'BM......`
- `tests/examples/test_logging_fail_fast_examples.py`
  - Test: `TestValidationAndDebuggingExamples::test_example_monitoring_logging_health
FAILED tests/00initial/test_init.py::test_init`
  - Error: `AttributeError: 'BM' object ......`
### Value Error (2 issues)
- `tests/integration/test_tmdb_agent_integration.py`
  - Test: `TestTMDBAgentIntegration::test_tool_error_handling_for_agents
FAILED tests/endtoend/test_text2image_apis.py::test_cloud_artifact_matches_in_memory_dimensions[SD35Large]
FAILED tests/endtoend/test_text2image_apis.py::test_generated_image_is_valid_and_nontrivial[DALLE]
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_cli_init_with_real_config_first_session
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_cli_init_with_real_config_subsequent_session_inheritance
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_cli_init_with_real_config_project_mismatch_error
FAILED tests/endtoend/test_llm_tool_calling.py::test_structured_output_without_tools[sonnet]
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_bootstrap_session_function_with_real_config
FAILED tests/endtoend/flows/test_rag_zotero_llm.py::test_rag_zotero_llama4_specific
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_bootstrap_session_with_config_function_with_real_config
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_cli_init_with_overrides_real_config
FAILED tests/endtoend/test_llms.py::TestPromptStyles::test_usertext_and_placeholder[osb blm-gemini25flash]
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_real_config_error_handling_missing_project
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_session_isolation_with_real_config
FAILED tests/integration/test_unified_bootstrap_integration.py::TestMainScriptRealExecution::test_main_script_scenario_with_real_config
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_weave_config_based_initialization
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_weave_missing_project_id_fails_fast
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_weave_missing_api_key_fails_fast
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_weave_trace_creation_and_submission
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_traceloop_missing_api_key_validation
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_otel_gcp_trace_setup
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_multiple_providers_configuration
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_disabled_providers_not_initialized
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_execution_context_fix_for_nonetype_error
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_config_based_credentials_override_environment
FAILED tests/integration/test_unified_tracing_config.py::TestTracingConfigExamples::test_minimal_weave_setup_example
FAILED tests/integration/test_unified_tracing_config.py::TestTracingConfigExamples::test_production_multi_provider_example
FAILED tests/integration/test_unified_tracing_config.py::TestTracingConfigExamples::test_development_weave_only_example
FAILED tests/integration/test_zotero_vector_integration.py::TestZoteroVectorIntegration::test_record_creation_from_zotero
FAILED tests/integration/test_zotero_vector_integration.py::TestZoteroVectorIntegration::test_pdf_download
FAILED tests/integration/test_zotero_vector_integration.py::TestZoteroVectorIntegration::test_fulltext_extraction
FAILED tests/endtoend/test_text2image_apis.py::test_generated_image_is_valid_and_nontrivial[SD35Large]
FAILED tests/integration/test_zotero_vector_integration.py::TestZoteroVectorIntegration::test_annotation_extraction_not_implemented
FAILED tests/endtoend/test_llms.py::TestPromptStyles::test_words_in_mouth[gemini25flash]
FAILED tests/integration/test_zotero_vector_integration.py::TestZoteroVectorIntegration::test_last_modified_version_handling
FAILED tests/runner/test_cli.py::test_main_console_mode`
  - Error: `ValueError: Invalid......`
- `tests/endtoend/test_text2image_apis.py`
  - Test: `test_cloud_artifact_matches_in_memory_dimensions[SD3]
FAILED tests/endtoend/test_text2image_apis.py::test_cloud_artifact_matches_in_memory_dimensions[SDXL]
FAILED tests/endtoend/test_llms.py::TestPromptStyles::test_usertext_and_placeholder[lady macbeth-haiku]
FAILED tests/endtoend/test_text2image_apis.py::test_allows_none_negative_prompt_and_still_produces_image[SDXLReplicate]
FAILED tests/test_pipeline.py::test_pipeline_tmdb_simple`
  - Error: `ValueError: Unsupp......`
### Validation Error (1 issues)
- `tests/utils/test_templating.py ________________`
  - Error: `import file mismatch:
imported module 'test_templating' has this __file__ attribute:
  /home/nic/src/writing/projects/buttermilk/tests/tools/test_templating.py
which is not the same as the test file we want to collect:
  /home/nic/src/writing/projects/buttermilk/tests/utils/test_templating.py
HINT: ...`

## Failures by Test Category

- **endtoend**: 16 failures
- **root**: 10 failures
- **examples**: 6 failures
- **data**: 6 failures
- **tools**: 5 failures
- **api**: 4 failures
- **agents**: 3 failures
- **runner**: 3 failures
- **utils**: 2 failures
- **unit**: 2 failures
- **groupchat**: 1 failures
- **integration**: 1 failures
- **storage**: 1 failures
- **validation**: 1 failures

## Priority Fixes (Blocking Test Execution)

No critical blocking issues found.
