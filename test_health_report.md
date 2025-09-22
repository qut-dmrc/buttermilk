# Test Health Dashboard

## Summary Statistics

- **Total Test Files**: 140
- **Total Tests**: 1017
- **Passed**: 392 (38.5%)
- **Failed**: 529 (52.0%)
- **Errors**: 66
- **Skipped**: 30
- **Collection Errors**: 1

## Failures by Error Type

### Other (55 issues)
- `tests/api/test_session_persistence.py`
  - Test: `TestSessionStorageService::test_should_persist_message
FAILED tests/api/test_osb_criteria_mapping.py::test_osb_flow_with_direct_query
FAILED tests/agents/test_markdown.py::TestFramedStatementMarkdown::test_framed_statement_as_markdown
FAILED tests/agents/test_agent_name.py::test_agent_name_generation_jmespath_failure
FAILED tests/agents/test_agentoutput.py::test_actual_judge_reasons_direct_dump
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_with_media
FAILED tests/agents/test_agentoutput.py::test_actual_agent_trace_full_dump_includes_nested_outputs
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_with_text
FAILED tests/data/test_records_openaimessages.py::test_record_no_keywords`
  - Error: `A......`
- `tests/data/test_records_openaimessages.py`
  - Test: `test_as_openai_message_no_media_no_text
FAILED tests/agents/test_agentoutput.py::test_actual_agent_trace_full_dump_with_default_outputs
FAILED tests/data/test_records_openaimessages.py::test_from_path_valid`
  - Error: `Attr......`
- `tests/data/test_chromadb_sync_fix.py`
  - Test: `TestChromaDBSyncFix::test_path_preservation_logic
FAILED tests/endtoend/test_llms_multimodal.py::test_multimodal_question[sad robot local image]
FAILED tests/agents/test_fetch.py::TestFetch::test_ingest_news[abc news web]
FAILED tests/endtoend/test_llmagents.py::test_llm_agent_direct_call[gemini25flash]
FAILED tests/endtoend/test_llmagents.py::test_llm_agent_direct_call[gpt5nano]
FAILED tests/agents/test_agent_name.py::test_agent_name_generation`
  - Error: `Assertio......`
  - ... and 52 more

### Assertion Error (3 issues)
- `tests/api/test_score_endpoints.py`
  - Test: `TestDataService::test_get_record_by_id_found
FAILED tests/00initial/test_bm_singleton.py::test_conf`
  - Error: `AssertionError: BM i......`
- `tests/tools/test_uploader.py`
  - Test: `TestAsyncDataUploader::test_uploader_initialization_with_real_storage
FAILED tests/tools/test_templating.py::test_template_synth`
  - Error: `AssertionError: ......`
- `tests/tools/test_uploader.py`
  - Test: `TestAsyncDataUploader::test_uploader_handles_storage_interface
FAILED tests/tools/test_uploader.py::TestAsyncDataUploaderErrorHandling::test_uploader_handles_invalid_buffer_size
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[criteria-criteria_hatefb_factorised-9000]
FAILED tests/tools/test_utils.py::test_b64_str_validator`
  - Error: `AssertionError: as......`
### Type Error (2 issues)
- `tests/endtoend/test_text2image_apis.py`
  - Test: `test_cloud_artifact_matches_in_memory_dimensions[DALLE]
FAILED tests/endtoend/flows/test_rag_zotero_llm.py::test_rag_zotero_with_structured_output[llama4maverick]
FAILED tests/endtoend/test_text2image_apis.py::test_cloud_artifact_matches_in_memory_dimensions[SD35Large]
FAILED tests/api/test_session_persistence.py::TestSessionStorageService::test_save_message
FAILED tests/endtoend/flows/test_rag_zotero_llm.py::test_rag_zotero_with_structured_output[opus]
FAILED tests/endtoend/test_text2image_apis.py::test_cloud_artifact_matches_in_memory_dimensions[VertexImagegenModels]
FAILED tests/endtoend/test_text2image_apis.py::test_allows_none_negative_prompt_and_still_produces_image[SD35Large]
FAILED tests/endtoend/flows/test_describe.py::test_run_flow_describe_only[llama32_90b]
FAILED tests/api/test_session_persistence.py::TestSessionGCSArchival::test_archive_to_gcs_success
FAILED tests/endtoend/test_text2image_apis.py::test_cloud_artifact_matches_in_memory_dimensions[SD3]
FAILED tests/endtoend/flows/test_rag_zotero_llm.py::test_rag_zotero_with_structured_output[sonnet]
FAILED tests/endtoend/test_text2image_apis.py::test_allows_none_negative_prompt_and_still_produces_image[VertexImagegenModels]
FAILED tests/endtoend/test_text2image_apis.py::test_cloud_artifact_matches_in_memory_dimensions[SDXL]
FAILED tests/data/test_records_openaimessages.py::test_from_uri_article[abc-gaza-https://www.abc.net.au/news/2025-01-16/jewish-palestinian-australia-gaza/104825486-He said it was a "relief" to hear the news of the ceasefire "which we were calling for, for the last 15 months".]
FAILED tests/endtoend/test_text2image_apis.py::test_allows_none_negative_prompt_and_still_produces_image[SD3]
FAILED tests/endtoend/test_text2image_apis.py::test_cloud_artifact_matches_in_memory_dimensions[SDXLReplicate]
FAILED tests/endtoend/flows/test_fastapi.py::test_run_flow`
  - Error: `TypeError: Async......`
- `tests/examples/test_logging_fail_fast_examples.py`
  - Test: `TestRealWorldScenarioExamples::test_example_error_recovery_workflow
FAILED tests/endtoend/flows/test_framing.py::test_framing_video[llama32_90b]
FAILED tests/endtoend/flows/test_rag_zotero_llm.py::test_rag_zotero_with_structured_output[gemini25flash]
FAILED tests/endtoend/flows/test_rag_zotero_llm.py::test_rag_zotero_with_structured_output[gemini25pro]
FAILED tests/endtoend/flows/test_rag_zotero_llm.py::test_rag_zotero_with_structured_output[gpt5mini]
FAILED tests/examples/test_logging_fail_fast_examples.py::TestValidationAndDebuggingExamples::test_example_debugging_broken_logging
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_error_handling_example
FAILED tests/examples/test_main_script_examples.py::TestMainScriptExamples::test_main_script_scenario_1_first_session_with_project
FAILED tests/api/test_osb_websocket_lifecycle.py::TestOSBWebSocketConnection::test_osb_session_initialization
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_manual_context_management
FAILED tests/api/test_osb_websocket_lifecycle.py::TestOSBWebSocketSessionIsolation::test_session_cleanup_on_disconnect
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceEdgeCases::test_nested_traces
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announces_on_initialize
FAILED tests/examples/test_main_script_examples.py::TestMainScriptExamples::test_main_script_scenario_2_different_project_new_context
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_detects_available_tools
FAILED tests/api/test_message_service_tokens.py::TestMessageServiceTokenExtraction::test_format_message_extracts_tokens_from_agent_output
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_responds_to_host_announcement
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_does_not_respond_to_non_host_announcement
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announces_on_cleanup
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announcement_includes_tools_and_message_types
FAILED tests/examples/test_logging_fail_fast_examples.py::TestValidationAndDebuggingExamples::test_example_verbose_level_mismatch_debugging
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_handles_announcement_errors_gracefully
FAILED tests/groupchat/test_agent_announcement_behavior.py::TestAgentAnnouncementBehavior::test_agent_announcement_in_invoke_lifecycle
FAILED tests/groupchat/test_agent_centric_tools.py::test_agent_tool_definition
FAILED tests/groupchat/test_agent_centric_tools.py::test_agent_announcement
FAILED tests/endtoend/test_imagegen.py::test_batch`
  - Error: `TypeError: BatchImageGen......`
### Import Error (1 issues)
- `tests/tools/test_slackui.py _________________`
  - Error: `ImportError while importing test module '/home/nic/src/writing/projects/buttermilk/tests/tools/test_...`
### Attribute Error (1 issues)
- `tests/api/test_osb_websocket_lifecycle.py`
  - Test: `TestOSBWebSocketPerformance::test_websocket_connection_performance
FAILED tests/integration/test_osb_websocket_integration.py::TestOSBWebSocketIntegration::test_osb_websocket_message_validation
FAILED tests/integration/test_osb_websocket_integration.py::TestOSBWebSocketIntegration::test_osb_session_creation_via_websocket
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_cli_init_with_real_config_subsequent_session_inheritance
FAILED tests/integration/test_osb_websocket_integration.py::TestOSBWebSocketIntegration::test_osb_multi_agent_query_routing
FAILED tests/integration/test_logging_fail_fast_integration.py::TestErrorRecoveryAndValidation::test_verbose_level_mismatch_detection
FAILED tests/integration/test_osb_websocket_integration.py::TestOSBWebSocketIntegration::test_osb_websocket_response_streaming
FAILED tests/integration/test_osb_websocket_integration.py::TestOSBWebSocketIntegration::test_osb_session_isolation
FAILED tests/examples/test_main_script_examples.py::TestMainScriptLoggingIntegration::test_main_script_logging_includes_session_context
FAILED tests/integration/test_osb_websocket_integration.py::TestOSBWebSocketErrorHandling::test_osb_websocket_connection_recovery
FAILED tests/integration/test_osb_websocket_integration.py::TestOSBWebSocketErrorHandling::test_osb_agent_failure_handling
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_basic_usage_example
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_weave_trace_creation_and_submission
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_traceloop_missing_api_key_validation
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_agent_integration_example
FAILED tests/integration/test_unified_bootstrap_integration.py::TestUnifiedBootstrapIntegration::test_cli_init_with_real_config_project_mismatch_error
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_batch_processing_pattern
FAILED tests/integration/test_unified_tracing_config.py::TestUnifiedTracingConfig::test_otel_gcp_trace_setup
FAILED tests/examples/test_standalone_trace_examples.py::TestStandaloneTraceExamples::test_inject_parent_trace_example
FAILED tests/00initial/test_init.py::test_init`
  - Error: `AttributeError: 'BM' object ......`
### Value Error (1 issues)
- `tests/storage/test_file_storage_column_mapping.py`
  - Test: `TestFileStorageRegressionTests::test_tja_config_compatibility
FAILED tests/storage/test_file_storage_column_mapping.py::TestFileStorageColumnMapping::test_simple_column_mapping
FAILED tests/test_bm_async_init.py::TestBMAsyncInitialization::test_initialization_completes_before_use
FAILED tests/storage/test_file_storage_column_mapping.py::TestFileStorageIntegration::test_jsonl_format_with_column_mapping
FAILED tests/endtoend/test_llmagents.py::test_llm_agent_template_metadata[haiku]
FAILED tests/runner/test_cli.py::test_main_console_mode`
  - Error: `ValueError: Invalid......`

## Failures by Test Category

- **root**: 25 failures
- **tools**: 8 failures
- **api**: 6 failures
- **integration**: 6 failures
- **unit**: 4 failures
- **examples**: 3 failures
- **data**: 2 failures
- **agents**: 2 failures
- **initial**: 2 failures
- **groupchat**: 2 failures
- **storage**: 2 failures
- **validation**: 1 failures

## Priority Fixes (Blocking Test Execution)

These issues prevent tests from even running:

1. `tests/tools/test_slackui.py _________________`
   - Type: import_error
   - Error: `ImportError while importing test module '/home/nic/src/writing/projects/buttermilk/tests/tools/test_slackui.py'.
Hint: make sure your test modules/pac...`