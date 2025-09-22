# Test Health Dashboard

## Summary Statistics

- **Total Test Files**: 140
- **Total Tests**: 1027
- **Passed**: 382 (37.2%)
- **Failed**: 522 (50.8%)
- **Errors**: 92
- **Skipped**: 31
- **Collection Errors**: 0

## Failures by Error Type

### Other (59 issues)
- `tests/data/test_chromadb_sync_fix.py`
  - Test: `TestChromaDBSyncFix::test_path_preservation_logic
FAILED tests/agents/test_markdown.py::TestFramedStatementMarkdown::test_framed_statement_as_markdown
FAILED tests/agents/test_agentoutput.py::test_actual_agent_trace_full_dump_with_default_outputs
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_with_text
FAILED tests/data/test_records_openaimessages.py::test_as_openai_message_no_media_no_text
FAILED tests/api/test_osb_criteria_mapping.py::test_osb_flow_with_direct_query
FAILED tests/endtoend/test_imagegen.py::test_model[SD35Large-stereotype]`
  - Error: `Ru......`
- `tests/endtoend/test_imagegen.py`
  - Test: `test_model[VertexImagegenModels-cat]
FAILED tests/data/test_records_openaimessages.py::test_from_path_valid`
  - Error: `Attr......`
- `tests/endtoend/test_imagegen.py`
  - Test: `test_model[VertexImagegenModels-stereotype]
FAILED tests/endtoend/test_imagegen.py::test_model[SD3-cat]`
  - Error: `RuntimeError: B......`
  - ... and 56 more

### Assertion Error (3 issues)
- `tests/api/test_osb_websocket_lifecycle.py`
  - Test: `TestOSBWebSocketSessionIsolation::test_session_data_isolation
FAILED tests/api/test_score_endpoints.py::TestDataService::test_get_record_by_id_found
FAILED tests/endtoend/flows/test_rag_zotero_llm.py::test_rag_zotero_with_structured_output[gpt5mini]
FAILED tests/endtoend/flows/test_framing.py::test_frames_text[lady macbeth-gpt5nano]
FAILED tests/api/test_session_persistence.py::TestSessionGCSArchival::test_archive_to_gcs_success
FAILED tests/00initial/test_bm_singleton.py::test_conf`
  - Error: `AssertionError: BM i......`
- `tests/tools/test_uploader.py`
  - Test: `TestAsyncDataUploader::test_uploader_initialization_with_real_storage
FAILED tests/tools/test_uploader.py::TestAsyncDataUploader::test_uploader_handles_storage_interface
FAILED tests/tools/test_uploader.py::TestAsyncDataUploaderErrorHandling::test_uploader_handles_invalid_buffer_size
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[criteria-criteria_hatefb_factorised-9000]
FAILED tests/tools/test_utils.py::test_b64_str_validator`
  - Error: `AssertionError: as......`
- `tests/tools/test_utils.py`
  - Test: `test_get_templates_default_pattern[criteria-criteria_ordinary-5240]
FAILED tests/tools/test_utils.py::test_get_templates_default_pattern[synth-synthesise-1000]
FAILED tests/tools/test_varmap.py::TestFlowVariableRouter::test_simple_path_resolution
FAILED tests/tools/test_templating.py::test_template_synth`
  - Error: `AssertionError: ......`
### Type Error (2 issues)
- `tests/endtoend/test_imagegen.py`
  - Test: `test_batch`
  - Error: `TypeError: BatchImageGen......`
- `tests/endtoend/test_text2image_apis.py`
  - Test: `test_generate_in_parallel_all_clients
FAILED tests/endtoend/flows/test_describe.py::test_run_flow_describe_only[llama32_90b]
FAILED tests/endtoend/test_text2image_apis.py::test_generated_image_is_valid_and_nontrivial[SD3]
FAILED tests/endtoend/test_text2image_apis.py::test_cloud_artifact_content_hash_is_stable_for_single_download[DALLE]
FAILED tests/endtoend/test_text2image_apis.py::test_cloud_artifact_content_hash_is_stable_for_single_download[SD35Large]
FAILED tests/endtoend/test_text2image_apis.py::test_cloud_artifact_content_hash_is_stable_for_single_download[VertexImagegenModels]
FAILED tests/endtoend/flows/test_fastapi.py::test_run_flow`
  - Error: `TypeError: Async......`
### Value Error (1 issues)
- `tests/tools/test_varmap.py`
  - Test: `TestFlowVariableRouterSpecialCases::test_empty_lists
FAILED tests/unit/test_bm_weave_client_delegation.py::TestBMWeaveClientDelegation::test_delegation_to_execution_context_success
FAILED tests/unit/test_bm_weave_client_delegation.py::TestBMWeaveClientDelegation::test_execution_context_get_weave_client_failure_propagates
FAILED tests/unit/test_bm_weave_client_delegation.py::TestBMWeaveClientDelegation::test_both_paths_return_valid_weave_client_interface
FAILED tests/unit/test_bm_weave_client_delegation.py::TestBMWeaveClientDelegation::test_import_isolation_in_fallback
FAILED tests/unit/test_bm_weave_client_delegation.py::TestBMWeaveClientBehaviorDocumentation::test_documented_delegation_behavior
FAILED tests/unit/test_bm_weave_client_delegation.py::TestBMWeaveClientBehaviorDocumentation::test_documented_fallback_behavior
FAILED tests/unit/test_bm_weave_client_delegation.py::TestBMWeaveClientBehaviorDocumentation::test_proper_weave_init_vs_fallback_documentation
FAILED tests/endtoend/test_text2image_apis.py::test_generated_image_is_valid_and_nontrivial[DALLE]
FAILED tests/unit/test_citator.py::test_citator_initialization_with_defaults
FAILED tests/integration/test_zotero_vector_integration.py::TestZoteroVectorIntegration::test_record_creation_from_zotero
FAILED tests/runner/test_cli.py::test_main_console_mode`
  - Error: `ValueError: Invalid......`

## Failures by Test Category

- **root**: 26 failures
- **api**: 10 failures
- **tools**: 9 failures
- **runner**: 3 failures
- **unit**: 3 failures
- **data**: 2 failures
- **agents**: 2 failures
- **initial**: 2 failures
- **examples**: 2 failures
- **groupchat**: 2 failures
- **integration**: 1 failures
- **storage**: 1 failures
- **utils**: 1 failures
- **validation**: 1 failures

## Priority Fixes (Blocking Test Execution)

No critical blocking issues found.
