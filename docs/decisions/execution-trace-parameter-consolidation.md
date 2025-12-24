---
title: ExecutionTrace Parameter Consolidation
type: decision
tags: [tracing, architecture, refactoring]
created: 2025-12-24
status: implemented
---

# ExecutionTrace Parameter Consolidation

## Decision

Consolidated trace parameter storage to eliminate duplication between three locations.

## Before

Three places stored parameters with overlapping content:

| Location | Content |
|----------|---------|
| `ExecutionTrace.parameters` | Processor config (model, template, criteria) |
| `ExecutionTrace.agent_info["parameters"]` | Full agent.parameters dict |
| `ExecutionTrace.inputs` | Runtime template variables |

This caused confusion about where to look for reproducibility data.

## After

Two distinct purposes:

| Location | Content | Purpose |
|----------|---------|---------|
| `trace.inputs` | Template variables (criteria, instructions, record fields) | Everything that affects template output |
| `trace.agent_info` | Static agent config (model, template, template_hash) | Agent identity |

## Changes Made

1. **llm_core.py:382-387** - expanded `resolved_inputs` to include `self.template_vars` (config template vars like criteria, instructions)
2. **contract.py** - removed `ExecutionTrace.parameters` field
3. **traces.schema.json** - removed `parameters` from BigQuery schema
4. **processor_core.py** - removed `parameters=` argument from `_emit_success_trace` and `_emit_error_trace`
5. **agent.py:create_agent_trace_info** - removed `"parameters": agent.parameters` from agent_info dict
6. **llm_core.py, classifier.py** - updated callers to remove `parameters=` argument
7. **Tests** - updated `test_execution_trace.py`, `test_llmcore_tracing.py`, `test_flow_e2e_minimal.py`

## Rationale

Single source of truth for reproducibility:
- `trace.inputs` contains everything that affects template output
- `agent_info` identifies the agent (model, template, hash)
- No duplication between fields

## Breaking Change

This removes the `parameters` field from `ExecutionTrace`. Any code accessing `trace.parameters` must change to `trace.inputs`.
