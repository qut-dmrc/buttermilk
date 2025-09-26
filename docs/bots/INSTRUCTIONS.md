<!-- This file should be read EVERY time. Keep it CONCISE and LIMITED to strictly required information. -->
# Buttermilk Project Instructions

This document provides the essential, project-specific instructions for agents working on the Buttermilk project. 

## Core Mission

Buttermilk provides AI and data tools for HASS (Humanities, Arts, and Social Sciences) researchers. The core mission is to make computational methods accessible while maintaining research rigor and reproducibility. Buttermilk aims to make it easy for HASS scholars to use AI tools in a way that is understandable, traceable, and reproducible.

## Key Project-Specific Information

*   **Primary Interface**: The `automod.cc` website, which presents research flows as an IRC-style group chat.
*   **Core Evaluation Pattern**: A `JUDGE` -> `SYNTH` groupchat pattern is used to evaluate how well different AI models can apply complex, human-readable guidelines to a piece of text.
*   **Data Workflow**: Raw data is in Google Cloud Storage (GCS), and results are stored in BigQuery as `ExecutionTrace` records.
*   **Configuration**: The project is heavily reliant on Hydra and YAML configuration files located in the `conf/` directory. NEVER use manual dictionary configuration.

## Development & Debugging

For Buttermilk-specific debugging, see `debugging.md`.

For information on the technology stack, see `techstack.md`.

For information on log analysis, see `logs.md`.

For information on data architecture, see `data-architecture.md`.

### Valid System Configuration Parameters

**Available Flows**: 
- `trans` - Transgender journalist ethics research flow
- `transllm` - LLM-based trans journalism analysis 
- `zot` - Zotero integration flow
- `osb` - Online Safety Benchmark flow

**Valid Criteria Templates**:
- `tja` - Trans Journalists Association stylebook criteria
- `glaad` - GLAAD media reference criteria

**Record ID Requirements**:
- **MUST** use actual record IDs from your data sources
- **NEVER** use placeholder values like 'demo_record', 'demo', 'test_record'

## Buttermilk-Specific Agents

*   **Debug Pipeline Manager (`AGENT-DEBUGGER.md`)**: An expert systems engineer for live debugging and validation. Its role is diagnostic only.
*   **Test Fixer Agent (`TEST_FIXER_AGENT.md`)**: A specialized agent for fixing broken tests using `ruff`.