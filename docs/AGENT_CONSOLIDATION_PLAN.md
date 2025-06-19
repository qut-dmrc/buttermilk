# Agent and Flow Consolidation Plan

## Current State Analysis

### Flows (12 total)
1. **Core flows**: osb, tox, trans (keep these)
2. **Test/demo flows**: simple, test, enhanced_rag_demo, z (remove/consolidate)
3. **Duplicate flows**: tox_allinone (merge into tox)
4. **Specialized flows**: judger, ingest (evaluate for removal)
5. **Criteria flows**: criteria/hate, criteria/trans (merge into parent flows)

### Host Agents (5 configurations)
1. **sequencer.yaml** - Simple sequence-based execution
2. **assistant.yaml** - LLM-based with panel_host template
3. **explorer.yaml** - Specialized for OSB exploration
4. **selector.yaml** - Appears unused
5. **ra.yaml** - Research assistant variant

### Templates (50+ jinja2 files)
Many near-duplicate templates with minor variations

## Consolidation Strategy

### 1. Flows: Reduce to 3 Core Flows

#### osb.yaml (Keep as-is)
- Purpose: Interactive OSB vector store queries
- Uses enhanced RAG agents
- Keep the comprehensive configuration

#### tox.yaml (Enhance)
```yaml
defaults:
  - _self_
  - /base/common_agents  # Common judge, synth, differences
  - /base/common_observers  # spy, owl, scorer
  - /agents/host@tox.observers.host: sequencer  # Simple sequence host
  - /storage@tox.storage: ${storage.toxicity}  # Configurable storage
  - /criteria@tox.parameters.criteria: toxicity  # All toxicity criteria

tox:
  orchestrator: buttermilk.orchestrators.groupchat.AutogenOrchestrator
  name: "toxicity_assessment"
  description: "Comprehensive toxicity assessment with multiple criteria"
  parameters:
    human_in_loop: ${run.human_in_loop}
    # Criteria can be overridden at runtime
```

#### trans.yaml (Enhance)
```yaml
defaults:
  - _self_
  - /base/common_agents
  - /base/common_observers
  - /agents/host@trans.observers.host: sequencer
  - /storage@trans.storage: ${storage.journalism}
  - /criteria@trans.parameters.criteria: trans

trans:
  orchestrator: buttermilk.orchestrators.groupchat.AutogenOrchestrator
  name: "journalism_quality"
  description: "Journalism quality assessment for trans issues"
  parameters:
    human_in_loop: ${run.human_in_loop}
```

### 2. Host Agents: Reduce to 2 Types

#### sequence_host.yaml (Rename from sequencer.yaml)
```yaml
# Simple sequence-based host that executes steps in order
sequence_host:
  role: HOST
  name: "Sequence Coordinator"
  agent_obj: HostAgent
  parameters:
    human_in_loop: ${run.human_in_loop}
    execution_mode: "sequential"  # Execute agents in defined order
```

#### llm_host.yaml (Merge assistant.yaml and explorer.yaml)
```yaml
# LLM-based host that delegates using structured tools
llm_host:
  role: HOST
  name: "Intelligent Coordinator"
  agent_obj: StructuredLLMHostAgent  # Use the new structured version
  parameters:
    model: ${llms.host}
    human_in_loop: ${run.human_in_loop}
    execution_mode: "adaptive"  # LLM decides execution flow
    template: ${host.template:host_structured_tools}  # Configurable template
```

### 3. Template Consolidation

#### Remove Duplicates
- Merge similar judge templates into one configurable template
- Consolidate RAG templates (rag_academic, rag_owl) into single rag.jinja2
- Remove unused templates (z.jinja2, test-specific templates)

#### Create Base Templates
```
templates/
├── base/
│   ├── judge.jinja2         # Configurable judging template
│   ├── synthesis.jinja2     # Synthesis with configurable focus
│   └── analysis.jinja2      # General analysis template
├── host/
│   ├── sequence.jinja2      # For sequence host
│   └── structured_tools.jinja2  # For LLM host
└── criteria/
    ├── toxicity.jinja2      # All toxicity criteria
    └── journalism.jinja2    # All journalism criteria
```

### 4. Common Components

#### /base/common_agents.yaml
```yaml
judge:
  _target_: /agents/judge
  parameters:
    template: ${templates.judge}
    
synth:
  _target_: /agents/synth
  parameters:
    template: ${templates.synthesis}
    
differences:
  _target_: /agents/differences
```

#### /base/common_observers.yaml
```yaml
spy:
  _target_: /agents/spy
  
owl:
  _target_: /agents/owl
  
scorer:
  _target_: /agents/scorer
```

## Implementation Steps

1. **Create base configurations** in `/conf/base/`
2. **Update the 3 core flows** to use composable configs
3. **Consolidate host agents** to 2 types
4. **Merge duplicate templates** with parameterization
5. **Remove obsolete flows and configs**
6. **Update documentation** with new structure

## Benefits

1. **Simplicity**: From 12 flows to 3, from 5 hosts to 2
2. **Composability**: Shared components via Hydra composition
3. **Configurability**: Override any component at runtime
4. **Maintainability**: Less duplication, clearer structure
5. **Extensibility**: Easy to add new flows by composing existing parts

## Configuration Example

```bash
# Run tox with custom criteria
uv run python -m buttermilk.runner flows/tox criteria=my_criteria

# Run trans with different storage
uv run python -m buttermilk.runner flows/trans storage.journalism=alt_storage

# Run osb with different host
uv run python -m buttermilk.runner flows/osb observers.host=llm_host
```