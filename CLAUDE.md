# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Buttermilk Project Development Guidelines
Primary Goal: Build reproducible, traceable, and HASS-researcher-friendly tools.

* HASS-Centric Design: Prioritize usability, clarity, extensibility, and accessibility for Humanities, Arts, and Social Sciences (HASS) researchers.
* Reproducibility & Traceability: Ensure that experiments, data processing, and results are reproducible and traceable. Design for robust logging and versioning.
* Modularity: Maintain a modular architecture. Core components are Agent, Orchestrator, and Contract. Prefer creating new Agent/Orchestrator subclasses over modifying core components.
* Fewer classes the better for later extensibility: prioritise using the same fundamental base classes throughout the codebase. For example, use AgentTrace or AgentOutputs throughout, rather than re-creating separate objects for specific purposes, like an object specifically for a particular frontend use case (within reason, obviously!)
* Store data in Buttermilk base classes: For example, AgentTrace is designed to contain all the information that is needed in addition to the AgentOutput data, in order to provide a reliable and fully observable trace. This means that AgentTrace objects can be easily re-created from BigQuery / GCS storage, and we should avoid excess conversion. Note that there are some data sources that are SQL views of the underlying Buttermilk object tables -- e.g. judge_scores is defined in judge_scores.sql and is a view that includes the FLOWS_TABLE (which contains AgentTrace objects directly). Don't hard-code SQL, but you can create new views if required.
* Asynchronous Operations: Embrace async/await for I/O-bound tasks, LLM calls, and concurrent operations to ensure responsiveness.
* Pay down tech debt: The project is still in "very early stages". There is left-over code that does not align with these principles or the direction of the project. You should seek confirmation but not hesitate to suggest removing or refactoring code that creates unecessary complexity and maintenance overhead.
* You might find a lot of linting errors. If you've been asked to complete a task, you should only fix the critical errors or problems you introduced. Don't go aruond looking for other stuff to do, although you should make a note of problems you noticed but didn't fix in your final report.
* Never change the general agent initialisation interface just to fix one agent
* Don't write disposable test scripts; write pytest unittests instead in /tests
* Don't add backwards compatibility when making changes. Make it work for our codebase, and don't support outdated approaches.
* Add a git commit after every logical chunk of changes. You don't need to ask permission to commit changes.

## TODO: Future Architectural Improvements

### Multimodal Content Handling
The current `Record.content: str | Sequence[str | Image]` design needs improvement for better multimodal support:

**Current Issues:**
- Mixed types in sequences are hard to query/filter
- Images stored as base64 in JSON bloat the database  
- No clear separation between "primary content" and "derived content"
- Unclear when to use binary vs caption vs both

**Design Options to Consider:**
1. **Separate Fields**: Split into `content: str`, `images: list[str]` (URIs), `image_captions: list[str]`
2. **Structured Content**: Use `ContentPart` objects with type + data + metadata
3. **Hybrid Storage**: Text in BigQuery, images in GCS with URI references

**Use Cases:**
- Binary-focused: Image generation, computer vision, OCR
- Caption-focused: Text analysis, search, semantic similarity
- Both: Multimodal LLMs, accessibility, content moderation

*Decision deferred - current base64 serialization works for now*

## Key Codebase Architecture Notes

### Configuration System
- **DataSourceConfig vs StorageConfig**: Two separate config models exist. DataSourceConfig has more fields (glob, agg, group, etc.). StorageConfig is used in newer storage system. Both support `columns` field for field mapping.
- **OmegaConf to Pydantic**: Raw configs from YAML files are OmegaConf objects. Must convert to proper Pydantic models using `DataSourceConfig(**dict(config))` before passing to data loaders.
- **columns field**: Maps source field names to Record field names. Can be empty `{}`. Pattern: `{target_record_field: source_data_field}`.

### Data Loading & Storage
- **create_data_loader()**: Expects `DataSourceConfig`, not raw config. All loaders support column mapping except deprecated ones.
- **Flow storage access**: `flow_runner.flows[flow_name].storage[dataset_name]` - flows can have multiple datasets
- **BM instance BigQuery**: Use `BigQueryDefaults().dataset_id` for default dataset, NOT `bm_instance.save.dataset_id` (save is a function)

### Frontend API Structure  
- **Frontend API routes**: Proxy to backend with dataset parameter support: `/api/records?flow=X&dataset=Y`
- **Flow/dataset separation**: Frontend has separate dropdowns for flow → dataset → records selection
- **Score page dataset handling**: Remove hardcoded 'tox' defaults, use URL params `flow` and `dataset`

### Storage Access Patterns
```python
# CORRECT - Convert raw config to DataSourceConfig
storage_config = DataService._convert_to_data_source_config(raw_config)
loader = create_data_loader(storage_config)

# CORRECT - Access dataset ID
from buttermilk._core.storage_config import BigQueryDefaults  
dataset_id = BigQueryDefaults().dataset_id

# WRONG - Don't use raw OmegaConf directly
loader = create_data_loader(raw_omegaconf_config)  # Fails validation

# WRONG - Don't access save as attribute  
dataset_id = bm_instance.save.dataset_id  # save is a function
```

## Development Commands

- **Install dependencies**: `uv install` (installs main + ml + dev groups by default)
- **Python command**: `uv run python` (all Python commands should use this prefix)
- **Run all tests**: `uv run python -m pytest`
- **Run single test**: `uv run python -m pytest tests/path/to/test_file.py::test_function`
- **Integration tests**: Tests marked with `@pytest.mark.integration` are excluded by default
- **Lint**: `uv run python -m ruff buttermilk`
- **Format**: `uv run python -m black buttermilk`
- **Type checking**: Uses mypy (configured in pyproject.toml)
- **Main CLI**: `uv run python buttermilk/runner/cli.py`

## The BM singleton

- All configuration is stored in composable YAML files in `conf/` and managed by Hydra.
- At run time, a main BM singleton instance is created. This holds project-level configuration and provides a standard interface for MLOps tasks.

## Configuration Hints

* To see the actual hydra config in use, use `-c job`, like: `uv run python -m buttermilk.runner.cli -c job +flow=tox +run=batch "+flows=[tox]" "run.mode=batch_run"`
* Use `uv run python -m buttermilk.runner.cli -c job` to check configurations.

## Execution Path (API Default)

1. **CLI Entry**: `cli.py` @hydra.main loads config from `conf/`
2. **API Mode**: Creates FastAPI app via `create_fastapi_app(bm, flow_runner)`
3. **FlowRunner**: Receives a RunRequest and then instantiates orchestrator using `OrchestratorFactory.create_orchestrator()`
4. **AutogenOrchestrator**: Manages agent lifecycle and internal pub/sub communication via `SingleThreadedAgentRuntime`
5. **Session Management**: `SessionManager` handles WebSocket connections and cleanup for frontend clients


## Flow Configurations

### Trans Flow (trans.yaml)
```yaml
orchestrator: buttermilk.orchestrators.groupchat.AutogenOrchestrator
data: tja_train (journalism training dataset)
agents: [judge, synth, differences]
observers: [spy, owl, scorer, fetch, host/sequencer]
criteria: trans (advocacy/journalism quality criteria)
```

### Tox Flow (tox.yaml)  
```yaml
orchestrator: buttermilk.orchestrators.groupchat.AutogenOrchestrator
data: drag (toxicity dataset)
agents: [judge, synth, differences] 
observers: [owl, scorer, host/sequencer]
criteria: hate (multiple toxicity criteria)
```
## Agent Class Hierarchy

```
Agent (ABC) <- AgentConfig
├── LLM-based agents (judge, synth, differences)
├── UIAgent <- Agent
│   ├── CLIUserAgent (console interaction)
│   └── WebSocketAgent (API interaction)
├── FlowControl agents
│   ├── HostAgent (conductor/sequencer role)
│   ├── ExplorerAgent (adaptive flow control)
│   └── LLMHostAgent (LLM-driven conductor)
└── Observer agents (spy, owl, scorer, fetch)
```

## Host/UI Options

### Host Agents (conductor role)
- **host/sequencer**: Basic sequential host (default)
- **host/explorer**: Adaptive exploration host  
- **host/assistant**: LLM-driven assistant host

### UI Options
- **console**: CLIUserAgent for terminal interaction
- **web**: WebSocket-based for browser/API clients
- **batch**: No UI for automated processing

## Component Flow

1. **Orchestrator Setup**: `AutogenOrchestrator._setup()` registers agents with `SingleThreadedAgentRuntime`
2. **Agent Registration**: Each agent becomes `AutogenAgentAdapter` wrapping Buttermilk `Agent`
3. **Topic Subscription**: Agents subscribe to main topic (`{bm.name}-{job}-{uuid}`) and role topics
4. **Message Flow**: Host agent sends `ConductorRequest` and `AgentInput`; other agents respond via internal pub/sub (managed by Autogen)
5. **Session Cleanup**: `SessionManager` handles resource cleanup and timeout

### Agent-based data processing

- Agent is the key abstraction for processing data
- Agents are strictly _independent_ from flows: mix and match as needed.
- Agents are stateful and can be configured to save data sent over internal Autogen pub/sub groupchat
- Agents are explicitly invoked with a `AgentInput` object that provides parameters and data, often including a `Record` item
- Subclasses of Agent must implement a `_process()` method that expects AgentInput and returns AgentOutput
- The invoke method is the normal interface point; it accepts an `AgentInput` and returns an `AgentTrace` that includes the `AgentOutput` and additional tracing information for full observability and logging. 
- `Observer` Agents are not invoked directly, but have their own logic that is usually triggered in their `_listen` events.

## Development hints
* When dumping pydantic objects, use pydantic's model_dump()
* Add 'pytestmark = pytest.mark.anyio' near the top of test modules