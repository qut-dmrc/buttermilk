# Buttermilk Architecture Documentation

## Project Overview

Buttermilk is a framework for building reproducible, traceable, and HASS-researcher-friendly multi-agent systems. It provides infrastructure for orchestrating agents, managing data flows, and ensuring robust logging and versioning.

## Architecture

### Core Components

#### Agent System
- **Base Agent Class** (`buttermilk._core.agent.Agent`): Abstract base class for all agents
  - Manages internal state (records, model context, data)
  - Provides lifecycle hooks (initialize, cleanup, reset)
  - Implements message handling and event processing
  - Supports tool definition generation via `get_tool_definitions()`

#### HOST Agents
- **HostAgent**: Base coordinator for group chats and flow control
- **SequenceHostAgent**: Executes agents in a predefined sequence
- **LLMHostAgent**: Uses LLM to decide next steps dynamically (uses natural language)
- **StructuredLLMHostAgent** (NEW): Refactored LLMHost using tool definitions
  - Discovers tools from participant agents automatically
  - Provides tools to LLM for direct invocation
  - Maintains backward compatibility with agents without tools
  - Consolidates previous variations (assistant, explorer, ra, selector) into a single configurable agent

#### Configuration System
- Hydra/OmegaConf based configuration
- YAML files define agents, flows, and orchestrators
- No manual dictionary configuration allowed

##### Available Host Configurations
- **`host/llm_host`**: Intelligent coordinator using StructuredLLMHostAgent
  - Adaptive execution mode where LLM decides workflow
  - Supports tool discovery from participant agents
  - Configurable via templates (e.g., `panel_host`)
- **`host/sequence_host`**: Sequential execution of agents in predefined order
  - Deterministic workflow execution
  - No LLM overhead for simple pipelines

### Data Flow
1. YAML configuration → Hydra → OmegaConf objects
2. OmegaConf → Pydantic models for validation
3. Agents process AgentInput → AgentOutput
4. AgentTrace captures full execution history

### Configuration Validation
- **Storage Configs**: Orchestrator uses StorageFactory.create_config() for discriminated union validation
- **Logger Configs**: LoggerConfig validators ensure required fields (project, location) for GCP
- **Agent Configs**: AgentConfig validates tool definitions and parameters
- **Early Validation**: All config validation happens during initialization, not at runtime

## Technology Stack
- **Language**: Python 3.10+
- **Core Dependencies**:
  - Pydantic v2 for data validation
  - Hydra-core for configuration
  - Autogen-core for agent communication
  - jsonschema for schema validation
- **Async**: asyncio for concurrent operations

## Project Structure
```
buttermilk/
├── _core/              # Core framework components
│   ├── agent.py        # Base Agent class
│   ├── config.py       # Configuration classes
│   ├── contract.py     # Data contracts (AgentInput, etc.)
│   ├── tool_definition.py    # NEW: Tool definition system
│   ├── mcp_decorators.py     # NEW: Tool decorators
│   └── schema_validation.py  # NEW: Schema utilities
├── agents/             # Agent implementations
│   ├── flowcontrol/    # HOST agents
│   └── example_tool_agent.py # NEW: Example tool agents
├── debug/              # Debugging infrastructure
│   ├── debug_agent.py  # LLM-driven debugging tools
│   └── README_DEBUGAGENT.md  # Debug tools documentation
├── orchestrators/      # Flow orchestration
├── runner/             # CLI and execution
└── utils/              # Utilities
```
