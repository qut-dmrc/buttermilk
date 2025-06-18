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
  - **NEW**: Supports tool definition generation via `get_tool_definitions()`
  - **NEW**: Unified request handling via `handle_unified_request()`

#### Tool Definition System (NEW - Issue #83)
- **Purpose**: Replace natural language agent descriptions with structured, type-safe tool definitions
- **Core Components**:
  - **AgentToolDefinition** (`buttermilk._core.tool_definition`): Structured tool definitions
    - Defines tool metadata, input/output schemas, MCP routes, permissions
    - Converts to various formats (Autogen, OpenAI, MCP)
  - **Decorators** (`buttermilk._core.mcp_decorators`):
    - `@tool`: Simple tool definition decorator
    - `@MCPRoute`: MCP route exposure decorator
    - Automatic JSON schema generation from type hints
  - **Schema Validation** (`buttermilk._core.schema_validation`):
    - JSON schema validation for inputs/outputs
    - Type coercion and example generation
  - **UnifiedRequest**: Consolidated request format for tool invocation
- **Integration Points**:
  - Agents expose tools via decorated methods
  - StructuredLLMHostAgent uses tools for coordination
  - MCP Server exposes tools as HTTP/WebSocket endpoints
  - Backward compatible with existing agents

#### HOST Agents
- **HostAgent**: Base coordinator for group chats and flow control
- **LLMHostAgent**: Uses LLM to decide next steps dynamically (uses natural language)
- **StructuredLLMHostAgent** (NEW): Refactored LLMHost using tool definitions
  - Discovers tools from participant agents automatically
  - Provides tools to LLM for direct invocation
  - Maintains backward compatibility with agents without tools

#### Configuration System
- Hydra/OmegaConf based configuration
- YAML files define agents, flows, and orchestrators
- No manual dictionary configuration allowed

### Data Flow
1. YAML configuration → Hydra → OmegaConf objects
2. OmegaConf → Pydantic models for validation
3. Agents process AgentInput → AgentOutput
4. AgentTrace captures full execution history

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
├── orchestrators/      # Flow orchestration
├── runner/             # CLI and execution
└── utils/              # Utilities
```

## Testing Strategy
- Unit tests for all core components
- Integration tests for agent communication
- pytest with async support
- Test files mirror source structure in `tests/`

## Development Commands
- Run Python: `uv run python ...`
- Run tests: `uv run pytest ...`
- View config: `uv run python -m buttermilk.runner.cli ... -c job`

## Development Guidelines
- Follow test-driven development
- Commit at every logical chunk
- Use GitHub issues for task tracking
- Prefer composition over inheritance
- Use Pydantic v2 validation
- Never change general agent interface for specific agents

## Security Considerations
- Tool permissions system for access control
- Schema validation prevents injection attacks
- Authentication required for MCP routes by default
- Never log or commit secrets

## Migration Strategy

### Moving from Natural Language to Tool Definitions
1. **Agents**: Add `@tool` decorators to methods (existing code continues to work)
2. **Flows**: Replace `sequencer` with `structured_sequencer` in configs
3. **Testing**: Use provided test utilities to verify tool schemas
4. **Gradual adoption**: Mix legacy and tool-based agents in same flow

### Example Migration
```yaml
# Before
- /agents/host@flow.observers.host: sequencer

# After  
- /agents/host@flow.observers.host: structured_sequencer
```

## Future Considerations
- ~~Phase 1: Tool Definition Framework~~ ✓ Complete
- ~~Phase 2: MCP Server implementation~~ ✓ Complete
- ~~Phase 3: Structured LLMHost~~ ✓ Complete
- Remote agent access via MCP
- Enhanced tool discovery and introspection
- Tool versioning and deprecation
- GraphQL API for tool queries