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

#### Debugging Infrastructure (Issue #116)
- **DebugAgent** (`buttermilk.debug.debug_agent`): LLM-driven debugging tools
  - Log reading tools for raw access to `/tmp/buttermilk_*.log` files
  - WebSocket client controller for interactive flow testing
  - Exposed via `@tool` decorator for MCP discovery
- **MCP Debug Server** (`scripts/run_debug_mcp_server.py`): HTTP server on port 8090
  - Exposes DebugAgent tools as REST endpoints
  - Enables LLM integration for automated debugging
- **Design Philosophy**: Simple tools, smart LLM - no intelligence in tools

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

## Testing Strategy
- Unit tests for all core components
- Integration tests for agent communication
- pytest with async support (pytest-asyncio)
- Test files mirror source structure in `tests/`
- Config validation tests ensure early failure with clear messages

## Development Commands
- Run Python: `uv run python ...`
- Run tests: `uv run pytest ...`
- View config: `uv run python -m buttermilk.runner.cli ... -c job`
- Add dev dependencies: `uv add --dev <package>`

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
2. **Flows**: Update host configurations to use `StructuredLLMHostAgent` via `llm_host.yaml`
3. **Testing**: Use provided test utilities to verify tool schemas
4. **Gradual adoption**: Mix legacy and tool-based agents in same flow

### Example Migration
```yaml
# Before (using legacy natural language host)
- /agents/host@flow.observers.host: host/llm_host  # (when agent_obj was LLMHostAgent)

# After (using structured tool-based host)
- /agents/host@flow.observers.host: host/llm_host  # (now uses StructuredLLMHostAgent)
```

## AutoGen MCP Integration (NEW - Issue #94)

### Problem Solved
- **Complex Tool Construction**: Eliminated complicated tool construction code by using AutoGen's `Tool` interface
- **Dual-Mode Operation**: Same agents now work in both groupchat and MCP contexts without code changes
- **Industry Standards**: Replaced natural language agent descriptions with structured tool definitions

### Core Components
- **AutoGenMCPAdapter** (`buttermilk.mcp.autogen_adapter`): Main bridge between Buttermilk and AutoGen MCP
  - Uses AutoGen's `McpWorkbench` for standard MCP protocol compliance
  - Wraps agent tools as AutoGen `Tool` objects via `MCPToolAdapter`
  - Integrates with tool discovery service for dynamic registration
- **ToolDiscoveryService** (`buttermilk.mcp.tool_registry`): Dynamic tool discovery and management
  - Automatically discovers tools from agent `@tool` decorators
  - Provides callback system for real-time tool registration
  - Maintains unified registry of all agent tools
- **Enhanced UnifiedRequest**: Extended to support both MCP and groupchat contexts
  - `from_mcp_call()` and `from_groupchat_step()` factory methods
  - Context detection via `is_mcp_request` and `is_groupchat_request` properties
- **MCPHostProvider**: Simplified API for MCP hosting capabilities
  - High-level interface for registering agents and starting MCP services
  - Handles orchestrator integration and tool management

### Usage Patterns

#### Dual-Mode Agent Definition
```python
class CalculatorAgent(Agent):
    @tool
    @MCPRoute("/calc/add")
    def add(self, a: float, b: float) -> float:
        return a + b

    async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
        # Standard groupchat processing
        return AgentOutput(...)
```

#### Groupchat Usage (Existing)
```python
# StructuredLLMHostAgent automatically discovers and uses tools
host = StructuredLLMHostAgent(...)
# Tools are available to LLM for structured calling
```

#### MCP Usage (New)
```python
# Expose agents as MCP tools
mcp_provider = MCPHostProvider()
mcp_provider.register_agent(calculator_agent)
await mcp_provider.start()

# External MCP clients can now discover and use tools
result = await mcp_provider.invoke_tool("calc_agent.add", {"a": 5, "b": 3})
```

### Benefits
- **Radical Simplification**: Agents are now simple with clean `_process()` method and `@tool` decorators
- **Zero Code Changes**: Same agent works in both contexts without modifications
- **Industry Standard**: Uses AutoGen's MCP implementation for compatibility
- **Dynamic Discovery**: Tools automatically discovered from decorators
- **Structured Tool Calling**: LLMs get proper schemas instead of natural language

### Integration with Existing Systems
- Fully backward compatible with existing groupchat flows
- StructuredLLMHostAgent can use dynamically discovered tools
- Existing agents can be incrementally migrated to use `@tool` decorators
- No changes required to orchestrators or existing flows

## Future Considerations
- ~~Phase 1: Tool Definition Framework~~ ✓ Complete
- ~~Phase 2: MCP Server implementation~~ ✓ Complete
- ~~Phase 3: Structured LLMHost~~ ✓ Complete
- ~~Phase 4: AutoGen MCP Integration~~ ✓ Complete (Issue #94)
- Remote agent access via MCP (external Buttermilk instances)
- Enhanced tool discovery and introspection
- Tool versioning and deprecation
- GraphQL API for tool queries
- Integration with other MCP clients (VS Code, IDEs)