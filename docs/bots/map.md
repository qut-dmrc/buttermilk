# Buttermilk Project Structure Map

## Directory Overview

```
buttermilk/
├── buttermilk/              # Core package
│   ├── _core/              # Framework essentials
│   ├── agents/             # Agent implementations
│   ├── orchestrators/      # Flow orchestration
│   ├── runner/             # CLI and execution
│   ├── utils/              # Utilities
│   ├── storage/            # Storage implementations
│   ├── frontend/           # UI components
│   ├── debug/              # Debugging tools
│   └── mcp/                # MCP integration
├── conf/                   # Configuration files
│   ├── agents/             # Agent configs
│   ├── flows/              # Flow definitions
│   ├── llms/               # Model configs
│   ├── run/                # Run modes
│   └── storage/            # Storage configs
├── docs/                   # Documentation
│   ├── bots/              # LLM developer docs
│   ├── developer-guide/    # Dev documentation
│   ├── user-guide/        # User documentation
│   └── reference/         # API reference
├── tests/                  # Test suite
├── scripts/               # Utility scripts
└── templates/             # Templates and examples
```

## Core Components (`buttermilk/_core/`)

### Essential Files
- **agent.py**: Base `Agent` class - all agents inherit from this
- **orchestrator.py**: Base `Orchestrator` class for flow management
- **contract.py**: Pydantic models for messages (`AgentInput`, `AgentTrace`)
- **config.py**: Configuration classes (`AgentConfig`, `FlowConfig`)
- **types.py**: Core types (`Record`, `MediaObj`)
- **llm.py**: LLM integration base classes

### New Tool System
- **tool_definition.py**: Tool definition structures
- **mcp_decorators.py**: `@tool` and `@MCPRoute` decorators
- **schema_validation.py**: JSON schema utilities

## Agent Implementations (`buttermilk/agents/`)

### Flow Control Agents
- **flowcontrol/**
  - `host.py`: Base HOST agent
  - `llm_host.py`: LLM-based coordinator
  - `structured_llm_host.py`: Tool-based coordinator
  - `sequence_host.py`: Sequential execution

### Specialized Agents
- **example_tool_agent.py**: Example with tool definitions
- **ui/**: User interface agents
  - `base.py`: Base UI agent
  - `cli_agent.py`: Terminal interface
  - `slack_agent.py`: Slack integration

## Configuration (`conf/`)

### Key Configuration Files
- **config.yaml**: Base configuration with defaults
- **local.yaml**: Local environment overrides

### Agent Configurations (`conf/agents/`)
- **rag/**: RAG agent configs
  - `researcher.yaml`: Research agent
  - `policy_analyst.yaml`: Analysis agent
- **host/**: Orchestrator configs
  - `llm_host.yaml`: LLM coordinator
  - `sequence_host.yaml`: Sequential flow

### Flow Definitions (`conf/flows/`)
- **osb.yaml**: OSB vector store query flow
- **trans.yaml**: Journalism assessment flow
- **tox.yaml**: Toxicity analysis flow
- **zot.yaml**: Zotero RAG flow

### Model Configurations (`conf/llms/`)
- **lite.yaml**: Lightweight models (gemini-flash)
- **full.yaml**: Full model suite (gemini-pro, claude, gpt)

## Debugging Tools (`buttermilk/debug/`)

### Key Files
- **debug_agent.py**: LLM-driven debugging agent
- **ws_debug_cli.py**: WebSocket debugging CLI
- **README_DEBUGAGENT.md**: Debug tool documentation

### Debug Commands
```bash
# Test WebSocket connection
uv run python -m buttermilk.debug.ws_debug_cli test-connection

# Start flow for debugging
uv run python -m buttermilk.debug.ws_debug_cli start <flow> --wait 20

# Send messages to flow
uv run python -m buttermilk.debug.ws_debug_cli send "message" --session <id>
```

## Runner & CLI (`buttermilk/runner/`)

### Main Entry Points
- **cli.py**: Main CLI entry point
  - Hydra integration
  - Flow execution
  - Configuration loading

### Key Components
- **groupchat.py**: Group chat orchestration
- **selector.py**: Dynamic agent selection
- **api.py**: REST API server

## Storage (`buttermilk/storage/`)

### Storage Implementations
- **local.py**: File-based storage
- **bigquery.py**: Google BigQuery integration
- **gcs.py**: Google Cloud Storage
- **factory.py**: Storage factory pattern

## MCP Integration (`buttermilk/mcp/`)

### Key Files
- **autogen_adapter.py**: AutoGen MCP bridge
- **tool_registry.py**: Tool discovery service
- **server.py**: MCP server implementation

## Test Structure (`tests/`)

### Test Organization
- Mirrors source structure
- **unit/**: Unit tests
- **integration/**: Integration tests
- **fixtures/**: Test data and mocks

## Important Files to Know

### Configuration Entry Points
- `/conf/config.yaml`: Main config file
- `/conf/local.yaml`: Local overrides
- `CLAUDE.md`: Bot developer instructions

### Documentation
- `/README.md`: Project overview
- `/docs/bots/README.md`: Bot knowledge bank
- `/docs/developer-guide/architecture.md`: Architecture details

### Debugging
- `/tmp/buttermilk*.log`: Runtime logs
- `scripts/run_debug_mcp_server.py`: Debug server

## Common Task Locations

### To Create a New Agent
1. Implement in `buttermilk/agents/your_agent.py`
2. Configure in `conf/agents/your_agent.yaml`
3. Add to flow in `conf/flows/your_flow.yaml`

### To Add a New Flow
1. Create `conf/flows/new_flow.yaml`
2. Reference agents from `conf/agents/`
3. Test with `uv run python -m buttermilk.runner.cli +flow=new_flow`

### To Debug Issues
1. Check logs in `/tmp/buttermilk*.log`
2. Use debug CLI in `buttermilk/debug/`
3. View config with `-c job` flag

### To Run Tests
1. Unit tests: `uv run pytest tests/unit/`
2. Integration: `uv run pytest tests/integration/`
3. Specific test: `uv run pytest tests/path/to/test.py::test_name`

## Environment Variables

### Required for Testing
- `GOOGLE_CLOUD_PROJECT`: GCP project ID
- `OPENAI_API_KEY`: OpenAI credentials
- `ANTHROPIC_API_KEY`: Anthropic credentials

### Optional
- `BUTTERMILK_LOG_LEVEL`: Logging verbosity
- `HYDRA_FULL_ERROR`: Show full Hydra errors

## Quick Navigation Tips

### Finding Things
- **Agent implementations**: Look in `buttermilk/agents/`
- **Configuration examples**: Check `conf/agents/` and `conf/flows/`
- **Test examples**: Mirror path in `tests/`
- **Documentation**: Start with `docs/bots/README.md`

### Common Patterns
- **Base classes**: Always in `_core/`
- **Configs**: YAML files in `conf/`
- **Tests**: Mirror source structure
- **Docs**: Organized by audience

Remember: When exploring the codebase, start with configuration files to understand structure, then look at base classes in `_core/`, and finally examine specific implementations.