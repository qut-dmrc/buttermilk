# Buttermilk Tool Definition System

## Overview

The Tool Definition System is a new framework in Buttermilk that replaces natural language agent descriptions with structured, type-safe tool definitions. This enables more reliable agent coordination and opens the door for Model Context Protocol (MCP) integration.

## Key Benefits

1. **Type Safety**: Tools have explicit input/output schemas with validation
2. **Discoverability**: Agents can programmatically discover available tools
3. **Reliability**: No more parsing errors from natural language instructions
4. **MCP Ready**: Tools can be exposed as MCP routes for external access
5. **Backward Compatible**: Existing agents continue to work without modification

## Core Components

### AgentToolDefinition

The foundation of the system, representing a single tool capability:

```python
from buttermilk._core.tool_definition import AgentToolDefinition

tool_def = AgentToolDefinition(
    name="search_documents",
    description="Search for documents in the knowledge base",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 10}
        },
        "required": ["query"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "results": {"type": "array"},
            "total": {"type": "integer"}
        }
    },
    mcp_route="/search",  # Optional MCP endpoint
    permissions=["read"]   # Optional access control
)
```

### Tool Decorators

Two decorators make it easy to mark methods as tools:

```python
from buttermilk._core.mcp_decorators import tool, MCPRoute
from buttermilk._core.agent import Agent

class ResearchAgent(Agent):
    @tool  # Basic tool decorator
    async def analyze_text(self, text: str) -> dict:
        """Analyze text content."""
        return {"sentiment": "positive", "length": len(text)}
    
    @tool(name="search_docs")  # Custom tool name
    @MCPRoute("/search", permissions=["read"])  # With MCP route
    def search_documents(self, query: str, limit: int = 10) -> dict:
        """Search documents with query."""
        return {"results": [], "total": 0}
```

### UnifiedRequest

A standardized request format for tool invocation:

```python
from buttermilk._core.tool_definition import UnifiedRequest

# Call a specific tool
request = UnifiedRequest(
    target="researcher.search_documents",  # agent.tool format
    inputs={"query": "climate change", "limit": 5},
    context={"user_id": "123"},  # Optional context
    metadata={"request_id": "abc"}  # Optional metadata
)

# Call agent's default handler
request = UnifiedRequest(
    target="researcher",  # Just agent name
    inputs={"task": "analyze this"}
)
```

### StructuredLLMHostAgent

A new host agent that uses tool definitions instead of natural language:

```python
from buttermilk.agents.flowcontrol.structured_llmhost import StructuredLLMHostAgent

host = StructuredLLMHostAgent(
    agent_name="host",
    model_name="gpt-4",
    role="HOST",
    parameters={
        "model": "gpt-4",
        "template": "host_structured_tools"
    }
)
```

## Migration Guide

### For Agent Developers

1. **Add tool decorators to your methods**:

```python
# Before
class MyAgent(Agent):
    def process_data(self, data: str) -> str:
        return data.upper()

# After
class MyAgent(Agent):
    @tool
    def process_data(self, data: str) -> str:
        """Process data by converting to uppercase."""
        return data.upper()
```

2. **Use type hints** - they're automatically converted to JSON schemas
3. **Add docstrings** - they become tool descriptions

### For Flow Developers

1. **Update host configuration**:

```yaml
# Before
- /agents/host@flow.observers.host: sequencer

# After  
- /agents/host@flow.observers.host: structured_sequencer
```

2. **Existing agents work as-is** - the system creates default tools for agents without explicit tool definitions

### For Orchestrator Developers

The new `handle_unified_request` method on agents provides a unified interface:

```python
# Works for any agent, with or without tools
result = await agent.handle_unified_request(
    UnifiedRequest(
        target="agent.tool_name",
        inputs={...}
    )
)
```

## Advanced Features

### MCP Server Integration

Tools with MCP routes can be exposed via HTTP/WebSocket:

```python
from buttermilk.mcp.server import MCPServer, MCPServerConfig

config = MCPServerConfig(
    mode="daemon",
    host="0.0.0.0",
    port=8000,
    auth_required=True
)

server = MCPServer(config=config, orchestrator=orchestrator)
await server.start()
```

### Schema Validation

All tool inputs are validated against their schemas:

```python
from buttermilk._core.schema_validation import validate_tool_input

# Automatic validation on tool calls
try:
    result = await agent.handle_unified_request(request)
except SchemaValidationError as e:
    print(f"Invalid input: {e}")
```

### Tool Discovery

Agents can discover available tools:

```python
# Get all tools from an agent
tools = agent.get_tool_definitions()

for tool in tools:
    print(f"{tool.name}: {tool.description}")
    print(f"  Input: {tool.input_schema}")
    print(f"  Output: {tool.output_schema}")
```

## Example: Complete Flow

Here's a complete example showing the system in action:

```python
# 1. Define an agent with tools
class AnalysisAgent(Agent):
    @tool
    @MCPRoute("/analyze")
    async def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment of text."""
        # ... implementation ...
        return {"sentiment": "positive", "score": 0.8}
    
    @tool
    def extract_keywords(self, text: str, max_keywords: int = 5) -> list[str]:
        """Extract keywords from text."""
        # ... implementation ...
        return ["keyword1", "keyword2"]

# 2. Create structured host
host = StructuredLLMHostAgent(
    agent_name="coordinator",
    model_name="gpt-4",
    role="HOST",
    parameters={"model": "gpt-4", "template": "host_structured_tools"}
)

# 3. Initialize with participants
host._participants = {"ANALYZER": AnalysisAgent(...)}
await host._initialize(callback_to_groupchat=...)

# 4. Host can now use tools via LLM function calling
# The LLM sees available tools and can invoke them directly
```

## Testing

The system includes comprehensive test utilities:

```python
from buttermilk._core.tool_definition import AgentToolDefinition
from buttermilk._core.schema_validation import validate_tool_input

# Test tool definition
def test_tool_schema():
    tool = AgentToolDefinition(
        name="test_tool",
        description="Test",
        input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        output_schema={"type": "string"}
    )
    
    # Validate inputs
    validate_tool_input(tool, {"x": 42})  # OK
    validate_tool_input(tool, {"x": "not an int"})  # Raises SchemaValidationError
```

## Best Practices

1. **Use descriptive tool names**: `analyze_sentiment` not `process`
2. **Provide clear descriptions**: These help LLMs understand when to use tools
3. **Define precise schemas**: Include all required fields and constraints
4. **Handle errors gracefully**: Tools should return error info, not raise exceptions
5. **Keep tools focused**: Each tool should do one thing well
6. **Use async for I/O**: Mark I/O-bound tools as async

## Troubleshooting

### Common Issues

1. **Tool not found**: Ensure the method has the `@tool` decorator
2. **Schema validation fails**: Check that inputs match the defined schema
3. **MCP route conflicts**: Ensure unique routes across all tools
4. **Async issues**: Remember to await async tool calls

### Debug Mode

Enable detailed logging to troubleshoot issues:

```python
import logging
logging.getLogger("buttermilk._core.tool_definition").setLevel(logging.DEBUG)
```

## Future Enhancements

- Automatic OpenAPI spec generation
- Tool versioning support
- Rate limiting and quotas
- Tool composition and chaining
- Runtime tool registration