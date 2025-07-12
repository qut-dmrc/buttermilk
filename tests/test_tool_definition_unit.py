"""Unit tests for the tool definition system.

Tests the AgentToolDefinition, decorators, and schema validation utilities.
"""

import pytest
from typing import Any, Literal

from buttermilk._core import AgentInput
from buttermilk._core.agent import Agent
from buttermilk._core.contract import AgentOutput
from buttermilk._core.tool_definition import (
    AgentToolDefinition,
    MCPServerConfig,
    UnifiedRequest,
)
from buttermilk._core.mcp_decorators import (
    MCPRoute,
    tool,
    extract_tool_definitions,
    _type_to_json_schema,
)
from buttermilk._core.schema_validation import (
    SchemaValidator,
    SchemaValidationError,
    validate_tool_input,
    validate_tool_output,
    coerce_to_schema,
    generate_example_from_schema,
)


class TestAgentToolDefinition:
    """Test AgentToolDefinition class."""
    
    def test_basic_tool_definition(self):
        """Test creating a basic tool definition."""
        tool_def = AgentToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"output": {"type": "string"}}},
        )
        
        assert tool_def.name == "test_tool"
        assert tool_def.description == "A test tool"
        assert tool_def.mcp_route is None
        assert tool_def.permissions == []
    
    def test_tool_definition_with_mcp_route(self):
        """Test tool definition with MCP route."""
        tool_def = AgentToolDefinition(
            name="analyze",
            description="Analyze data",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            mcp_route="/analyze",
            permissions=["read:data"],
        )
        
        assert tool_def.mcp_route == "/analyze"
        assert tool_def.permissions == ["read:data"]
    
    def test_to_autogen_schema(self):
        """Test conversion to Autogen tool schema."""
        tool_def = AgentToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            },
            output_schema={"type": "string"},
        )
        
        schema = tool_def.schema
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "A test tool"
        assert schema["function"]["parameters"] == tool_def.input_schema
    
    
    def test_to_mcp_route_definition(self):
        """Test conversion to MCP route definition."""
        tool_def = AgentToolDefinition(
            name="analyze",
            description="Analyze data",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            mcp_route="/api/analyze",
            permissions=["read", "write"],
        )
        
        route_def = tool_def.to_mcp_route_definition()
        assert route_def is not None
        assert route_def["path"] == "/api/analyze"
        assert route_def["method"] == "POST"
        assert route_def["handler"] == "analyze"
        assert route_def["permissions"] == ["read", "write"]
    
    def test_to_mcp_route_definition_none(self):
        """Test MCP route definition when no route specified."""
        tool_def = AgentToolDefinition(
            name="test",
            description="Test",
            input_schema={},
            output_schema={},
        )
        
        assert tool_def.to_mcp_route_definition() is None


class TestMCPServerConfig:
    """Test MCPServerConfig class."""
    
    def test_default_config(self):
        """Test default server configuration."""
        config = MCPServerConfig()
        assert config.mode == "embedded"
        assert config.port == 8787
        assert config.auth_required is True
        assert config.allowed_origins == ["*"]
    
    def test_daemon_mode(self):
        """Test daemon mode configuration."""
        config = MCPServerConfig(
            mode="daemon",
            port=9000,
            auth_required=False,
            allowed_origins=["http://localhost:3000"]
        )
        assert config.mode == "daemon"
        assert config.port == 9000
        assert config.auth_required is False
        assert config.allowed_origins == ["http://localhost:3000"]


class TestUnifiedRequest:
    """Test UnifiedRequest class."""
    
    def test_basic_request(self):
        """Test basic unified request."""
        request = UnifiedRequest(
            target="agent_name.tool_name",
            inputs={"key": "value"},
            context={"session": "123"},
            metadata={"user_id": "456"}
        )
        
        assert request.agent_name == "agent_name"
        assert request.tool_name == "tool_name"
        assert request.inputs == {"key": "value"}
    
    def test_agent_only_target(self):
        """Test request with agent-only target."""
        request = UnifiedRequest(target="agent_name")
        assert request.agent_name == "agent_name"
        assert request.tool_name is None
    
    def test_empty_collections(self):
        """Test request with empty collections."""
        request = UnifiedRequest(target="test")
        assert request.inputs == {}
        assert request.context == {}
        assert request.metadata == {}


class TestDecorators:
    """Test tool and MCPRoute decorators."""
    
    def test_tool_decorator_basic(self):
        """Test basic tool decorator."""
        @tool
        def my_tool(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y
        
        assert hasattr(my_tool, "_tool_metadata")
        assert my_tool._tool_metadata["name"] == "my_tool"
        assert my_tool._tool_metadata["description"] == "Add two numbers."
        assert my_tool._tool_metadata["include_in_mcp"] is True
    
    def test_tool_decorator_with_args(self):
        """Test tool decorator with arguments."""
        @tool(name="custom_name", description="Custom description", include_in_mcp=False)
        def my_tool(x: int) -> int:
            return x * 2
        
        assert my_tool._tool_metadata["name"] == "custom_name"
        assert my_tool._tool_metadata["description"] == "Custom description"
        assert my_tool._tool_metadata["include_in_mcp"] is False
    
    def test_mcp_route_decorator(self):
        """Test MCPRoute decorator."""
        @MCPRoute("/compute", permissions=["execute"], description="Compute result")
        def compute(a: float, b: float) -> float:
            return a * b
        
        assert hasattr(compute, "_mcp_route")
        assert compute._mcp_route["path"] == "/compute"
        assert compute._mcp_route["permissions"] == ["execute"]
        assert compute._mcp_route["description"] == "Compute result"
        assert compute._mcp_route["include_in_tools"] is True
    
    def test_type_to_json_schema(self):
        """Test type conversion to JSON schema."""
        # Basic types
        assert _type_to_json_schema(str) == {"type": "string"}
        assert _type_to_json_schema(int) == {"type": "integer"}
        assert _type_to_json_schema(float) == {"type": "number"}
        assert _type_to_json_schema(bool) == {"type": "boolean"}
        assert _type_to_json_schema(list) == {"type": "array"}
        assert _type_to_json_schema(dict) == {"type": "object"}
        
        # None type
        assert _type_to_json_schema(type(None)) == {"type": "null"}
        
        # List with type parameter
        from typing import List
        schema = _type_to_json_schema(List[str])
        assert schema == {"type": "array", "items": {"type": "string"}}


class TestSchemaValidation:
    """Test schema validation utilities."""
    
    def test_schema_validator_valid(self):
        """Test validator with valid data."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }
        
        validator = SchemaValidator(schema)
        validator.validate({"name": "John", "age": 30})
        assert validator.is_valid({"name": "Jane"})
    
    def test_schema_validator_invalid(self):
        """Test validator with invalid data."""
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"]
        }
        
        validator = SchemaValidator(schema)
        
        with pytest.raises(SchemaValidationError) as exc_info:
            validator.validate({"count": "not a number"})
        assert "count: " in str(exc_info.value)
        
        assert not validator.is_valid({})  # Missing required field
    
    def test_validate_partial(self):
        """Test partial validation (ignoring required)."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
        
        validator = SchemaValidator(schema)
        # Full validation would fail without age
        with pytest.raises(SchemaValidationError):
            validator.validate({"name": "John"})
        
        # Partial validation should pass
        validator.validate_partial({"name": "John"})
    
    def test_coerce_to_schema(self):
        """Test data coercion to match schema."""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "ratio": {"type": "number"},
                "active": {"type": "boolean"},
                "name": {"type": "string"}
            }
        }
        
        data = {
            "count": "123",
            "ratio": "3.14",
            "active": "true",
            "name": 42
        }
        
        coerced = coerce_to_schema(schema, data)
        assert coerced["count"] == 123
        assert coerced["ratio"] == 3.14
        assert coerced["active"] is True
        assert coerced["name"] == "42"
    
    def test_generate_example_from_schema(self):
        """Test example generation from schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2
                },
                "active": {"type": "boolean"}
            },
            "required": ["name", "age"]
        }
        
        example = generate_example_from_schema(schema)
        assert isinstance(example, dict)
        assert "name" in example
        assert "age" in example
        assert isinstance(example["name"], str)
        assert isinstance(example["age"], int)
        assert example["age"] >= 0
    
    def test_tool_input_validation(self):
        """Test tool input validation helper."""
        schema = {
            "type": "object",
            "properties": {"x": {"type": "number"}},
            "required": ["x"]
        }
        
        # Valid input
        validated = validate_tool_input(schema, {"x": 3.14})
        assert validated == {"x": 3.14}
        
        # Invalid input
        with pytest.raises(SchemaValidationError):
            validate_tool_input(schema, {"x": "not a number"})


class TestAgentToolGeneration:
    """Test tool generation from agents."""
    
    def test_extract_tools_from_agent(self):
        """Test extracting tool definitions from an agent."""
        
        class TestAgent(Agent):
            async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
                return AgentOutput(source="test", role="test", outputs={})
            
            @tool
            def simple_tool(self, text: str) -> str:
                """Process text."""
                return text.upper()
            
            @MCPRoute("/analyze")
            async def analyze(self, data: dict[str, Any]) -> dict[str, Any]:
                """Analyze data."""
                return {"result": "analyzed"}
            
            def _private_method(self):
                """Should not be included."""
                pass
            
            def public_method_no_decorator(self):
                """Should not be included without decorator."""
                pass
        
        agent = TestAgent(agent_name="test", model_name="test", role="test")
        tools = extract_tool_definitions(agent)
        
        assert len(tools) == 2
        
        # Check simple_tool
        simple_tool_def = next(t for t in tools if t.name == "simple_tool")
        assert simple_tool_def.description == "Process text."
        assert simple_tool_def.mcp_route == "/simple_tool"
        assert "text" in simple_tool_def.input_schema["properties"]
        
        # Check analyze tool
        analyze_def = next(t for t in tools if t.name == "analyze")
        assert analyze_def.description == "Analyze data."
        assert analyze_def.mcp_route == "/analyze"
        assert "data" in analyze_def.input_schema["properties"]
    
    def test_agent_get_tool_definitions(self):
        """Test Agent.get_tool_definitions() method."""
        
        class CalculatorAgent(Agent):
            async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
                return AgentOutput(source="calc", role="calc", outputs={})
            
            @tool(name="add", description="Add two numbers")
            def add(self, a: float, b: float) -> float:
                return a + b
            
            @tool(include_in_mcp=False)
            def subtract(self, a: float, b: float) -> float:
                """Subtract b from a."""
                return a - b
        
        agent = CalculatorAgent(agent_name="calc", model_name="calc", role="calculator")
        tools = agent.get_tool_definitions()
        
        assert len(tools) == 2
        
        add_tool = next(t for t in tools if t.name == "add")
        assert add_tool.description == "Add two numbers"
        assert add_tool.mcp_route == "/add"
        
        subtract_tool = next(t for t in tools if t.name == "subtract")
        assert subtract_tool.description == "Subtract b from a."
        assert subtract_tool.mcp_route is None  # include_in_mcp=False