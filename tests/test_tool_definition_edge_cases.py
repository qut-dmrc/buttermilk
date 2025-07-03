"""Edge case tests for the tool definition system."""

import pytest
from typing import Any, Optional, Union, Literal
from unittest.mock import Mock, AsyncMock
import inspect

from pydantic import BaseModel, Field

from buttermilk._core.agent import Agent
from buttermilk._core.contract import AgentInput, AgentOutput
from buttermilk._core.tool_definition import AgentToolDefinition
from buttermilk._core.schema_validation import SchemaValidationError
from buttermilk._core.mcp_decorators import tool, MCPRoute, extract_tool_definitions, _type_to_json_schema
from buttermilk._core.schema_validation import (
    SchemaValidator,
    validate_tool_input,
    coerce_to_schema,
    merge_schemas
)

# Use anyio for async tests
pytestmark = pytest.mark.anyio


class ComplexTypeModel(BaseModel):
    """Complex Pydantic model for testing."""
    id: int
    name: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: Literal["active", "inactive", "pending"] = "pending"
    parent: Optional["ComplexTypeModel"] = None


class TestEdgeCaseTypeConversion:
    """Test edge cases in type to JSON schema conversion."""
    
    def test_complex_pydantic_model(self):
        """Test conversion of complex Pydantic models."""
        schema = _type_to_json_schema(ComplexTypeModel)
        
        # Pydantic models with self-references use $ref and $defs
        assert "$defs" in schema or "definitions" in schema or "$ref" in schema
        
        # If it has a $ref, it's a valid Pydantic schema
        if "$ref" in schema:
            assert "$defs" in schema
            # The actual definition should be in $defs
            model_name = schema["$ref"].split("/")[-1]
            assert model_name in schema["$defs"]
            actual_schema = schema["$defs"][model_name]
            assert actual_schema["type"] == "object"
            assert "properties" in actual_schema
            assert "id" in actual_schema["properties"]
            assert "name" in actual_schema["properties"]
        else:
            # Direct schema without $ref
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "id" in schema["properties"]
            assert "name" in schema["properties"]
    
    def test_optional_types(self):
        """Test Optional type conversion."""
        from typing import Optional
        
        schema = _type_to_json_schema(Optional[str])
        assert schema["type"] == "string"
        assert schema["nullable"] is True
        
        schema = _type_to_json_schema(Optional[int])
        assert schema["type"] == "integer"
        assert schema["nullable"] is True
    
    def test_union_types(self):
        """Test Union type conversion."""
        from typing import Union
        
        schema = _type_to_json_schema(Union[str, int])
        assert "anyOf" in schema
        assert len(schema["anyOf"]) == 2
        assert {"type": "string"} in schema["anyOf"]
        assert {"type": "integer"} in schema["anyOf"]
    
    def test_list_with_complex_types(self):
        """Test List with complex element types."""
        from typing import List
        
        schema = _type_to_json_schema(List[ComplexTypeModel])
        assert schema["type"] == "array"
        assert "items" in schema
        
        # Items should be the Pydantic model schema
        items_schema = schema["items"]
        # Could be a $ref or direct schema
        if "$ref" in items_schema:
            assert "$defs" in items_schema or items_schema["$ref"].startswith("#/")
        else:
            assert items_schema.get("type") == "object" or "$defs" in items_schema
    
    def test_dict_with_typed_values(self):
        """Test Dict with specific value types."""
        from typing import Dict
        
        # Currently simplified to just {"type": "object"}
        schema = _type_to_json_schema(Dict[str, int])
        assert schema["type"] == "object"
    
    def test_none_type(self):
        """Test None type conversion."""
        schema = _type_to_json_schema(type(None))
        assert schema == {"type": "null"}
    
    def test_any_type(self):
        """Test Any type defaults to string."""
        from typing import Any
        
        schema = _type_to_json_schema(Any)
        assert schema == {"type": "string"}


class TestEdgeCaseAgentTools:
    """Test edge cases in agent tool extraction."""
    
    def test_agent_with_private_methods(self):
        """Test that private methods are not exposed as tools."""
        
        class PrivateMethodAgent(Agent):
            async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
                return AgentOutput(agent_id=self.agent_id, outputs={})
            
            @tool
            def public_tool(self) -> str:
                return "public"
            
            @tool
            def _private_tool(self) -> str:
                """This should not be exposed."""
                return "private"
            
            def _another_private(self) -> str:
                """Also not exposed."""
                return "private"
        
        agent = PrivateMethodAgent(agent_name="test", model_name="test", role="test")
        tools = agent.get_tool_definitions()
        
        # Only public_tool should be extracted
        assert len(tools) == 1
        assert tools[0].name == "public_tool"
    
    def test_method_with_complex_signature(self):
        """Test tool with complex method signature."""
        
        class ComplexSignatureAgent(Agent):
            async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
                return AgentOutput(agent_id=self.agent_id, outputs={})
            
            @tool
            def complex_tool(
                self,
                required_str: str,
                optional_int: int = 42,
                *args,
                keyword_only: bool,
                optional_keyword: str = "default",
                **kwargs
            ) -> dict[str, Any]:
                """Tool with complex signature."""
                return {"result": "complex"}
        
        agent = ComplexSignatureAgent(agent_name="test", model_name="test", role="test")
        tools = agent.get_tool_definitions()
        
        assert len(tools) == 1
        tool_def = tools[0]
        
        # Check required parameters
        assert "required_str" in tool_def.input_schema["required"]
        assert "keyword_only" in tool_def.input_schema["required"]
        
        # Check optional parameters have defaults
        assert "optional_int" not in tool_def.input_schema["required"]
        assert "optional_keyword" not in tool_def.input_schema["required"]
    
    def test_tool_without_return_type(self):
        """Test tool method without return type annotation."""
        
        class NoReturnTypeAgent(Agent):
            async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
                return AgentOutput(agent_id=self.agent_id, outputs={})
            
            @tool
            def no_return_tool(self, value: str):
                """Tool without return type annotation."""
                return value.upper()
        
        agent = NoReturnTypeAgent(agent_name="test", model_name="test", role="test")
        tools = agent.get_tool_definitions()
        
        assert len(tools) == 1
        tool_def = tools[0]
        
        # Should default to object output schema
        assert tool_def.output_schema == {"type": "object"}
    
    def test_multiple_decorators_same_method(self):
        """Test method with both @tool and @MCPRoute decorators."""
        
        class MultiDecoratorAgent(Agent):
            async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
                return AgentOutput(agent_id=self.agent_id, outputs={})
            
            @tool(name="custom_name")
            @MCPRoute("/custom", permissions=["admin"])
            def multi_decorated(self, data: str) -> str:
                """Method with multiple decorators."""
                return data
        
        agent = MultiDecoratorAgent(agent_name="test", model_name="test", role="test")
        tools = agent.get_tool_definitions()
        
        # Should use MCPRoute info since it's the outer decorator
        assert len(tools) == 1
        tool_def = tools[0]
        assert tool_def.name == "multi_decorated"  # From method name
        assert tool_def.mcp_route == "/custom"
        assert tool_def.permissions == ["admin"]


class TestSchemaValidationEdgeCases:
    """Test edge cases in schema validation."""
    
    def test_nested_object_validation(self):
        """Test validation of deeply nested objects."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "profile": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer"}
                            },
                            "required": ["name"]
                        }
                    },
                    "required": ["profile"]
                }
            },
            "required": ["user"]
        }
        
        validator = SchemaValidator(schema)
        
        # Valid nested data
        valid_data = {
            "user": {
                "profile": {
                    "name": "John",
                    "age": 30
                }
            }
        }
        validator.validate(valid_data)  # Should not raise
        
        # Missing required nested field
        invalid_data = {
            "user": {
                "profile": {
                    "age": 30  # Missing required 'name'
                }
            }
        }
        with pytest.raises(SchemaValidationError) as exc_info:
            validator.validate(invalid_data)
        assert "name" in str(exc_info.value)
    
    def test_array_item_validation(self):
        """Test validation of array items."""
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "value": {"type": "string"}
                },
                "required": ["id"]
            },
            "minItems": 1
        }
        
        validator = SchemaValidator(schema)
        
        # Valid array
        valid_data = [
            {"id": 1, "value": "first"},
            {"id": 2, "value": "second"}
        ]
        validator.validate(valid_data)
        
        # Invalid item in array
        invalid_data = [
            {"id": 1, "value": "first"},
            {"value": "missing id"}  # Missing required 'id'
        ]
        with pytest.raises(SchemaValidationError):
            validator.validate(invalid_data)
        
        # Empty array (violates minItems)
        with pytest.raises(SchemaValidationError):
            validator.validate([])
    
    def test_coerce_edge_cases(self):
        """Test edge cases in type coercion."""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "enabled": {"type": "boolean"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
        
        # Test various coercion scenarios
        data = {
            "count": "123.45",  # Float string to int
            "enabled": "false",  # String to bool
            "tags": ["tag1", 123, True]  # Mixed types to string
        }
        
        coerced = coerce_to_schema(schema, data)
        
        assert coerced["count"] == 123  # Truncated to int
        assert coerced["enabled"] is False
        assert coerced["tags"] == ["tag1", "123", "True"]  # All converted to strings
    
    def test_additional_properties_handling(self):
        """Test handling of additionalProperties in schema."""
        # Schema that forbids additional properties
        strict_schema = {
            "type": "object",
            "properties": {
                "allowed": {"type": "string"}
            },
            "additionalProperties": False
        }
        
        data = {
            "allowed": "value",
            "extra": "should be removed"
        }
        
        coerced = coerce_to_schema(strict_schema, data)
        assert "allowed" in coerced
        assert "extra" not in coerced  # Removed because additionalProperties is False
    
    def test_merge_conflicting_schemas(self):
        """Test merging schemas with conflicts."""
        schema1 = {
            "type": "object",
            "properties": {
                "field": {"type": "string"},
                "shared": {"type": "integer"}
            },
            "required": ["field"]
        }
        
        schema2 = {
            "type": "object",
            "properties": {
                "field": {"type": "number"},  # Conflicts with schema1
                "other": {"type": "boolean"}
            },
            "required": ["other"]
        }
        
        merged = merge_schemas(schema1, schema2)
        
        # Should have all properties
        assert "field" in merged["properties"]
        assert "shared" in merged["properties"]
        assert "other" in merged["properties"]
        
        # Required fields should be combined
        assert set(merged["required"]) == {"field", "other"}


class TestUnifiedRequestEdgeCases:
    """Test edge cases for UnifiedRequest handling."""
    
    async def test_tool_not_decorated(self):
        """Test calling non-decorated method as tool."""
        
        class RegularMethodAgent(Agent):
            async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
                return AgentOutput(agent_id=self.agent_id, outputs={})
            
            def regular_method(self, value: str) -> str:
                """Not decorated as tool."""
                return value.upper()
        
        agent = RegularMethodAgent(agent_name="test", model_name="test", role="test")
        
        from buttermilk._core.tool_definition import UnifiedRequest
        request = UnifiedRequest(
            target="test.regular_method",
            inputs={"value": "test"}
        )
        
        with pytest.raises(ValueError, match="Tool regular_method not found"):
            await agent.handle_unified_request(request)
    
    async def test_tool_with_validation_error(self):
        """Test tool that raises validation error."""
        
        class ValidationErrorAgent(Agent):
            async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
                return AgentOutput(agent_id=self.agent_id, outputs={})
            
            @tool
            def strict_tool(self, count: int) -> dict:
                """Tool with strict validation."""
                if count < 0:
                    raise ValueError("Count must be non-negative")
                return {"count": count}
        
        agent = ValidationErrorAgent(agent_name="test", model_name="test", role="test")
        
        from buttermilk._core.tool_definition import UnifiedRequest
        request = UnifiedRequest(
            target="test.strict_tool",
            inputs={"count": -5}
        )
        
        # The tool itself raises ValueError
        with pytest.raises(ValueError, match="Count must be non-negative"):
            await agent.handle_unified_request(request)
    
    async def test_empty_unified_request(self):
        """Test UnifiedRequest with minimal data."""
        
        class MinimalAgent(Agent):
            async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
                return AgentOutput(
                    agent_id=self.agent_id,
                    outputs={"processed": True}
                )
        
        agent = MinimalAgent(agent_name="test", model_name="test", role="test")
        
        from buttermilk._core.tool_definition import UnifiedRequest
        request = UnifiedRequest(target="test")  # Minimal request
        
        result = await agent.handle_unified_request(request)
        assert result == {"processed": True}


class TestMCPRouteEdgeCases:
    """Test edge cases for MCP route handling."""
    
    def test_mcp_route_without_permissions(self):
        """Test MCPRoute with empty permissions."""
        
        class NoPermissionAgent(Agent):
            async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
                return AgentOutput(agent_id=self.agent_id, outputs={})
            
            @MCPRoute("/public")
            def public_endpoint(self) -> dict:
                """Public endpoint without permissions."""
                return {"status": "ok"}
        
        agent = NoPermissionAgent(agent_name="test", model_name="test", role="test")
        tools = agent.get_tool_definitions()
        
        assert len(tools) == 1
        assert tools[0].permissions == []
        assert tools[0].mcp_route == "/public"
    
    def test_mcp_route_path_variations(self):
        """Test various MCP route path formats."""
        
        class PathVariationAgent(Agent):
            async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
                return AgentOutput(agent_id=self.agent_id, outputs={})
            
            @MCPRoute("simple")  # No leading slash
            def route1(self) -> str:
                return "1"
            
            @MCPRoute("/with/multiple/segments")
            def route2(self) -> str:
                return "2"
            
            @MCPRoute("/with-dashes-and_underscores")
            def route3(self) -> str:
                return "3"
        
        agent = PathVariationAgent(agent_name="test", model_name="test", role="test")
        tools = agent.get_tool_definitions()
        
        assert len(tools) == 3
        
        # Check paths
        paths = [t.mcp_route for t in tools]
        assert "simple" in paths  # Kept as-is
        assert "/with/multiple/segments" in paths
        assert "/with-dashes-and_underscores" in paths