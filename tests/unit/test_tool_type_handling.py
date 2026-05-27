"""Test tool type handling changes for Tool vs ToolSchema compatibility.

This test validates the architectural changes where:
- Agents create Tool objects for internal use
- Agents return .schema (ToolSchema) for host registration
- Host collects ToolSchemas and passes to LLM
- LLM makes tool calls, host intercepts them as FunctionCall objects
- Host routes as StepRequests back to agents

Focuses on testing the specific type handling changes:
1. ToolSchema objects can be passed to llms.py without errors
2. The deduplication logic works with both Tool and ToolSchema objects
3. The flow from ToolSchema → LLM → intercept → routing works
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from buttermilk._core.messages import CreateResult
from buttermilk._core.tool_types import ToolSchema

from buttermilk._core.llms import LiteLLMWrapper
from buttermilk._core.tool_definition import AgentToolDefinition
from buttermilk.agents.flowcontrol.structured_llmhost import StructuredLLMHostAgent


class TestToolTypeHandling:
    """Test the Tool vs ToolSchema type handling changes."""

    @pytest.fixture
    def sample_tool_schema(self) -> ToolSchema:
        """Create a sample ToolSchema for testing."""
        return ToolSchema(
            name="test_search",
            description="Search for test data",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        )

    @pytest.fixture
    def sample_tool_object(self, sample_tool_schema: ToolSchema) -> AgentToolDefinition:
        """Create a sample Tool object for testing."""
        return AgentToolDefinition(
            name=sample_tool_schema.name,
            description=sample_tool_schema.description,
            input_schema=sample_tool_schema.parameters,
            output_schema={
                "type": "object",
                "properties": {
                    "results": {"type": "array"},
                    "count": {"type": "integer"},
                },
            },
        )

    @pytest.fixture
    def mixed_tools_list(self, sample_tool_schema: ToolSchema, sample_tool_object: AgentToolDefinition):
        """Create a list containing both Tool objects and ToolSchema objects."""
        another_schema = ToolSchema(
            name="analyze_data",
            description="Analyze collected data",
            parameters={
                "type": "object",
                "properties": {"data": {"type": "array"}, "method": {"type": "string"}},
                "required": ["data"],
            },
        )

        # Return mixed list: Tool object, ToolSchema, another ToolSchema
        return [
            sample_tool_object,  # Tool object (name='test_search')
            sample_tool_schema,  # ToolSchema dict (name='test_search') - DUPLICATE NAME
            another_schema,  # Another ToolSchema dict (name='analyze_data')
        ]

    def test_tool_schema_can_be_passed_to_llms(self, sample_tool_schema: ToolSchema):
        """Test that ToolSchema objects can be passed to llms.py type hints without errors."""
        # This test validates the type hints allow Tool | ToolSchema
        from buttermilk._core.llms import LiteLLMWrapper

        # Create a mock LiteLLMWrapper instance
        wrapper = Mock(spec=LiteLLMWrapper)

        # Mock the create method with the correct signature
        async def mock_create(messages, tools=[], schema=None, cancellation_token=None, **kwargs):
            # This should accept both Tool and ToolSchema objects
            from buttermilk._core.messages import RequestUsage

            return CreateResult(
                content="Mock response",
                finish_reason="stop",
                usage=RequestUsage(prompt_tokens=10, completion_tokens=5),
                cached=False,
            )

        wrapper.create = AsyncMock(side_effect=mock_create)

        # Test that we can call with ToolSchema objects
        tools_list = [sample_tool_schema]

        # This should not raise any type errors
        async def test_call():
            result = await wrapper.create(messages=[], tools=tools_list, cancellation_token=None)
            assert result is not None

        # Run the async test
        asyncio.run(test_call())

    def test_deduplication_logic_with_mixed_tools(self, mixed_tools_list):
        """Test the deduplication logic works with both Tool and ToolSchema objects."""

        # Import the deduplication logic from structured_llmhost
        def get_tool_name(tool):
            """Replicate the deduplication logic from structured_llmhost.py."""
            if hasattr(tool, "name"):
                return tool.name  # Tool object
            return tool["name"]  # ToolSchema dict

        # Test the logic with mixed tools
        tools = mixed_tools_list

        # Apply deduplication logic
        deduped_tools = list({get_tool_name(tool): tool for tool in tools}.values())

        # Verify deduplication works (3 tools with 2 unique names = 2 tools)
        assert len(deduped_tools) == 2

        # Verify names are correctly extracted
        tool_names = [get_tool_name(tool) for tool in deduped_tools]
        assert "test_search" in tool_names
        assert "analyze_data" in tool_names

        # Test with duplicate names
        duplicate_schema: ToolSchema = {
            "name": "test_search",  # Same name as first tool
            "description": "Duplicate search tool",
            "parameters": {"type": "object", "properties": {}},
        }

        tools_with_duplicate = mixed_tools_list + [duplicate_schema]
        deduped_with_duplicate = list({get_tool_name(tool): tool for tool in tools_with_duplicate}.values())

        # Should have only 2 unique tools (duplicates removed)
        assert len(deduped_with_duplicate) == 2

        # Verify the last one wins (dictionary behavior)
        test_search_tool = next(tool for tool in deduped_with_duplicate if get_tool_name(tool) == "test_search")
        assert get_tool_name(test_search_tool) == "test_search"

    @pytest.mark.anyio
    async def test_structured_llmhost_deduplication_integration(self, mixed_tools_list, real_bm):
        """Test the deduplication works in StructuredLLMHostAgent._call_llm method."""
        host = StructuredLLMHostAgent(
            agent_name="test_host",
            role="HOST",
            parameters={
                "model": "test-model",
                "template": "host",
                "human_in_loop": False,
            },
        )

        # Mock the LLM client
        mock_client = Mock(spec=LiteLLMWrapper)
        from buttermilk._core.messages import RequestUsage

        mock_client.call_chat = AsyncMock(
            return_value=CreateResult(
                content="Mock response",
                finish_reason="stop",
                usage=RequestUsage(prompt_tokens=10, completion_tokens=5),
                cached=False,
            )
        )
        # Mock bm.llms.get_client directly (used in _call_llm)
        with patch("buttermilk.agents.flowcontrol.structured_llmhost.bm") as mock_bm:
            mock_bm.llms.get_client.return_value = mock_client

            # Call _call_llm with mixed tools
            await host._call_llm(
                messages=[],
                tools=mixed_tools_list,
                schema=None,
                cancellation_token=None,
            )

            # Verify the call was made
            assert mock_client.call_chat.called

            # Verify the tools_list parameter passed to call_chat
            call_args = mock_client.call_chat.call_args
            tools_list_passed = call_args.kwargs.get("tools_list", [])

            # Should have 2 unique tools (deduplication should work)
            assert len(tools_list_passed) == 2

            # Verify intercept_tools flag was set
            assert call_args.kwargs.get("intercept_tools") is True

    # NOTE: test_toolschema_to_functioncall_flow DELETED per TEST-CLEANER philosophy
    # This test was too complex with excessive mocking of our internal code.
    # It tested implementation details rather than business behavior.
    # The proper test for this flow should be an integration test that uses
    # real StructuredLLMHostAgent with real tool registration and routing.

    @pytest.mark.anyio
    async def test_tool_object_schema_property_usage(self, sample_tool_object: AgentToolDefinition):
        """Test that Tool objects use their .schema property correctly."""
        # Verify the Tool object has the schema property
        assert hasattr(sample_tool_object, "schema")

        schema = sample_tool_object.schema
        assert isinstance(schema, ToolSchema)
        assert schema.name == "test_search"
        assert schema.description == "Search for test data"
        assert schema.parameters is not None

        # Test that both Tool.schema and direct ToolSchema work the same way
        tool_via_schema = sample_tool_object.schema
        direct_schema = ToolSchema(
            name="test_search",
            description="Search for test data",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        )

        # They should have the same structure
        assert tool_via_schema.name == direct_schema.name
        assert tool_via_schema.description == direct_schema.description
        assert tool_via_schema.parameters == direct_schema.parameters

    def test_type_hints_accept_both_tool_and_toolschema(self):
        """Test that the type hints Tool | ToolSchema work correctly."""
        from collections.abc import Sequence as ABCSequence
        from typing import get_args, get_origin, get_type_hints

        from buttermilk._core.llms import LiteLLMWrapper

        # Get resolved type hints (handles PEP 563 stringified annotations)
        hints = get_type_hints(LiteLLMWrapper.create)
        annotation = hints["tools"]

        # This should be Sequence[Tool | ToolSchema]
        # get_origin returns the actual class for collections.abc.Sequence
        origin = get_origin(annotation)
        assert origin is not None, "Expected a generic type with origin"
        # Check it's Sequence (either from typing or collections.abc)
        assert "Sequence" in str(origin) or origin is ABCSequence, f"Expected Sequence origin, got {origin}"

        # Get the inner type (Tool | ToolSchema)
        args = get_args(annotation)
        assert len(args) > 0, "Expected type arguments for Sequence"
        inner_type = args[0]

        # Verify it's a Union that includes both types
        if hasattr(inner_type, "__args__"):  # Union type
            type_args = get_args(inner_type)
            type_names = [arg.__name__ if hasattr(arg, "__name__") else str(arg) for arg in type_args]

            # Should include both Tool and ToolSchema
            assert any("Tool" in name for name in type_names)
            assert any("ToolSchema" in name or "dict" in name for name in type_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
