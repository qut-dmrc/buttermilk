"""Integration tests for refactored LLMAgent using LLMCore."""

from unittest.mock import patch

import pytest
from autogen_core.models import SystemMessage, UserMessage
from pydantic import BaseModel

from buttermilk._core.contract import AgentInput, AgentOutput
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.llm_core import LLMResult
from buttermilk._core.types import Record
from buttermilk.agents.llm import LLMAgent


class OutputForTesting(BaseModel):
    """Test structured output model."""

    action: str
    reason: str


class TestLLMAgentRefactoring:
    """Integration tests to verify LLMAgent works correctly with LLMCore."""

    def test_llmagent_initializes_llmcore(self):
        """Test that LLMAgent properly initializes LLMCore."""
        agent = LLMAgent(agent_name="test_agent", role="TESTER", parameters={"model": "gpt-4", "template": "test_template", "temperature": 0.5})

        # Verify LLMCore is initialized
        assert hasattr(agent, "llm_core")
        assert agent.llm_core.model == "gpt-4"
        assert agent.llm_core.template == "test_template"
        assert agent.llm_core.parameters["temperature"] == 0.5

    def test_llmagent_with_output_model(self):
        """Test LLMAgent with structured output model."""
        agent = LLMAgent(
            agent_name="structured_agent", role="ANALYZER", parameters={"model": "claude-3", "template": "analyze"}, output_model=OutputForTesting
        )

        assert agent.output_model == OutputForTesting
        assert agent.llm_core.output_model == OutputForTesting

    def test_llmagent_passes_tools_to_llmcore(self):
        """Test that LLMAgent passes tools to LLMCore."""
        from buttermilk._core.config import ToolConfig

        # AgentConfig expects tools as a dict with ToolConfig values
        tool_config = ToolConfig(
            description="Test tool that doubles a number",
            tool_obj="test_function",  # Reference to the function
        )

        agent = LLMAgent(
            agent_name="tool_agent",
            role="TOOL_USER",
            parameters={"model": "gpt-4", "template": "test"},
            tools={"test_tool": tool_config},  # Pass as dict with ToolConfig
        )

        # The agent should have tools configuration
        assert hasattr(agent, "_tools")
        # LLMCore should receive the tools
        assert agent.llm_core.tools is not None

    @pytest.mark.asyncio
    async def test_llmagent_process_basic_input(self):
        """Test LLMAgent processes basic AgentInput correctly."""
        agent = LLMAgent(agent_name="basic_agent", role="PROCESSOR", parameters={"model": "gpt-4", "template": "test"})

        # Create test input
        agent_input = AgentInput(inputs={"text": "Process this text"}, parameters={"extra_param": "value"}, context=[], records=[])

        # Mock LLMCore processing
        mock_result = LLMResult(
            content="Processed output",
            metadata={
                "model": "gpt-4",
                "usage": {"total_tokens": 100},
                "finish_reason": "stop",
                "template": {"template_name": "test", "template_hash": "abc123"},
            },
            trace_id="trace-001",
        )

        with patch.object(agent.llm_core, "process_with_llm") as mock_process:
            mock_process.return_value = mock_result

            # Process the input
            result = await agent._process(message=agent_input)

            # Verify AgentOutput structure
            assert isinstance(result, AgentOutput)
            assert result.agent_id == agent.agent_id
            assert result.outputs == "Processed output"
            assert result.error == []  # error field defaults to empty list

            # Verify metadata includes both agent and LLM info
            assert result.metadata["agent_name"] == agent.agent_name
            assert result.metadata["agent_id"] == agent.agent_id
            assert result.metadata["agent_model"] == "gpt-4"
            assert result.metadata["usage"]["total_tokens"] == 100
            assert result.metadata["template_name"] == "test"

            # Verify LLMCore was called correctly
            mock_process.assert_called_once()
            call_args = mock_process.call_args
            assert call_args[1]["inputs"] == {"text": "Process this text"}
            assert call_args[1]["context"] == []
            assert call_args[1]["records"] == []

    @pytest.mark.asyncio
    async def test_llmagent_process_with_context(self):
        """Test LLMAgent processes context correctly."""
        agent = LLMAgent(agent_name="context_agent", role="CONTEXTUAL", parameters={"model": "gpt-4", "template": "chat"})

        # Create input with context
        context = [SystemMessage(content="You are a helpful assistant", source="test"), UserMessage(content="Previous message", source="test")]

        agent_input = AgentInput(inputs={"prompt": "Current question"}, context=context)

        mock_result = LLMResult(content="Response with context", metadata={})

        with patch.object(agent.llm_core, "process_with_llm") as mock_process:
            mock_process.return_value = mock_result

            await agent._process(message=agent_input)

            # Verify context was passed to LLMCore
            call_args = mock_process.call_args
            assert call_args[1]["context"] == context

    @pytest.mark.asyncio
    async def test_llmagent_process_with_records(self):
        """Test LLMAgent processes records correctly."""
        agent = LLMAgent(agent_name="record_agent", role="RECORD_PROCESSOR", parameters={"model": "gpt-4", "template": "process_records"})

        # Create test records
        record = Record(content="Record 1")

        agent_input = AgentInput(inputs={}, record=record)

        mock_result = LLMResult(content="Processed records", metadata={})

        with patch.object(agent.llm_core, "process_with_llm") as mock_process:
            mock_process.return_value = mock_result

            await agent._process(message=agent_input)

            # Verify records were passed to LLMCore
            call_args = mock_process.call_args
            assert call_args[1]["record"] == record

    @pytest.mark.asyncio
    async def test_llmagent_process_structured_output(self):
        """Test LLMAgent with structured output from LLMCore."""
        agent = LLMAgent(
            agent_name="structured_agent", role="ANALYZER", parameters={"model": "gpt-4", "template": "analyze"}, output_model=OutputForTesting
        )

        agent_input = AgentInput(inputs={"situation": "Test scenario"})

        # Mock structured output from LLMCore
        structured_obj = OutputForTesting(action="proceed", reason="All checks passed")
        mock_result = LLMResult(content=structured_obj, metadata={"model": "gpt-4", "template": {"template_name": "analyze"}})

        with patch.object(agent.llm_core, "process_with_llm") as mock_process:
            mock_process.return_value = mock_result

            result = await agent._process(message=agent_input)

            # Verify structured output is preserved
            assert isinstance(result.outputs, OutputForTesting)
            assert result.outputs.action == "proceed"
            assert result.outputs.reason == "All checks passed"

    @pytest.mark.asyncio
    async def test_llmagent_parameter_merging(self):
        """Test that task parameters properly override agent parameters."""
        agent = LLMAgent(
            agent_name="param_agent",
            role="FLEXIBLE",
            parameters={"model": "gpt-4", "template": "default", "temperature": 0.7, "custom_param": "agent_value"},
        )

        agent_input = AgentInput(
            inputs={"text": "test"},
            parameters={
                "temperature": 0.9,  # Override
                "custom_param": "task_value",  # Override
            },
        )

        mock_result = LLMResult(content="Result", metadata={}, template_metadata={})

        with patch.object(agent.llm_core, "process_with_llm") as mock_process:
            mock_process.return_value = mock_result

            await agent._process(message=agent_input)

            # Verify merged parameters were set on LLMCore
            assert agent.llm_core.parameters["temperature"] == 0.9
            assert agent.llm_core.parameters["custom_param"] == "task_value"
            assert agent.llm_core.parameters["model"] == "gpt-4"  # Not overridden

    @pytest.mark.asyncio
    async def test_llmagent_handles_processing_error(self):
        """Test LLMAgent handles ProcessingError from LLMCore."""
        agent = LLMAgent(agent_name="error_agent", role="ERROR_HANDLER", parameters={"model": "gpt-4", "template": "test"})

        agent_input = AgentInput(inputs={"text": "test"})

        with patch.object(agent.llm_core, "process_with_llm") as mock_process:
            mock_process.side_effect = ProcessingError("Template not found")

            with pytest.raises(ProcessingError, match="Template not found"):
                await agent._process(message=agent_input)

    @pytest.mark.asyncio
    async def test_llmagent_handles_unexpected_error(self):
        """Test LLMAgent wraps unexpected errors from LLMCore."""
        agent = LLMAgent(agent_name="error_agent", role="ERROR_HANDLER", parameters={"model": "gpt-4", "template": "test"})

        agent_input = AgentInput(inputs={"text": "test"})

        with patch.object(agent.llm_core, "process_with_llm") as mock_process:
            mock_process.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(ProcessingError, match="Unexpected error in agent"):
                await agent._process(message=agent_input)

    @pytest.mark.asyncio
    async def test_llmagent_preserves_parent_trace_id(self):
        """Test that parent trace ID is passed through to LLMCore."""
        agent = LLMAgent(agent_name="trace_agent", role="TRACER", parameters={"model": "gpt-4", "template": "test"})

        agent_input = AgentInput(inputs={"text": "test"}, parent_call_id="parent-trace-xyz")

        mock_result = LLMResult(content="Result", metadata={}, template_metadata={})

        with patch.object(agent.llm_core, "process_with_llm") as mock_process:
            mock_process.return_value = mock_result

            await agent._process(message=agent_input)

            # Verify parent trace ID was passed
            call_args = mock_process.call_args
            assert call_args[1]["parent_trace_id"] == "parent-trace-xyz"

    @pytest.mark.asyncio
    async def test_llmagent_template_metadata_preserved(self):
        """Test that template metadata is preserved for ExecutionTrace."""
        agent = LLMAgent(agent_name="metadata_agent", role="METADATA", parameters={"model": "gpt-4", "template": "test"})

        agent_input = AgentInput(inputs={"text": "test"})

        mock_result = LLMResult(content="Result", metadata={"template": {"template_name": "test", "template_hash": "hash123", "unfilled_vars": []}})

        with patch.object(agent.llm_core, "process_with_llm") as mock_process:
            mock_process.return_value = mock_result

            result = await agent._process(message=agent_input)

            # Verify template metadata is in agent's internal state
            assert agent.template_metadata == mock_result.metadata.get("template")

            # Verify it's included in output metadata
            assert result.metadata["template_name"] == "test"
            assert result.metadata["template_hash"] == "hash123"
