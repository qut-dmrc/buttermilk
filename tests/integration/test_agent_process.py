"""
Integration tests for individual agent processes.

This module provides utilities and test cases to run full examples of individual agent processes 
in isolation, ensuring they work as expected and identifying common failure patterns.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogen_core import DefaultTopicId, MessageContext, CancellationToken
import weave

from buttermilk._core.agent import Agent
from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
    StepRequest,
    ConductorRequest,
    GroupchatMessageTypes,
)
from buttermilk._core.types import Record
from buttermilk.agents.evaluators.scorer import LLMScorer, QualScore, QualScoreCRA
from buttermilk.agents.flowcontrol.sequencer import Sequencer
from buttermilk.agents.flowcontrol.host import LLMHostAgent
from buttermilk.libs.autogen import AutogenAgentAdapter
from buttermilk.bm import bm, logger

# Try to import LLMJudge, but don't fail if it doesn't exist
try:
    from buttermilk.agents.judge import AgentReasons, LLMJudge
except ImportError:
    # Create placeholder for tests if not available
    AgentReasons = MagicMock()
    LLMJudge = MagicMock()


class AgentIntegrationTest:
    """Base class for testing individual agent processes."""
    
    def __init__(self, agent_cls, agent_config, adapter_topic="test_topic"):
        """
        Initialize the test harness for an agent.
        
        Args:
            agent_cls: The agent class to test
            agent_config: Configuration for the agent
            adapter_topic: Topic for the AutogenAgentAdapter
        """
        self.agent_cls = agent_cls
        self.agent_config = agent_config
        self.adapter_topic = adapter_topic
        self.adapter = None
        
    async def setup(self):
        """Set up the agent with its adapter."""
        self.adapter = AutogenAgentAdapter(
            topic_type=self.adapter_topic,
            agent_cls=self.agent_cls,
            agent_cfg=self.agent_config,
        )
        await self.adapter.agent.initialize()
        
    async def direct_invoke(self, message, source="test_source"):
        """
        Directly invoke the agent's __call__ method, bypassing the adapter.
        
        This is useful for testing the core agent functionality without the adapter layer.
        
        Args:
            message: The input message (AgentInput, ConductorRequest, etc.)
            source: Source identifier string
            
        Returns:
            The agent's output
        """
        if not self.adapter:
            raise ValueError("Agent adapter not initialized. Call setup() first.")
            
        return await self.adapter.agent(
            message=message,
            cancellation_token=None,
            public_callback=AsyncMock(),
            message_callback=AsyncMock(),
            source=source
        )
        
    async def listen_with(self, message, source="test_source"):
        """
        Test the agent's _listen method directly.
        
        Args:
            message: The input message
            source: Source identifier string
        """
        if not self.adapter:
            raise ValueError("Agent adapter not initialized. Call setup() first.")
            
        await self.adapter.agent._listen(
            message=message,
            cancellation_token=None,
            public_callback=AsyncMock(),
            message_callback=AsyncMock(),
            source=source
        )
        
    async def adapter_invoke(self, message):
        """
        Invoke the agent through its adapter.
        
        Args:
            message: The input message
            
        Returns:
            The adapter's output
        """
        if not self.adapter:
            raise ValueError("Agent adapter not initialized. Call setup() first.")
            
        mock_context = MagicMock(spec=MessageContext)
        mock_context.cancellation_token = None
        mock_context.sender = "test_sender/instance_id"
        mock_context.topic_id = "test_topic_id"
        
        if isinstance(message, AgentInput):
            return await self.adapter.handle_invocation(message, mock_context)
        elif isinstance(message, ConductorRequest):
            return await self.adapter.handle_conductor_request(message, mock_context)
        else:
            raise ValueError(f"Unsupported message type for adapter_invoke: {type(message)}")
        
    async def cleanup(self):
        """Clean up any resources."""
        # Nothing to clean up by default
        pass


# Mock fixtures for testing
@pytest.fixture
def mock_weave():
    """Mock weave for all tests."""
    with patch("buttermilk.bm.bm.weave", new_callable=MagicMock) as mock_weave:
        mock_call = MagicMock()
        mock_call.id = "mock_call_id"
        mock_call.ref = "mock_ref"
        mock_call.apply_scorer = AsyncMock(name="apply_scorer")
        mock_weave.get_call.return_value = mock_call
        with patch("weave.Scorer", MagicMock):
            with patch("weave.apply_scorer", AsyncMock()):
                with patch("weave.op", lambda func=None: func):
                    yield mock_weave


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for tests."""
    with patch("buttermilk.bm.bm.llms.get_autogen_chat_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="Mock LLM response")
        mock_get_client.return_value = mock_client
        yield mock_client


@pytest.mark.anyio
class TestScorerAgent:
    """Integration tests for the LLMScorer agent."""
    
    @pytest.fixture
    async def scorer_test(self, mock_weave, mock_llm_client):
        """Fixture for a scorer agent test harness."""
        config = AgentConfig(
            role="SCORER", 
            name="Test Scorer", 
            description="Test scorer agent",
            parameters={
                "model": "mock-model",
                "template": "score"
            }
        )
        test_harness = AgentIntegrationTest(LLMScorer, config)
        await test_harness.setup()
        
        # Mock the _extract_vars method to return test data
        test_harness.adapter.agent._extract_vars = AsyncMock()
        test_harness.adapter.agent._extract_vars.return_value = {
            "expected": "Expected answer",
            "records": [Record(content="Test content", data={"ground_truth": "Expected answer"})],
            "assessor": "scorer-test",
            "answers": [
                {
                    "agent_id": "judge-abc",
                    "agent_name": "Test Judge",
                    "answer_id": "test123"
                }
            ]
        }
        
        # Mock _process to return a valid score
        test_harness.adapter.agent._process = AsyncMock()
        test_harness.adapter.agent._process.return_value = AgentOutput(
            agent_info="scorer-test",
            outputs=QualScore(assessments=[
                QualScoreCRA(correct=True, feedback="This is correct")
            ])
        )
        
        yield test_harness
        await test_harness.cleanup()
    
    async def test_scorer_direct_invoke(self, scorer_test):
        """Test directly invoking the scorer agent."""
        input_msg = AgentInput(
            inputs={"key": "value"},
            prompt="Score this response"
        )
        
        # Call the agent directly
        result = await scorer_test.direct_invoke(input_msg)
        
        # Verify the agent's _process method was called
        scorer_test.adapter.agent._process.assert_called_once()
        assert isinstance(result, AgentOutput)
        assert not result.is_error
        
    async def test_scorer_listen_with_valid_message(self, scorer_test):
        """Test the scorer's _listen method with a valid message."""
        # Setup a valid judge output
        judge_output = AgentOutput(
            agent_info="judge-abc",
            role="JUDGE",
            outputs=AgentReasons(
                conclusion="Judge conclusion",
                prediction=True,
                reasons=["Judge reason"],
                confidence="high"
            ),
            records=[Record(content="Test", data={"ground_truth": "Expected"})],
            tracing={"weave": "mock_trace_id"}
        )
        
        # Create a callback for testing
        callback = AsyncMock()
        
        # Override the agent's public callback to test it
        scorer_test.adapter.agent._listen = AsyncMock(wraps=scorer_test.adapter.agent._listen)
        
        # Test listening
        await scorer_test.listen_with(judge_output, source="judge-abc")
        
        # Check extract_vars was called
        scorer_test.adapter.agent._extract_vars.assert_called_once()
        
        # Check that weave.get_call was used with the trace ID
        bm.weave.get_call.assert_called_once_with("mock_trace_id")


@pytest.mark.anyio
class TestDifferentiatorAgent:
    """
    Integration tests for the differentiator agent.
    
    This specifically tests the agent with regard to the BaseModel __private_attributes__
    error seen in the logs.
    """
    
    async def test_pydantic_model_compatibility(self):
        """Test compatibility of Pydantic models to catch attribute errors."""
        # This tests the compatibility of Pydantic models that could have issues with
        # __private_attributes__ which appeared in the error logs
        
        from pydantic import BaseModel, Field, PrivateAttr
        
        # Create a model that uses private attributes and verify compatibility
        class TestModel(BaseModel):
            field1: str = Field(default="test")
            _private: str = PrivateAttr(default="private")
            
        model = TestModel()
        
        # With Pydantic v1, we should have __fields__
        # With Pydantic v2, we should have __pydantic_fields__
        assert hasattr(model, "__pydantic_fields__") or hasattr(model, "__fields__")
        
        # Verify we can access private attributes correctly
        assert model._private == "private"
        
        # Try to use model_dump (v2) or dict (v1)
        data = None
        if hasattr(model, "model_dump"):
            data = model.model_dump()
        else:
            data = model.dict()
            
        assert "field1" in data
        assert "_private" not in data  # Private attributes should not be in the output dict


@pytest.mark.anyio
class TestAPIRateLimit:
    """Tests for handling API rate limits appropriately."""
    
    @pytest.fixture
    async def rate_limited_client(self):
        """Fixture that returns an LLM client that simulates rate limits."""
        with patch("buttermilk.bm.bm.llms.get_autogen_chat_client") as mock_get_client:
            mock_client = MagicMock()
            
            # Have the mock client raise a rate limit error
            rate_limit_error = Exception(
                "Error code: 429 - [{{'error': {{'code': 429, 'message': 'You exceeded your current quota'}}}}]"
            )
            # Add status_code as a dynamic attribute (this is common in HTTP exception objects)
            rate_limit_error.__dict__["status_code"] = 429
            
            # First call raises error, second succeeds
            mock_client.complete = AsyncMock(side_effect=[rate_limit_error, "Success after retry"])
            mock_get_client.return_value = mock_client
            yield mock_client
            
    async def test_llm_rate_limit_handling(self, rate_limited_client):
        """Test that the system properly handles LLM rate limits."""
        from buttermilk._core.retry import before_sleep_log
                
        with patch("buttermilk._core.retry.before_sleep_log", AsyncMock()) as mock_log:
            # Create an LLM agent with retries
            from buttermilk.agents.llm import LLMAgent
            
            config = AgentConfig(
                role="TEST", 
                name="Rate Test", 
                description="Test rate limits",
                parameters={
                    "model": "test-model",
                    "retry_attempts": 3,  # Ensure it will retry
                    "retry_delay": 0.1    # Small delay for test
                }
            )
            
            # Create agent but patch the retry mechanism for testing
            with patch("buttermilk._core.retry.AsyncRetrying") as mock_retry:
                # Set up the mock retry to actually call the function once with error, once successfully
                mock_retry_instance = MagicMock()
                
                # This will run the decorated function through our controlled retry sequence
                mock_retry_instance.__aiter__.return_value = [0, 1]  # Two attempts
                mock_retry.return_value = mock_retry_instance
                
                agent = LLMAgent(**config.model_dump())
                await agent.initialize()
                
                # Create an input for the agent
                input_msg = AgentInput(prompt="Test prompt")
                
                # Call the agent directly
                result = await agent(
                    message=input_msg,
                    cancellation_token=None,
                    public_callback=AsyncMock(),
                    message_callback=AsyncMock(),
                    source="test"
                )
                
                # Verify the retry mechanism was properly configured
                mock_retry.assert_called_once()
                
                # Check that we logged the retry
                mock_log.assert_called()
                
                # Verify we eventually got a successful result
                assert result is not None
                assert not getattr(result, "is_error", True)


@pytest.mark.anyio
class TestAutogenGroupChat:
    """
    Test the full autogen orchestrator with mocked components.
    
    This validates the groupchat functionality that's used in batch.yaml.
    """
    
    @pytest.fixture
    def mock_runtime(self):
        """Mock the autogen SingleThreadedAgentRuntime."""
        with patch("buttermilk.runner.groupchat.SingleThreadedAgentRuntime") as mock_runtime_class:
            mock_runtime = MagicMock()
            mock_runtime.add_subscription = AsyncMock()
            mock_runtime.get = AsyncMock(return_value="agent_instance_id")
            mock_runtime.send_message = AsyncMock(return_value="response")
            mock_runtime.publish_message = AsyncMock()
            mock_runtime.start = MagicMock()
            mock_runtime.stop_when_idle = AsyncMock()
            mock_runtime._run_context = True  # To simulate a started runtime
            mock_runtime_class.return_value = mock_runtime
            yield mock_runtime
    
    @pytest.fixture
    async def mock_orchestrator(self, mock_runtime, mock_weave, mock_llm_client):
        """Create a mocked AutogenOrchestrator with proper agent variants."""
        from buttermilk.runner.groupchat import AutogenOrchestrator
        from buttermilk._core.agent import AgentVariants
        
        # Creating proper AgentVariants objects for each agent type
        host_variant = AgentVariants(
            role="host",
            name="Test Host",
            description="Test host",
            agent_obj="Sequencer"
        )
        
        judge_variant = AgentVariants(
            role="judge",
            name="Test Judge",
            description="Test judge",
            agent_obj="LLMJudge",
            parameters={"model": "test-model"}
        )
        
        scorer_variant = AgentVariants(
            role="scorer",
            name="Test Scorer",
            description="Test scorer",
            agent_obj="LLMScorer",
            parameters={"model": "test-model", "template": "score"}
        )
        
        # Create orchestrator with proper variant objects
        orchestrator = AutogenOrchestrator(
            name="test",
            description="Test orchestrator",
            agents={
                "host": host_variant,
                "judge": judge_variant,
                "scorer": scorer_variant
            },
            tools={},
            parameters={"task": "Test task"},
            data=[]
        )
        
        # Mock agent registration
        with patch.object(AutogenAgentAdapter, "register", new_callable=AsyncMock) as mock_register:
            mock_register.return_value = "agent_type_id"
            await orchestrator._setup()
            
        # Mock other methods to avoid real execution
        orchestrator._execute_step = AsyncMock()
        orchestrator._get_host_suggestion = AsyncMock()
        orchestrator._get_host_suggestion.return_value = StepRequest(
            role="JUDGE", 
            prompt="Test prompt",
            description="Test step"
        )
        
        # To end the run loop after one iteration
        def side_effect():
            orchestrator._get_host_suggestion.return_value = StepRequest(
                role="END",
                prompt="Done",
                description="End flow"
            )
            
        orchestrator._execute_step.side_effect = side_effect
        
        yield orchestrator
        await orchestrator._cleanup()
    
    async def test_orchestrator_run(self, mock_orchestrator):
        """Test that the orchestrator can run without errors."""
        from buttermilk._core.contract import RunRequest
        
        # Create a run request
        request = RunRequest(
            record_id="test_id",
            prompt="Test prompt"
        )
        
        # Patch FetchRecord to avoid actual fetching
        with patch("buttermilk.agents.fetch.FetchRecord") as mock_fetch_cls:
            mock_fetch = MagicMock()
            mock_fetch._run = AsyncMock()
            mock_fetch._run.return_value = MagicMock(results=[Record(content="Test")])
            mock_fetch_cls.return_value = mock_fetch
            
            # Run the orchestrator with our mocks
            await mock_orchestrator._run(request)
            
            # Verify the FetchRecord was called
            mock_fetch._run.assert_called_once()
            
            # Verify we asked the host for suggestions
            mock_orchestrator._get_host_suggestion.assert_called()
            
            # Verify we executed steps
            mock_orchestrator._execute_step.assert_called_once()


async def run_single_agent_test(agent_cls, agent_config, input_message):
    """
    Utility function to run a standalone test for a single agent.
    
    This can be used from the command line to test individual agents.
    
    Args:
        agent_cls: The agent class to test
        agent_config: Configuration for the agent
        input_message: The input message to send to the agent
        
    Returns:
        The agent's output
    """
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create the test harness
    test_harness = AgentIntegrationTest(agent_cls, agent_config)
    await test_harness.setup()
    
    try:
        # Call the agent
        result = await test_harness.direct_invoke(input_message)
        return result
    finally:
        await test_harness.cleanup()
