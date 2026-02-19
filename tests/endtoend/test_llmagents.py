import pytest

# Buttermilk core imports
from buttermilk._core.contract import AgentInput, AgentOutput, ExecutionTrace
from buttermilk._core.llms import CHAT_MODELS, CHEAP_CHAT_MODELS
from buttermilk._core.types import Record
from buttermilk.agents.judge import (
    Judge,
    JudgeReasons,
)  # Import Judge and its output model

# Specific Agents being tested
from buttermilk.agents.llm import LLMAgent


@pytest.fixture
def request_paris() -> AgentInput:
    return AgentInput(inputs={"prompt": "What is the capital of France?"})


@pytest.fixture
def request_chief(fight_no_more_forever) -> AgentInput:
    return AgentInput(record=fight_no_more_forever)


@pytest.mark.anyio
@pytest.mark.parametrize("model_name", CHEAP_CHAT_MODELS)  # Parametrize over cheap models
async def test_llm_agent_direct_call(real_bm, model_name: str, request_paris: AgentInput):
    """Test direct invocation of a basic LLMAgent using ._process()."""
    agent = LLMAgent(
        role="tester",
        name="Basic Assistant",
        description="Test basic LLM call",
        parameters={"model": model_name, "template": "ra"},
    )

    response = await agent._process(message=request_paris)

    # _process returns AgentOutput, not ExecutionTrace
    assert isinstance(response, AgentOutput)
    assert not response.is_error, f"Agent returned error: {response.error}"
    assert response.outputs, "Agent should produce output"
    # Check if output is string and contains 'Paris' (case-insensitive)
    if isinstance(response.outputs, str):
        assert "paris" in response.outputs.lower()
    elif isinstance(response.outputs, dict):
        # Handle cases where output might be dict
        assert "paris" in str(response.outputs).lower()
    else:
        # Weaker assertion if output is neither string nor dict
        assert "paris" in str(response.outputs).lower()


@pytest.mark.anyio
@pytest.mark.parametrize("model_name", CHAT_MODELS)
async def test_judge_agent_process(real_bm, model_name: str, request_chief: AgentInput, fight_no_more_forever: Record):
    """Test direct invocation of Judge agent's _process method with a record."""
    # Templates
    judge_params = {
        "model": model_name,
        "template": "judge",
        "criteria": "is this content safe or toxic?",  # Example criteria
    }
    agent = Judge(
        role="testing",
        name="Test Judge",
        description="Test judge agent",
        parameters=judge_params,
    )

    # Correctly call _process with 'message' keyword arg
    result = await agent._process(message=request_chief)

    assert isinstance(result, AgentOutput), "Result should be AgentOutput"
    assert not result.is_error, f"Judge agent returned error: {result.error}"
    assert result.outputs is not None, "Judge agent should produce outputs"

    # Assert that the output is the expected Reasons model
    assert isinstance(result.outputs, JudgeReasons), f"Expected JudgeReasons output, got {type(result.outputs)}"

    # Check fields within Reasons
    assert isinstance(result.outputs.prediction, bool), "'prediction' field should be boolean"
    assert isinstance(result.outputs.reasons, list), "'reasons' field should be a list"
    assert len(result.outputs.reasons) > 0, "'reasons' list should not be empty"
    assert isinstance(result.outputs.conclusion, str), "'conclusion' field should be string"

    # Example content check (adapt based on expected behavior for the given record/criteria)
    # This is harder to make deterministic without mocking the LLM.
    # We can check if certain keywords appear, but the exact output varies.
    reasons_text = " ".join(result.outputs.reasons).lower()
    # Depending on criteria, check for keywords like 'surrender', 'fight', etc.
    assert "surrender" in reasons_text or "fight" in reasons_text, "Reasons should relate to the speech content"


@pytest.mark.anyio
@pytest.mark.parametrize("model_name", CHEAP_CHAT_MODELS)  # Parametrize over cheap models
async def test_scorer(real_bm, model_name: str, request_paris: AgentInput):
    """Test direct invocation of a basic LLMAgent using __call__."""
    LLMAgent(
        role="tester",
        name="Basic Assistant",
        description="Test basic LLM call",
        parameters={"model": model_name, "template": "simple"},
    )

    # response = await agent(message=request_paris)  # Use __call__

    # assert isinstance(response, ExecutionTrace)
    # assert not response.is_error, f"Agent returned error: {response.error}"
    # assert response.outputs, "Agent should produce output"
    # # Check if output is string and contains 'Paris' (case-insensitive)
    # if isinstance(response.outputs, str):
    #     assert "paris" in response.outputs.lower()
    # elif isinstance(response.outputs, dict):  # Handle cases where output might be dict
    #     assert "paris" in str(response.outputs).lower()
    # else:
    #     # Weaker assertion if output is neither string nor dict
    #     assert "paris" in str(response.outputs).lower()


@pytest.mark.anyio
@pytest.mark.parametrize("model_name", CHEAP_CHAT_MODELS)
async def test_llm_agent_template_metadata(real_bm, model_name: str):
    """Test that LLMAgent includes template metadata in AgentOutput."""
    # The 'simple' template expects {{var}}, not {{prompt}}
    request = AgentInput(inputs={"var": "What is the capital of France?"})

    agent = LLMAgent(
        role="tester",
        name="Template Test Agent",
        description="Test template metadata tracking",
        parameters={"model": model_name, "template": "simple"},
    )

    # Call the agent to get the output
    result = await agent._process(message=request)

    # Verify the result is an AgentOutput
    assert isinstance(result, AgentOutput), "Result should be AgentOutput"
    assert not result.is_error, f"Agent returned error: {result.error}"

    # Check that template metadata is included (nested under "template" key)
    assert "template" in result.metadata, "Template metadata should be in metadata"
    template_meta = result.metadata["template"]
    assert "template_name" in template_meta, "Template name should be in template metadata"
    assert "template_hash" in template_meta, "Template hash should be in template metadata"

    # Verify the template metadata values
    assert template_meta["template_name"] == "simple", "Template name should match"
    assert isinstance(template_meta["template_hash"], str), "Template hash should be string"
    # Template hash is a raw SHA256 hex string (64 chars, no prefix)
    assert len(template_meta["template_hash"]) == 64, "Template hash should be 64 hex chars (SHA256)"

    # Test that ExecutionTrace also includes the metadata when created from output
    agent_info = {
        "name": "test_agent",
        "role": "TESTER",
        "instructions": "Test instructions",
    }

    trace = ExecutionTrace.from_output(output=result, inputs=request, agent_info=agent_info)

    # Verify the trace includes the template metadata (also nested under "template")
    assert isinstance(trace, ExecutionTrace), "Trace should be ExecutionTrace"
    assert "template" in trace.metadata, "Template metadata should be in ExecutionTrace metadata"
    trace_template_meta = trace.metadata["template"]
    assert "template_name" in trace_template_meta, "Template name should be in ExecutionTrace template metadata"
    assert "template_hash" in trace_template_meta, "Template hash should be in ExecutionTrace template metadata"
    assert trace_template_meta["template_name"] == "simple", "Template name should match in trace"
    assert trace_template_meta["template_hash"] == template_meta["template_hash"], "Template hash should match between output and trace"
