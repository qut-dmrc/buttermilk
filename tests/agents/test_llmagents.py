
import pytest

# Buttermilk core imports
from buttermilk._core.contract import AgentInput, AgentTrace
from buttermilk._core.llms import CHEAP_CHAT_MODELS
from buttermilk._core.types import Record
from buttermilk.agents.judge import AgentReasons, Judge  # Import Judge and its output model

# Specific Agents being tested
from buttermilk.agents.llm import LLMAgent


@pytest.fixture
def request_paris() -> AgentInput:
    return AgentInput(prompt="What is the capital of France?")


@pytest.fixture
def request_chief(fight_no_more_forever) -> AgentInput:
    return AgentInput(records=[fight_no_more_forever])


@pytest.mark.anyio
@pytest.mark.parametrize("model_name", CHEAP_CHAT_MODELS)  # Parametrize over cheap models
async def test_llm_agent_direct_call(model_name: str, request_paris: AgentInput):
    """Test direct invocation of a basic LLMAgent using __call__."""
    agent = LLMAgent(role="tester", name="Basic Assistant", description="Test basic LLM call", parameters={"model": model_name})
    await agent.initialize()  # Ensure agent is initialized

    response = await agent(message=request_paris)  # Use __call__

    assert isinstance(response, AgentTrace)
    assert not response.is_error, f"Agent returned error: {response.error}"
    assert response.outputs, "Agent should produce output"
    # Check if output is string and contains 'Paris' (case-insensitive)
    if isinstance(response.outputs, str):
        assert "paris" in response.outputs.lower()
    elif isinstance(response.outputs, dict):  # Handle cases where output might be dict
        assert "paris" in str(response.outputs).lower()
    else:
        # Weaker assertion if output is neither string nor dict
        assert "paris" in str(response.outputs).lower()


@pytest.mark.anyio
@pytest.mark.parametrize("model_name", CHEAP_CHAT_MODELS)  # Parametrize over cheap models
async def test_judge_agent_process(model_name: str, request_chief: AgentInput, fight_no_more_forever: Record):
    """Test direct invocation of Judge agent's _process method with a record."""
    # Templates
    judge_params = {
        "model": model_name,
        "template": "judge",
        "criteria": "is this content safe or toxic?",  # Example criteria
    }
    agent = Judge(role="testing", name="Test Judge", description="Test judge agent", parameters=judge_params)
    await agent.initialize()  # Ensure agent is initialized

    # Correctly call _process with 'message' keyword arg
    result = await agent._process(message=request_chief)

    assert isinstance(result, AgentTrace), "Result should be AgentTrace"
    assert not result.is_error, f"Judge agent returned error: {result.error}"
    assert result.outputs is not None, "Judge agent should produce outputs"

    # Assert that the output is the expected AgentReasons model
    assert isinstance(result.outputs, AgentReasons), f"Expected AgentReasons output, got {type(result.outputs)}"

    # Check fields within AgentReasons
    assert isinstance(result.outputs.prediction, bool), "'prediction' field should be boolean"
    assert isinstance(result.outputs.reasons, list), "'reasons' field should be a list"
    assert len(result.outputs.reasons) > 0, "'reasons' list should not be empty"
    assert isinstance(result.outputs.confidence, str), "'confidence' field should be string"
    assert result.outputs.confidence in ["high", "medium", "low"], "'confidence' should be high, medium, or low"
    assert isinstance(result.outputs.conclusion, str), "'conclusion' field should be string"

    # Example content check (adapt based on expected behavior for the given record/criteria)
    # This is harder to make deterministic without mocking the LLM.
    # We can check if certain keywords appear, but the exact output varies.
    reasons_text = " ".join(result.outputs.reasons).lower()
    assert "joseph" in reasons_text, "Reasons should mention 'Joseph'"
    # Depending on criteria, check for other keywords like 'surrender', 'fight', etc.
    assert "surrender" in reasons_text or "fight no more" in reasons_text, "Reasons should relate to the speech content"


@pytest.mark.anyio
@pytest.mark.parametrize("model_name", CHEAP_CHAT_MODELS)  # Parametrize over cheap models
async def test_scorer(model_name: str, request_paris: AgentInput):
    """Test direct invocation of a basic LLMAgent using __call__."""
    agent = LLMAgent(role="tester", name="Basic Assistant", description="Test basic LLM call", parameters={"model": model_name})
    # await agent.initialize()  # Ensure agent is initialized

    # response = await agent(message=request_paris)  # Use __call__

    # assert isinstance(response, AgentTrace)
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
