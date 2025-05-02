"""Tests the LLMScorer agent, potentially in a simulated flow context.
"""

import json
from typing import Any  # For fixture type hint

import pytest

# Buttermilk core types
from buttermilk._core.contract import AgentInput, AgentTrace
from buttermilk._core.llms import CHEAP_CHAT_MODELS  # Use cheaper models for testing
from buttermilk.agents.evaluators.scorer import LLMScorer, QualScore, QualScoreCRA  # Scorer and its output models

# Agent classes and models


@pytest.fixture(params=CHEAP_CHAT_MODELS)  # Parametrize over cheap models
def scorer_agent(request) -> LLMScorer:
    """Fixture to create an LLMScorer instance."""
    # Instantiate LLMScorer correctly
    return LLMScorer(
        role="scorer",  # Correct role
        name="Test Scorer Agent",
        description="Scorer test agent",
        parameters={
            "template": "score",  # Assumes 'score.jinja2' exists and is appropriate
            "model": request.param,  # Use parametrized model
            "criteria": "criteria_ordinary",  # Example criteria context for the template
            # "formatting": "json_rules", # LLMAgent usually handles JSON output if _output_model is set
        },
        # Inputs mapping might be needed depending on the 'score' template
        inputs={
            "answer": "judge.outputs",  # Example: map judge output to 'answer' template var
            "expected": "ground_truth",  # Example: map ground truth to 'expected' template var
        },
    )


@pytest.fixture
def judge_output_fixture() -> dict[str, Any]:
    """Fixture providing a sample dictionary representing a serialized AgentTrace
    from a Judge agent, including ground truth in nested records.
    """
    # Note: Using a dictionary derived from JSON is okay for fixtures,
    # but ideally, we'd construct Pydantic models directly if possible.
    json_str = """
    {
        "role": "judge",
        "error": [],
        "metadata": {
            "finish_reason": "stop",
            "usage": { "prompt_tokens": 1905, "completion_tokens": 244 }
        },
        "inputs": {
            "role": "judge",
            "error": [],
            "metadata": {},
            "inputs": {},
            "parameters": {
                "template": "judge",
                "model": "gemini2flashlite",
                "criteria": "criteria_ordinary",
                "formatting": "json_rules"
            },
            "context": [],
            "records": [
                {
                    "record_id": "8YxHsqsrdKQG5VweBp7hYY",
                    "metadata": { "title": "fight_no_more_forever" },
                    "alt_text": null,
                    "ground_truth": "This is a classic example of counterspeech, where Chief Joseph laments the atrocities committed by British-Americans against Native American peoples.",
                    "uri": null,
                    "content": "Tell General Howard I know his heart. What he told me before, I have it in my heart. I am tired of fighting. Our Chiefs are killed; Looking Glass is dead, Ta Hool Hool Shute is dead. The old men are all dead. It is the young men who say yes or no. He who led on the young men is dead. It is cold, and we have no blankets; the little children are freezing to death. My people, some of them, have run away to the hills, and have no blankets, no food. No one knows where they are - perhaps freezing to death. I want to have time to look for my children, and see how many of them I can find. Maybe I shall find them among the dead. Hear me, my Chiefs! I am tired; my heart is sick and sad. From where the sun now stands I will fight no more forever.",
                    "mime": "text/plain"
                }
            ],
            "prompt": ""
        },
        "outputs": {
            "reasons": [
                "The content appears to be a transcription of a speech given by Chief Joseph...",
                "RULE 1: The content is not directed at a marginalized group...",
                "RULE 2: The content does not originate from a position of power or privilege...",
                "RULE 3: The content does not subordinate or treat the group as inferior..."
            ],
            "prediction": false,
            "confidence": "high",
            "conclusion": "The speech reflects on hardship but doesn't violate the criteria."
        }
    }
    """
    # Minor correction: AgentReasons doesn't have severity/labels by default
    # Also, Judge output might be directly AgentReasons, not nested under 'outputs'
    # Adjusting the fixture slightly based on documented Judge Agent
    data = json.loads(json_str)
    # Simulate AgentTrace(outputs=AgentReasons(...)) structure more closely if needed
    # For simplicity, keep using the dict structure, assuming the test extracts correctly
    return data


@pytest.mark.anyio
async def test_run_scorer_agent(scorer_agent: LLMScorer, judge_output_fixture: dict[str, Any]):
    """Tests running the LLMScorer agent by providing input derived from a sample Judge output.
    Verifies that the scorer produces output conforming to the QualScore model.
    """
    # 1. Prepare Input for Scorer
    # Extract the necessary parts from the judge_output fixture
    judge_outputs_data = judge_output_fixture.get("outputs", {})
    judge_input_data = judge_output_fixture.get("inputs", {})
    judge_records = judge_input_data.get("records", [])

    if not judge_records:
        pytest.fail("Judge output fixture is missing records.")

    # Assuming the last record contains the relevant ground truth
    # Note: This structure assumes ground_truth is directly in the record dict, not record.data
    # Adjust based on actual Record structure if needed.
    ground_truth = judge_records[-1].get("ground_truth")
    if not ground_truth:
        pytest.fail("Ground truth not found in the judge output fixture's records.")

    # Create the AgentInput for the scorer
    scorer_input_data = AgentInput(
        role="scorer",  # Set the role for context if needed
        # Provide the judge's output and ground truth under keys expected by the scorer's template/input mapping
        inputs={
            "judge_outputs": judge_outputs_data,  # Pass the judge's structured output
            "expected": ground_truth,  # Pass the ground truth
            # Add other inputs if the 'score' template requires them
        },
        # Pass the relevant record(s) containing the original content and ground truth
        records=judge_records[-1:],  # Pass only the last record (containing GT)
    )

    # Initialize the scorer agent (important if it has async init tasks)
    await scorer_agent.initialize()

    # 2. Execute Scorer Agent
    # Use the standard __call__ method
    result = await scorer_agent(message=scorer_input_data)

    # 3. Assertions
    assert isinstance(result, AgentTrace), "Scorer should return an AgentTrace object."
    assert not result.is_error, f"Scorer returned an error: {result.error}"
    assert result.outputs is not None, "Scorer output should not be None."

    # Check if the output conforms to the QualScore model
    assert isinstance(result.outputs, QualScore), f"Scorer output type is {type(result.outputs)}, expected QualScore."

    # Check the structure of QualScore
    assert hasattr(result.outputs, "assessments"), "QualScore output must have 'assessments' field."
    assert isinstance(result.outputs.assessments, list), "'assessments' field should be a list."

    # Check if assessments list is non-empty (assuming scorer should produce assessments)
    assert len(result.outputs.assessments) > 0, "'assessments' list should not be empty."

    # Check the structure of the first assessment (QualScoreCRA)
    first_assessment = result.outputs.assessments[0]
    assert isinstance(first_assessment, QualScoreCRA), "Items in 'assessments' should be QualScoreCRA objects."
    assert hasattr(first_assessment, "correct"), "QualScoreCRA must have 'correct' field."
    assert isinstance(first_assessment.correct, bool), "'correct' field must be boolean."
    assert hasattr(first_assessment, "feedback"), "QualScoreCRA must have 'feedback' field."
    assert isinstance(first_assessment.feedback, str), "'feedback' field must be a string."

    # Check the computed correctness score
    assert hasattr(result.outputs, "correctness"), "QualScore should have 'correctness' property."
    correctness = result.outputs.correctness
    assert isinstance(correctness, float) or correctness is None, "'correctness' should be float or None."
    if isinstance(correctness, float):
        assert 0.0 <= correctness <= 1.0, "'correctness' score must be between 0.0 and 1.0."

    logger.info(f"Scorer result (Correctness: {correctness}): {result.outputs}")
