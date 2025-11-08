import pytest

from buttermilk._core.llms import AutoGenWrapper
from buttermilk.agents.judge import JudgeReasons

# Example parsing exercises for LLM parser.


examples_list_reasons = [
    (
        JudgeReasons,
        """{
            "conclusion": "The article does not violate the Trans Journalists Association's Stylebook and Coverage Guide.",
            "reasons": [
                "The article consistently uses Valentina Petrillo's current name and she/her pronouns throughout",
                "The article appropriately identifies Petrillo as transgender only where relevant to the story context",
                "The article uses respectful, medically accurate language when describing her transition timeline"
            ],
            "prediction": false,
            "uncertainty": "low"
        }""",
        True,
    )
]


# Mock wrapper to test parsing only.
@pytest.fixture
def llm_wrapper() -> AutoGenWrapper:
    """Mock AutoGenWrapper for testing parsing only."""
    from unittest.mock import Mock

    class MockWrapper(AutoGenWrapper):
        pass

    # Provide required fields
    mock_factory = Mock()
    mock_info = {
        "model": "test",
        "vision": False,
        "function_calling": False,
        "json_output": True,
        "family": "test-family",
        "structured_output": True,
        "multiple_system_messages": False,
    }
    return MockWrapper(client=None, client_factory=mock_factory, model_info=mock_info)


@pytest.mark.anyio
@pytest.mark.parametrize("output_obj, text, should_succeed", examples_list_reasons)
async def test_parse_list_reasons(llm_wrapper, output_obj, text, should_succeed):
    result = await llm_wrapper._parse_structured_output(text, schema=output_obj)
    assert should_succeed, f"Parsing was expected to fail but succeeded: {result}"
    assert isinstance(result, output_obj), (
        f"Parsed result is not of type {output_obj}: {result}"
    )
    assert result is not None, "Parsed result is None"
    assert hasattr(result, "reasons"), "Parsed result does not have 'reasons' attribute"
    assert isinstance(result.reasons, list), "'reasons' attribute is not a list"
    assert len(result.reasons) > 0, "'reasons' list is empty"
    for reason in result.reasons:
        assert isinstance(reason, str), f"Reason is not a string: {reason}"
        assert len(reason) > 0, "Reason is an empty string"
