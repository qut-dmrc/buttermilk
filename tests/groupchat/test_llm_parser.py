import pytest

from buttermilk._core.llms import _parse_structured_output
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


@pytest.mark.anyio
@pytest.mark.parametrize("output_obj, text, should_succeed", examples_list_reasons)
async def test_parse_list_reasons(output_obj, text, should_succeed):
    """Test parsing structured output using the standalone parse function."""
    result = await _parse_structured_output(text, schema=output_obj)
    assert should_succeed, f"Parsing was expected to fail but succeeded: {result}"
    assert isinstance(result, output_obj), f"Parsed result is not of type {output_obj}: {result}"
    assert result is not None, "Parsed result is None"
    assert hasattr(result, "reasons"), "Parsed result does not have 'reasons' attribute"
    assert isinstance(result.reasons, list), "'reasons' attribute is not a list"
    assert len(result.reasons) > 0, "'reasons' list is empty"
    for reason in result.reasons:
        assert isinstance(reason, str), f"Reason is not a string: {reason}"
        assert len(reason) > 0, "Reason is an empty string"
