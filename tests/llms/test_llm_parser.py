import pytest

from buttermilk._core.llms import AutoGenWrapper
from buttermilk.agents.judge import JudgeReasons

# Example parsing exercises for LLM parser.


examples_list_reasons = [
    (
        JudgeReasons,
        """{
            "conclusion": "The article does not violate the Trans Journalists Association's Stylebook and Coverage Guide. The reporting demonstrates strong adherence to the guidelines by respectfully covering Valentina Petrillo's historic Paralympic selection, using appropriate language and terminology, centering her voice through extensive direct quotes, and providing necessary policy context without resorting to sensationalism or problematic framing.",
            "reasons": "[\"The article consistently uses Valentina Petrillo's current name and she/her pronouns throughout, following the guideline to 'use someone's current name and pronouns when writing about that person's past, unless they request otherwise.'\", \"The article appropriately identifies Petrillo as transgender only where relevant to the story context - her historic status as the first openly transgender Paralympic athlete - rather than gratuitously calling attention to her gender identity, adhering to 'do not call unnecessary attention to a trans person's gender.'\", \"The article uses respectful, medically accurate language when describing her transition timeline, noting she 'began living as a woman in 2018 before commencing hormone therapy in 2019' without unnecessary medical details or salacious scrutiny.\", \"The reporting centers trans people's voices by extensively quoting Petrillo herself about her selection and experiences, following the guidance that 'trans people speak, not just be spoken about.'\", \"The article treats Petrillo's transgender identity as factual rather than using problematic phrasing like 'identifies as' - simply stating she is transgender, adhering to the '\"identifies as\" just means \"is\"' guidance.\", \"The article avoids all politicized or inaccurate phrases identified in the stylebook, including 'transgendered,' 'biological sex' (used pejoratively), 'transgenderism,' 'male-bodied,' 'female-bodied,' and other loaded terminology.\", \"The piece provides appropriate context about Paralympic policies regarding transgender athletes and includes relevant expert voices from Paralympic officials, explaining policy complexities without taking sides.\", \"When covering opposition to her participation, the article reports factually on the petition and legal challenges without amplifying harmful rhetoric or misgendering Petrillo, following guidance on 'when sources deadname or misgender a trans person.'\", \"The article focuses substantially on Petrillo's athletic achievements and Paralympic selection rather than centering the story primarily around her transgender status, treating her as an athlete first.\", \"The reporting avoids assumptions about competitive advantage and presents the complex regulatory landscape fairly, noting that different sporting bodies have different policies based on their interpretations of fairness and inclusion.\"]",
            "prediction": false,
            "uncertainty": "low",
        }""",
        True,
    )
]


# Mock wrapper to test parsing only.
@pytest.fixture
def llm_wrapper() -> AutoGenWrapper:
    """Mock AutoGenWrapper for testing parsing only."""

    class MockWrapper(AutoGenWrapper):
        pass

    return MockWrapper(client=None)


@pytest.mark.anyio
@pytest.mark.parametrize("output_obj, text, should_succeed", examples_list_reasons)
async def test_parse_list_reasons(llm_wrapper, output_obj, text, should_succeed):
    result = await llm_wrapper._parse_structured_output(text)
    assert should_succeed, f"Parsing was expected to fail but succeeded: {result}"
    assert isinstance(result, output_obj), f"Parsed result is not of type {output_obj}: {result}"
    assert result is not None, "Parsed result is None"
    assert hasattr(result, "reasons"), "Parsed result does not have 'reasons' attribute"
    assert isinstance(result.reasons, list), "'reasons' attribute is not a list"
    assert len(result.reasons) > 0, "'reasons' list is empty"
    for reason in result.reasons:
        assert isinstance(reason, str), f"Reason is not a string: {reason}"
        assert len(reason) > 0, "Reason is an empty string"
