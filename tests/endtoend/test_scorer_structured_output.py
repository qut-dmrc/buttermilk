"""End-to-end test for LLMScorer structured output with real input data.

This test verifies that lite models can produce valid QualScore structured output
when given real-world scorer input data from production BigQuery traces.
"""

import pytest

pytestmark = pytest.mark.slow
from buttermilk._core.messages import SystemMessage, UserMessage

from buttermilk.agents.evaluators.scorer import QualScore

# Real input data from BigQuery traces
REAL_TEMPLATE_VARS = {
    "instructions": "Your task is to review one or more draft answers...",
    "source": [
        {
            "content": """Parents Explain Why Trans Kids Should Be Banned From Prom

DAVENPORT, IA—In the wake of growing controversy surrounding transgender bathroom rights, parents across the nation are speaking out about why they believe transgender students should be banned from attending prom. The Onion asked parents to explain their reasoning, and this is what they said.

"My daughter worked really hard on her dress, and I don't want some boy in a dress stealing her spotlight."

"I just don't think it's appropriate for my son to slow dance with someone who was born a girl. What if he gets confused about his own identity?"

"Prom is supposed to be a magical night, and having transgender students there would just make everyone uncomfortable."

"I'm worried that if we let transgender kids go to prom, next thing you know they'll want to use the bathroom of their choice. We have to draw the line somewhere."

"My daughter shouldn't have to compete with someone who has an unfair biological advantage when it comes to finding a date."

"I think it's important to protect traditional values, and that means keeping prom the way it's always been—for boys and girls, not for people who can't decide which one they are."

"Honestly, I just think it would be too distracting. Prom should be about celebrating our kids, not making some kind of political statement."

"I don't want my son dancing with someone who might have a penis. That's just gross."

"These kids are confused enough as it is. The last thing they need is to see someone at prom who's dressed as the opposite gender. It'll just confuse them more."

"I mean, what's next? Are we going to let dogs go to prom? Where does it end?"

"I think we need to remember that God made us male and female for a reason. Letting transgender kids go to prom is just encouraging them to defy God's plan."

"My daughter has been looking forward to prom since she was in middle school. I don't want her night ruined because some boy decided he wants to wear a dress."

"Look, I'm not saying I have anything against transgender people. I just don't think prom is the right place for them. Maybe they could have their own separate prom or something."

"I'm concerned about the message it sends to other kids. If we normalize this kind of behavior, what's to stop more kids from deciding they're transgender?"

"Prom is a tradition, and traditions are important. If we start changing things now, pretty soon nothing will be sacred anymore."

"I just think it's unfair to the other kids. They shouldn't have to deal with this kind of thing at prom."

"My son's prom should be a night he remembers forever—for the right reasons. Having transgender students there would just make it weird."

"I don't want my daughter feeling like she has to compete with someone who has masculine features. It's not a fair playing field."

"At the end of the day, prom is about celebrating boys and girls becoming young men and women. Transgender students just don't fit into that narrative."

"I think we need to protect our children from this kind of ideology. Prom is supposed to be innocent and fun, not some political battleground."
""",
            "dataset_name": "tja",
            "ground_truth": {
                "reasons": ["The article does not violate the guidelines"],
                "violating": False,
            },
            "record_id": "onion_trans_prom",
        }
    ],
    "expected": [{"reasons": ["The article does not violate the guidelines"], "violating": False}],
    "answers": [
        {
            "agent_id": "SYNTHESISER-HBMW9V",
            "answer_id": "KWF6uTR2Cp7Xjk6hFdEQfp",
            "result": {
                "conclusion": "The content does not violate any of the provided criteria. The text is a piece of political satire from The Onion, intended to critique, mock, and condemn anti-transgender prejudice and the erosion of LGBTQ+ rights. While the article contains highly offensive and prejudicial statements, these are presented as fictional quotes from absurdly bigoted characters. The journalistic criteria (TJA, GLAAD, Trans_Simplified) generally permit the use of derogatory language or false narratives when contained within direct quotes that reveal the bias of the speaker. The media standards (APC) allow for the reporting of controversial expressions in the public interest for the purpose of commentary or critique. In this context, the satire serves to expose and ridicule the bigotry, thus adhering to the spirit and exceptions of the guidelines.",
                "prediction": False,
                "reasons": [
                    "The content is clearly identified as satire from 'The Onion,' which is crucial for interpreting the application of journalistic and ethical criteria.",
                    "**tja (Trans Journalists Association Stylebook):** This criterion is not violated.",
                    "**glaad (GLAAD Media Reference Guide):** This criterion is not violated.",
                    "**trans_simplified (Simplified Transgender Coverage Guidelines):** This criterion is not violated.",
                    "**apc (Australian Press Council Standards):** This criterion is not violated.",
                ],
                "uncertainty": "low",
            },
        }
    ],
}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "model_name",
    ["claude-haiku-4-5@20251001", "google/gemini-3.1-flash-lite", "gpt-5-nano"],
)
async def test_scorer_structured_output_with_real_data(
    real_bm,
    session_runner,
    model_name: str,
):
    """Test LLMScorer structured output with real input data from BigQuery.

    This test verifies that lite models can produce valid QualScore structured
    output when given real-world scorer template input. It directly invokes the
    LLM with the scorer template and validates the response parses correctly.

    Args:
        real_bm: Real BM instance from testing configuration
        session_runner: Session-scoped async fixture for single event loop
        model_name: Name of the model to test (parametrized across lite models)
    """
    # Get the LLM instance for this model
    llm = real_bm.llms[model_name]

    # Load and render the scorer template
    from buttermilk.utils.templating import load_template

    rendered_prompt, _, _, _ = load_template(
        template="score",
        template_vars=REAL_TEMPLATE_VARS,
    )

    # Split into system and user messages based on template structure
    # The template has "# System:" and "# User:" markers
    parts = rendered_prompt.split("# User:")
    system_content = parts[0].replace("# System:", "").strip()
    user_content = parts[1].strip() if len(parts) > 1 else ""

    # Create messages for the LLM
    messages = [
        SystemMessage(content=system_content),
        UserMessage(content=user_content, source="user"),
    ]

    # Call LLM with QualScore schema for structured output
    response = await llm.create(messages=messages, schema=QualScore)

    # Parse and validate the response
    parsed_response = QualScore.model_validate_json(response.content)

    # Assert response is valid QualScore with all required fields
    assert isinstance(parsed_response, QualScore)
    assert hasattr(parsed_response, "critical_errors")
    assert hasattr(parsed_response, "ground_truth_alignment")
    assert hasattr(parsed_response, "confidence")
    assert hasattr(parsed_response, "summary")

    # Verify critical_errors structure
    assert hasattr(parsed_response.critical_errors, "hallucinated_rule")
    assert hasattr(parsed_response.critical_errors, "hallucinated_fact")
    assert hasattr(parsed_response.critical_errors, "misinterpreted_fact")
    assert hasattr(parsed_response.critical_errors, "misapplied_rule")
    assert hasattr(parsed_response.critical_errors, "logical_error")
    assert hasattr(parsed_response.critical_errors, "abuse_of_discretion")
    assert hasattr(parsed_response.critical_errors, "reasons")
    assert isinstance(parsed_response.critical_errors.reasons, list)

    # Verify ground_truth_alignment is a list
    assert isinstance(parsed_response.ground_truth_alignment, list)

    # Verify confidence is a valid Confidence enum value
    assert parsed_response.confidence in ["high", "medium", "low"]

    # Verify summary is a non-empty string
    assert isinstance(parsed_response.summary, str)
    assert len(parsed_response.summary) > 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    "model_name",
    ["claude-haiku-4-5@20251001", "google/gemini-3.1-flash-lite", "gpt-5-nano"],
)
async def test_scorer_agent_with_real_data(
    real_bm,
    session_runner,
    model_name: str,
):
    """Test LLMScorer agent (full agent path, not just LLM) with real data.

    This tests the flow path: LLMScorer._process() → LLMCore → LLM.create()
    to identify where the parsing failure occurs when going through the agent
    rather than calling the LLM directly.

    Args:
        real_bm: Real BM instance from testing configuration
        session_runner: Session-scoped async fixture for single event loop
        model_name: Name of the model to test (parametrized across lite models)
    """
    from buttermilk._core.contract import AgentInput
    from buttermilk.agents.evaluators.scorer import LLMScorer, QualScore

    # Instantiate scorer agent with config matching flow config (scorer.yaml)
    # LLMAgent expects config fields as **kwargs, not an AgentConfig object
    scorer = LLMScorer(
        agent_id=f"SCORERS-TEST-{model_name}",
        role="SCORERS",
        parameters={
            "template": "score",
            "model": model_name,
        },
        inputs={
            "instructions": "instructions",
            "source": "source",
            "expected": "expected",
            "answers": "answers",
        },
        bm=real_bm,
    )

    # Create AgentInput with the real template_vars data
    agent_input = AgentInput(
        inputs=REAL_TEMPLATE_VARS,  # The same data as the direct test
    )

    # Call the agent's _process method
    result = await scorer._process(message=agent_input)

    # Verify result is valid AgentOutput containing QualScore
    assert result is not None
    assert hasattr(result, "outputs")
    assert isinstance(result.outputs, QualScore)

    # Verify QualScore structure (same checks as direct test)
    parsed_response = result.outputs

    # Assert response is valid QualScore with all required fields
    assert isinstance(parsed_response, QualScore)
    assert hasattr(parsed_response, "critical_errors")
    assert hasattr(parsed_response, "ground_truth_alignment")
    assert hasattr(parsed_response, "confidence")
    assert hasattr(parsed_response, "summary")

    # Verify critical_errors structure
    assert hasattr(parsed_response.critical_errors, "hallucinated_rule")
    assert hasattr(parsed_response.critical_errors, "hallucinated_fact")
    assert hasattr(parsed_response.critical_errors, "misinterpreted_fact")
    assert hasattr(parsed_response.critical_errors, "misapplied_rule")
    assert hasattr(parsed_response.critical_errors, "logical_error")
    assert hasattr(parsed_response.critical_errors, "abuse_of_discretion")
    assert hasattr(parsed_response.critical_errors, "reasons")
    assert isinstance(parsed_response.critical_errors.reasons, list)

    # Verify ground_truth_alignment is a list
    assert isinstance(parsed_response.ground_truth_alignment, list)

    # Verify confidence is a valid Confidence enum value
    assert parsed_response.confidence in ["high", "medium", "low"]

    # Verify summary is a non-empty string
    assert isinstance(parsed_response.summary, str)
    assert len(parsed_response.summary) > 0
