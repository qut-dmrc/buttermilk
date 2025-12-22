"""Tests for scorer agent JMESPath extraction and template filling.

Verifies that:
1. JMESPath expressions in scorer.yaml correctly extract data from flow state
2. source is extracted as string (record's text representation)
3. expected (ground_truth) is extracted as dict
4. score.jinja2 template renders correctly with extracted values
"""

import jmespath
import pytest

from buttermilk.utils.templating import load_template


@pytest.fixture
def flow_state():
    """Mock flow state with FETCH and JUDGE outputs.

    This represents the data available in host._data after FETCH and JUDGE have run.
    """
    return {
        "FETCH": {
            "agent_id": "FETCH-abc123",
            "call_id": "fetch-call-001",
            "outputs": {
                "content": "Italian sprinter Valentina Petrillo is set to become the first openly transgender athlete to compete at the Paralympic Games. The visually impaired competitor, 50, was selected to represent Italy.",
                "ground_truth": {
                    "reasons": [
                        "Article uses correct pronouns throughout",
                        "Gender identity is relevant to the story",
                        "Article acknowledges opposing views",
                    ],
                    "label": True,
                },
                "record_id": "test_record_001",
                "metadata": {
                    "title": "Transgender Paralympian",
                    "outlet": "Sky News",
                },
            },
        },
        "JUDGE": {
            "agent_id": "JUDGE-def456",
            "agent_info": {"agent_id": "JUDGE-def456"},
            "call_id": "judge-call-001",
            "outputs": {
                "conclusion": "The article adheres to guidelines.",
                "reasons": ["Uses correct terminology", "Respects pronouns"],
                "prediction": False,
            },
            "messages": [
                {"role": "system", "content": "You are an expert analyst evaluating content against HRC guidelines."},
                {"role": "user", "content": "Analyze this article..."},
            ],
            "error": [],
        },
        "SYNTHESISER": {
            "agent_id": "SYNTH-ghi789",
            "agent_info": {"agent_id": "SYNTH-ghi789"},
            "call_id": "synth-call-001",
            "outputs": {
                "conclusion": "Synthesized analysis shows compliance.",
                "reasons": ["Combined analysis confirms adherence"],
            },
            "messages": [
                {"role": "system", "content": "You synthesize multiple analyst perspectives."},
            ],
            "error": [],
        },
    }


class TestScorerJMESPathExtraction:
    """Test JMESPath expressions from scorer.yaml."""

    def test_source_extraction_is_string(self, flow_state):
        """Verify source extracts as string from FETCH.outputs.content."""
        expr = "FETCH.outputs.content"
        result = jmespath.search(expr, flow_state)

        assert result is not None
        assert isinstance(result, str)
        assert "Valentina Petrillo" in result
        assert "transgender athlete" in result

    def test_expected_extraction_is_dict(self, flow_state):
        """Verify expected (ground_truth) extracts as dict from FETCH.outputs.ground_truth."""
        expr = "FETCH.outputs.ground_truth"
        result = jmespath.search(expr, flow_state)

        assert result is not None
        assert isinstance(result, dict)
        assert "reasons" in result
        assert isinstance(result["reasons"], list)
        assert len(result["reasons"]) == 3

    def test_instructions_extraction_with_fallback(self, flow_state):
        """Verify instructions extracts from JUDGE or SYNTHESISER messages."""
        expr = "JUDGE.messages[0].content || SYNTHESISER.messages[0].content"
        result = jmespath.search(expr, flow_state)

        assert result is not None
        assert isinstance(result, str)
        assert "expert analyst" in result or "synthesize" in result

    def test_instructions_fallback_when_judge_missing(self, flow_state):
        """Verify instructions falls back to SYNTHESISER when JUDGE messages empty."""
        # Remove JUDGE messages
        flow_state_no_judge = flow_state.copy()
        flow_state_no_judge["JUDGE"] = flow_state["JUDGE"].copy()
        flow_state_no_judge["JUDGE"]["messages"] = []

        expr = "JUDGE.messages[0].content || SYNTHESISER.messages[0].content"
        result = jmespath.search(expr, flow_state_no_judge)

        assert result is not None
        assert "synthesize" in result

    def test_answers_extraction_multi_agent(self, flow_state):
        """Verify answers extracts from both JUDGE and SYNTHESISER."""
        expr = "[JUDGE,SYNTHESISER][].{agent_id: agent_info.agent_id, result: outputs, answer_id: call_id, error: error}"
        result = jmespath.search(expr, flow_state)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 2

        # Check JUDGE entry
        judge_entry = next((r for r in result if "JUDGE" in r["agent_id"]), None)
        assert judge_entry is not None
        assert judge_entry["result"]["conclusion"] == "The article adheres to guidelines."

        # Check SYNTHESISER entry
        synth_entry = next((r for r in result if "SYNTH" in r["agent_id"]), None)
        assert synth_entry is not None
        assert synth_entry["result"]["conclusion"] == "Synthesized analysis shows compliance."


class TestScoreTemplateRendering:
    """Test score.jinja2 template renders correctly with extracted values."""

    def test_template_renders_with_extracted_values(self, flow_state):
        """Verify score template renders with source as string and expected as dict."""
        # Extract values using scorer.yaml JMESPath expressions
        source = jmespath.search("FETCH.outputs.content", flow_state)
        expected = jmespath.search("FETCH.outputs.ground_truth", flow_state)
        instructions = jmespath.search(
            "JUDGE.messages[0].content || SYNTHESISER.messages[0].content",
            flow_state
        )
        answers = jmespath.search(
            "[JUDGE,SYNTHESISER][].{agent_id: agent_info.agent_id, result: outputs, answer_id: call_id, error: error}",
            flow_state
        )

        # Render template
        rendered, undefined_vars, template_hash = load_template(
            template="score",
            parameters={
                "source": source,
                "expected": expected,
                "instructions": instructions,
                "answers": answers,
            },
        )

        # Verify rendering
        assert "Italian sprinter Valentina Petrillo" in rendered  # source content
        assert "Article uses correct pronouns" in rendered  # ground_truth reason
        assert "expert analyst" in rendered  # instructions
        assert len(template_hash) == 64  # SHA256 hex

    def test_template_handles_ground_truth_with_reasons_list(self, flow_state):
        """Verify template iterates over ground_truth.reasons when it's a dict with reasons."""
        expected = jmespath.search("FETCH.outputs.ground_truth", flow_state)

        # Verify expected has the structure the template expects
        assert isinstance(expected, dict)
        assert "reasons" in expected

        rendered, _, _ = load_template(
            template="score",
            parameters={
                "source": "Test source",
                "expected": expected,
                "instructions": "Test instructions",
                "answers": [{"agent_id": "test", "result": {}}],
            },
        )

        # Template should iterate over reasons (uses loop.index0)
        assert "0:" in rendered  # First reason with index
        assert "Article uses correct pronouns" in rendered

    def test_template_handles_ground_truth_as_string(self):
        """Verify template handles expected when it's a plain string."""
        rendered, _, _ = load_template(
            template="score",
            parameters={
                "source": "Test source",
                "expected": "This is a simple string ground truth",
                "instructions": "Test instructions",
                "answers": [{"agent_id": "test", "result": {}}],
            },
        )

        assert "simple string ground truth" in rendered

    def test_template_renders_multiple_answers(self, flow_state):
        """Verify template renders all answers from multiple agents."""
        answers = jmespath.search(
            "[JUDGE,SYNTHESISER][].{agent_id: agent_info.agent_id, result: outputs, answer_id: call_id, error: error}",
            flow_state
        )

        rendered, _, _ = load_template(
            template="score",
            parameters={
                "source": "Test source",
                "expected": {"reasons": ["Test reason"]},
                "instructions": "Test instructions",
                "answers": answers,
            },
        )

        # Both answers should be rendered
        assert "JUDGE" in rendered or "adheres to guidelines" in rendered
        assert "SYNTH" in rendered or "Synthesized analysis" in rendered


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
