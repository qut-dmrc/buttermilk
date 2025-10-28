"""Integration tests for agent variable extraction using REAL configurations.

This module tests variable extraction across agents using their actual
configurations from conf/agents/, not fake test data.

Key focus areas:
1. Ground truth extraction using REAL JMESPath expressions from agent configs
2. Template rendering with REAL templates
3. Fail-fast behavior when variables are undefined
4. End-to-end validation from data extraction through template rendering

The critical bug being tested: When JMESPath extraction fails, variables
remain undefined causing template rendering to fail (as it should with
fail_on_unfilled_parameters=True).
"""

from pathlib import Path

import pytest

from buttermilk._core.contract import AgentOutput
from buttermilk._core.message_data import extract_message_data
from buttermilk._core.types import Record
from buttermilk.agents.judge import JudgeReasons
from buttermilk.utils.templating import load_template

# Get the actual config directory
CONF_DIR = str(Path(__file__).parent.parent.parent / "buttermilk" / "conf")


@pytest.fixture
def real_scorer_config(real_bm):
    """Get the REAL scorer agent configuration from conf/agents/scorer.yaml."""
    # Scorer is in observers, not agents
    # Return the DictConfig directly - don't convert to plain dict
    # The code expects OmegaConf objects, not plain dicts
    scorer_cfg = real_bm.cfg.run.flows["trans"]["observers"]["scorer"]
    return scorer_cfg


@pytest.fixture
def real_judge_config(real_bm):
    """Get the REAL judge agent configuration from conf/agents/judge.yaml."""
    # Return the DictConfig directly - don't convert to plain dict
    judge_cfg = real_bm.cfg.run.flows["trans"]["agents"]["judge"]
    return judge_cfg


@pytest.fixture
def real_fetch_config(real_bm):
    """Get the REAL fetch agent configuration from conf/agents/fetch.yaml."""
    # Return the DictConfig directly - don't convert to plain dict
    fetch_cfg = real_bm.cfg.run.flows["trans"]["agents"]["fetch"]
    return fetch_cfg


@pytest.fixture
def sample_record_with_ground_truth() -> Record:
    """Sample record matching the actual TJA dataset structure.

    This uses the REAL data structure from the actual trans flow,
    based on the record you provided.
    """
    return Record(
        record_id="gender_clinics_south_australia",
        dataset_name="tja",
        split_type="train",
        content=(
            "Children as young as three years old are being referred to a gender diversity\n"
            "clinic, prompting a South Australian MP to call for an independent review of gender\n"
            "dysphoria treatment..."
        ),
        metadata={
            "created_at": "2025-10-13T14:25:22.047890",
            "date": "2024-06-27T00:00:00",
            "outlet": "7News",
            "title": "Children as young as three being referred to gender clinics in South Australia",
            "uri": "https://7news.com.au/news/children-as-young-as-three-being-referred-to-gender-clinics-in-south-australia--c-15153886",
            "record_hash": "8104037784b10b1ac03bfc3ec74f1ff355f79ee1232912aaf45b59ee6fff327b",
            "ground_truth_hash": "6a346d16450441d303165b369fa41a65baf94afe675924f6c1e821312753131f",
        },
        ground_truth={
            "reasons": [
                "Violates by not centreing the voices of trans children and their families",
                "Violates by over-relying on anecdotal evidence that runs contrary to broad scientific consensus",
                "Violates by not contextualising the number of children receiving gender-affirming care",
                "Violates by quoting non-experts on the technical topic of gender-affirming care",
            ],
            "violating": True,
        },
    )


class TestAgentVariableExtraction:
    """Test variable extraction using REAL agent configurations."""

    def test_real_scorer_config_loads(self, real_scorer_config):
        """Verify we can load the real scorer configuration."""
        assert real_scorer_config is not None
        assert "inputs" in real_scorer_config
        assert "record" in real_scorer_config["inputs"]
        assert "expected" in real_scorer_config["inputs"]

        # Verify the ACTUAL JMESPath expressions from the real config
        record_expr = real_scorer_config["inputs"]["record"]
        assert "[FETCH.outputs]" in record_expr
        assert "*.record" in record_expr

        expected_expr = real_scorer_config["inputs"]["expected"]
        assert "ground_truth" in expected_expr

    def test_real_scorer_inputs_mapping(self, real_scorer_config):
        """Test the REAL input mappings from scorer.yaml."""
        inputs = real_scorer_config["inputs"]

        # These are the ACTUAL mappings from conf/agents/scorer.yaml
        assert inputs["model"] == "*.agent_info.parameters.model|[0]"
        assert inputs["criteria"] == "*.agent_info.parameters.criteria|[0]"
        assert inputs["template"] == "*.agent_info.parameters.template|[0]"
        assert inputs["record"] == "[FETCH.outputs]||*.record||*.inputs.record"
        # Note: The template now uses record.ground_truth, not a separate 'expected' field
        assert "answers" in inputs

    def test_extract_record_from_real_fetch_output(self, real_scorer_config, sample_record_with_ground_truth: Record):
        """Test extraction using REAL JMESPath expressions from scorer config.

        The current config extracts 'record' (which contains ground_truth inside it).
        The template then accesses record.ground_truth.
        """
        # Create a realistic FETCH agent output
        fetch_output = AgentOutput(
            call_id="fetch-call-123",
            agent_id="FETCH-ABC123",
            outputs=sample_record_with_ground_truth,
            messages=[],
            metadata=sample_record_with_ground_truth.metadata.copy(),
            error=[],
        )

        # Use the REAL input mappings from the actual config
        real_inputs = real_scorer_config["inputs"]

        # Extract data using the REAL JMESPath expressions
        extracted_data = extract_message_data(
            message=fetch_output,
            source="FETCH-ABC123",  # Source agent ID
            input_mappings=real_inputs,
        )

        print("\n=== FETCH Output Extraction ===")
        print(f"JMESPath expression: {real_inputs['record']}")
        print(f"Extracted keys: {list(extracted_data.keys())}")

        # Check if record was extracted
        if "record" not in extracted_data:
            pytest.fail(
                f"BUG: The 'record' field was not extracted from FETCH output.\n"
                f"JMESPath expression: {real_inputs['record']}\n"
                f"Extracted keys: {list(extracted_data.keys())}\n"
                f"This causes the scorer template to fail with 'record' is undefined."
            )

        # Verify the record was extracted as a list (JMESPath returns lists)
        record_data = extracted_data["record"]
        print(f"Record type: {type(record_data)}")
        print(f"Record data (first 200 chars): {str(record_data)[:200]}")

        # If it's a list, get the first element
        if isinstance(record_data, list):
            assert len(record_data) > 0, "Record list is empty"
            record_dict = record_data[0]
        else:
            record_dict = record_data

        # Verify ground_truth is accessible
        assert "ground_truth" in record_dict, f"ground_truth not found in extracted record. Keys: {record_dict.keys()}"
        assert "reasons" in record_dict["ground_truth"]
        assert len(record_dict["ground_truth"]["reasons"]) == 4

    def test_jmespath_record_extraction_patterns(self, real_scorer_config, sample_record_with_ground_truth: Record):
        """Test the JMESPath expression for record extraction with different message structures.

        The config uses: "[FETCH.outputs]||*.record||*.inputs.record"

        This should extract record from:
        - FETCH.outputs (when message is from FETCH)
        - *.record (when message has record at top level)
        - *.inputs.record (when message has record in inputs)
        """
        import jmespath

        from buttermilk.utils import scrub_serializable

        record_expr = real_scorer_config["inputs"]["record"]
        print("\n=== Testing JMESPath Expression ===")
        print(f"Expression: {record_expr}")

        # Test 1: FETCH message structure
        print("\n[Test 1] FETCH message: {FETCH: {outputs: <record>}}")
        fetch_data = {"FETCH": {"outputs": scrub_serializable(sample_record_with_ground_truth.model_dump())}}
        result1 = jmespath.search(record_expr, fetch_data)
        print(f"  Result: {type(result1)} with {len(result1) if result1 else 0} items")
        assert result1 is not None, "FETCH.outputs pattern failed"

        # Test 2: Message with record at top level
        print("\n[Test 2] Message with top-level record: {JUDGE: {record: <record>}}")
        judge_data_toplevel = {"JUDGE": {"record": scrub_serializable(sample_record_with_ground_truth.model_dump())}}
        result2 = jmespath.search(record_expr, judge_data_toplevel)
        print(f"  Result: {type(result2)} with {len(result2) if result2 else 0} items")
        assert result2 is not None, "*.record pattern failed"

        # Test 3: Message with record in inputs (REAL structure from JUDGE/SYNTH)
        print("\n[Test 3] Message with record in inputs: {JUDGE: {inputs: {record: <record>}}}")
        judge_data_inputs = {"JUDGE": {"inputs": {"record": scrub_serializable(sample_record_with_ground_truth.model_dump())}}}
        result3 = jmespath.search(record_expr, judge_data_inputs)
        print(f"  Result: {type(result3)} with {len(result3) if result3 else 0} items")

        # THIS IS WHERE THE BUG IS!
        if result3 is None:
            print("\n❌ BUG FOUND: *.inputs.record pattern returns None!")
            print("Testing the fallback patterns separately:")

            # Test each part of the OR separately
            print(f"\n  [FETCH.outputs]: {jmespath.search('[FETCH.outputs]', judge_data_inputs)}")
            print(f"  *.record: {jmespath.search('*.record', judge_data_inputs)}")
            print(f"  *.inputs.record: {jmespath.search('*.inputs.record', judge_data_inputs)}")

            pytest.fail(
                "BUG: The JMESPath expression fails to extract record from JUDGE/SYNTH outputs!\n"
                f"Expression: {record_expr}\n"
                "Pattern '*.inputs.record' does not match structure {{JUDGE: {{inputs: {{record: ...}}}}}}\n"
                "This causes the scorer template to fail with 'record' is undefined."
            )

        assert result3 is not None, "*.inputs.record pattern failed for JUDGE structure"

    def test_template_rendering_succeeds_with_proper_extraction(self, real_scorer_config, sample_record_with_ground_truth: Record):
        """Test that template renders successfully when extraction works.

        Uses the REAL scorer template with properly extracted data.
        This shows what SHOULD happen when the extraction bug is fixed.
        """
        # Use the REAL template from config
        real_template = real_scorer_config["parameters"]["template"]

        # Manually construct the inputs that SHOULD come from extraction
        # NOTE: Template expects 'expected' which contains the ground_truth data
        # Score template requires: expected, answers, criteria, instructions, source
        proper_inputs = {
            "expected": sample_record_with_ground_truth.ground_truth,
            "answers": [
                {
                    "agent_id": "JUDGE-TEST",
                    "result": JudgeReasons(reasons=["Reason 1", "Reason 2"], prediction=True, conclusion="Test conclusion", uncertainty="medium"),
                    "answer_id": "call-123",
                }
            ],
            "criteria": ["Test criterion 1", "Test criterion 2"],  # Required by score template
            "instructions": "Evaluate the judge's reasoning against the ground truth",  # Required
            "source": sample_record_with_ground_truth.content,  # Required - the source content being judged
        }

        # Render the REAL template with proper data
        rendered, undefined_vars, template_hash = load_template(template=real_template, parameters=proper_inputs)

        # MUST render without undefined variables
        assert len(undefined_vars) == 0, f"Template has undefined variables: {undefined_vars}"

        # Verify ground truth reasons appear in output
        for reason in sample_record_with_ground_truth.ground_truth["reasons"]:
            # Check if part of the reason appears
            assert any(word in rendered for word in reason.split()[:3]), f"Ground truth reason not found in output: {reason[:50]}..."


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
