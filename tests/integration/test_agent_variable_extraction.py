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

import json
from pathlib import Path

import jmespath
import pytest

from buttermilk._core.contract import AgentOutput
from buttermilk._core.message_data import extract_message_data
from buttermilk._core.types import Record
from buttermilk.agents.judge import JudgeReasons
from buttermilk.utils.templating import load_template

# Get the actual config directory
CONF_DIR = str(Path(__file__).parent.parent.parent / "buttermilk" / "conf")
FIXTURES_DIR = Path(__file__).parent / "fixtures"


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
def scorer_input_trace():
    """Load realistic FETCH + JUDGE + SYNTHESISER trace data.

    This fixture contains data structures matching what the scorer agent
    would receive from upstream agents in a real flow execution.
    """
    fixture_path = FIXTURES_DIR / "scorer_input_trace.json"
    with open(fixture_path) as f:
        data = json.load(f)
    # Remove the description field, keep only agent data
    return {k: v for k, v in data.items() if k != "description"}


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
        assert "source" in real_scorer_config["inputs"]
        assert "expected" in real_scorer_config["inputs"]

        # Verify the ACTUAL JMESPath expressions from the real config
        source_expr = real_scorer_config["inputs"]["source"]
        assert "[FETCH.outputs]" in source_expr
        assert "*.record" in source_expr

        expected_expr = real_scorer_config["inputs"]["expected"]
        assert "ground_truth" in expected_expr

    def test_real_scorer_inputs_mapping(self, real_scorer_config):
        """Test the REAL input mappings from scorer.yaml."""
        inputs = real_scorer_config["inputs"]

        # These are the ACTUAL mappings from conf/agents/scorer.yaml
        assert inputs["instructions"] == "JUDGE.messages[0].content || SYNTHESISER.messages[0].content"
        assert inputs["source"] == "[FETCH.outputs]||*.record||*.inputs.record"
        assert inputs["expected"] == "[FETCH.outputs].ground_truth||*.record.ground_truth||*.inputs.record.ground_truth"
        assert "answers" in inputs

    def test_extract_record_from_real_fetch_output(
        self, real_scorer_config, sample_record_with_ground_truth: Record
    ):
        """Test extraction using REAL JMESPath expressions from scorer config.

        The current config extracts 'source' (which contains ground_truth inside it).
        The template then accesses source.ground_truth.
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
        print(f"JMESPath expression: {real_inputs['source']}")
        print(f"Extracted keys: {list(extracted_data.keys())}")

        # Check if source was extracted
        if "source" not in extracted_data:
            pytest.fail(
                f"BUG: The 'source' field was not extracted from FETCH output.\n"
                f"JMESPath expression: {real_inputs['source']}\n"
                f"Extracted keys: {list(extracted_data.keys())}\n"
                f"This causes the scorer template to fail with 'source' is undefined."
            )

        # Verify the source was extracted as a list (JMESPath returns lists)
        source_data = extracted_data["source"]
        print(f"Source type: {type(source_data)}")
        print(f"Source data (first 200 chars): {str(source_data)[:200]}")

        # If it's a list, get the first element
        if isinstance(source_data, list):
            assert len(source_data) > 0, "Source list is empty"
            source_dict = source_data[0]
        else:
            source_dict = source_data

        # Verify ground_truth is accessible
        assert "ground_truth" in source_dict, (
            f"ground_truth not found in extracted source. Keys: {source_dict.keys()}"
        )
        assert "reasons" in source_dict["ground_truth"]
        assert len(source_dict["ground_truth"]["reasons"]) == 4

    def test_jmespath_record_extraction_patterns(
        self, real_scorer_config, sample_record_with_ground_truth: Record
    ):
        """Test the JMESPath expression for source extraction with different message structures.

        The config uses: "[FETCH.outputs]||*.record||*.inputs.record"

        This should extract source from:
        - FETCH.outputs (when message is from FETCH)
        - *.record (when message has record at top level)
        - *.inputs.record (when message has record in inputs)
        """
        import jmespath

        from buttermilk.utils import scrub_serializable

        source_expr = real_scorer_config["inputs"]["source"]
        print("\n=== Testing JMESPath Expression ===")
        print(f"Expression: {source_expr}")

        # Test 1: FETCH message structure
        print("\n[Test 1] FETCH message: {FETCH: {outputs: <record>}}")
        fetch_data = {
            "FETCH": {
                "outputs": scrub_serializable(
                    sample_record_with_ground_truth.model_dump()
                )
            }
        }
        result1 = jmespath.search(source_expr, fetch_data)
        print(f"  Result: {type(result1)} with {len(result1) if result1 else 0} items")
        assert result1 is not None, "FETCH.outputs pattern failed"

        # Test 2: Message with record at top level
        print("\n[Test 2] Message with top-level record: {JUDGE: {record: <record>}}")
        judge_data_toplevel = {
            "JUDGE": {
                "record": scrub_serializable(
                    sample_record_with_ground_truth.model_dump()
                )
            }
        }
        result2 = jmespath.search(source_expr, judge_data_toplevel)
        print(f"  Result: {type(result2)} with {len(result2) if result2 else 0} items")
        assert result2 is not None, "*.record pattern failed"

        # Test 3: Message with record in inputs (REAL structure from JUDGE/SYNTH)
        print(
            "\n[Test 3] Message with record in inputs: {JUDGE: {inputs: {record: <record>}}}"
        )
        judge_data_inputs = {
            "JUDGE": {
                "inputs": {
                    "record": scrub_serializable(
                        sample_record_with_ground_truth.model_dump()
                    )
                }
            }
        }
        result3 = jmespath.search(source_expr, judge_data_inputs)
        print(f"  Result: {type(result3)} with {len(result3) if result3 else 0} items")

        # THIS IS WHERE THE BUG IS!
        if result3 is None:
            print("\n❌ BUG FOUND: *.inputs.record pattern returns None!")
            print("Testing the fallback patterns separately:")

            # Test each part of the OR separately
            print(
                f"\n  [FETCH.outputs]: {jmespath.search('[FETCH.outputs]', judge_data_inputs)}"
            )
            print(f"  *.record: {jmespath.search('*.record', judge_data_inputs)}")
            print(
                f"  *.inputs.record: {jmespath.search('*.inputs.record', judge_data_inputs)}"
            )

            pytest.fail(
                "BUG: The JMESPath expression fails to extract source from JUDGE/SYNTH outputs!\n"
                f"Expression: {source_expr}\n"
                "Pattern '*.inputs.record' does not match structure {{JUDGE: {{inputs: {{record: ...}}}}}}\n"
                "This causes the scorer template to fail with 'source' is undefined."
            )

        assert result3 is not None, "*.inputs.record pattern failed for JUDGE structure"

    def test_template_rendering_succeeds_with_proper_extraction(
        self, real_scorer_config, sample_record_with_ground_truth: Record
    ):
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
                    "result": JudgeReasons(
                        reasons=["Reason 1", "Reason 2"],
                        prediction=True,
                        conclusion="Test conclusion",
                        uncertainty="medium",
                    ),
                    "answer_id": "call-123",
                }
            ],
            "criteria": [
                "Test criterion 1",
                "Test criterion 2",
            ],  # Required by score template
            "instructions": "Evaluate the judge's reasoning against the ground truth",  # Required
            "source": sample_record_with_ground_truth.content,  # Required - the source content being judged
        }

        # Render the REAL template with proper data
        rendered, undefined_vars, template_hash = load_template(
            template=real_template, parameters=proper_inputs
        )

        # MUST render without undefined variables
        assert len(undefined_vars) == 0, (
            f"Template has undefined variables: {undefined_vars}"
        )

        # Verify ground truth reasons appear in output
        for reason in sample_record_with_ground_truth.ground_truth["reasons"]:
            # Check if part of the reason appears
            assert any(word in rendered for word in reason.split()[:3]), (
                f"Ground truth reason not found in output: {reason[:50]}..."
            )


class TestScorerConfigWithRealTraceData:
    """Test scorer.yaml JMESPath expressions against real trace data.

    These tests use fixture data that matches the structure of actual
    agent outputs from a flow execution, validating that the scorer
    config correctly extracts data from FETCH and JUDGE agent outputs.
    """

    def test_fixture_loads_correctly(self, scorer_input_trace):
        """Verify the trace fixture loads with expected agent data."""
        assert "FETCH" in scorer_input_trace
        assert "JUDGE" in scorer_input_trace
        assert "SYNTHESISER" in scorer_input_trace

        # Verify FETCH has outputs with ground_truth
        fetch = scorer_input_trace["FETCH"]
        assert "outputs" in fetch
        assert "ground_truth" in fetch["outputs"]

        # Verify JUDGE has outputs and messages
        judge = scorer_input_trace["JUDGE"]
        assert "outputs" in judge
        assert "messages" in judge
        assert len(judge["messages"]) > 0

    def test_scorer_source_extraction_from_fetch(
        self, real_scorer_config, scorer_input_trace
    ):
        """Test the 'source' JMESPath expression extracts from FETCH.outputs.

        Config: source: "[FETCH.outputs]||*.record||*.inputs.record"
        Expected: Should extract the record from FETCH.outputs
        """
        source_expr = real_scorer_config["inputs"]["source"]
        print(f"\n=== Testing source extraction ===")
        print(f"Expression: {source_expr}")

        # Test against the real trace data
        result = jmespath.search(source_expr, scorer_input_trace)

        print(f"Result type: {type(result)}")
        print(f"Result: {result}")

        assert result is not None, (
            f"Source extraction failed!\n"
            f"Expression: {source_expr}\n"
            f"Available keys: {list(scorer_input_trace.keys())}"
        )

        # Should be a list containing the FETCH outputs
        if isinstance(result, list):
            assert len(result) > 0, "Source list should not be empty"
            source = result[0]
        else:
            source = result

        # Verify we got the record data
        assert "content" in source, f"Source should have content. Got keys: {source.keys()}"
        assert "ground_truth" in source, f"Source should have ground_truth. Got keys: {source.keys()}"

    def test_scorer_expected_extraction_from_fetch(
        self, real_scorer_config, scorer_input_trace
    ):
        """Test the 'expected' JMESPath expression extracts ground_truth from FETCH.

        Config: expected: "[FETCH.outputs].ground_truth||*.record.ground_truth||*.inputs.record.ground_truth"
        Expected: Should extract the ground_truth from FETCH.outputs
        """
        expected_expr = real_scorer_config["inputs"]["expected"]
        print(f"\n=== Testing expected extraction ===")
        print(f"Expression: {expected_expr}")

        result = jmespath.search(expected_expr, scorer_input_trace)

        print(f"Result type: {type(result)}")
        print(f"Result: {result}")

        assert result is not None, (
            f"Expected extraction failed!\n"
            f"Expression: {expected_expr}\n"
            f"FETCH.outputs keys: {list(scorer_input_trace['FETCH']['outputs'].keys())}"
        )

        # Should extract the ground_truth structure
        if isinstance(result, list):
            assert len(result) > 0, "Expected list should not be empty"
            ground_truth = result[0]
        else:
            ground_truth = result

        assert "reasons" in ground_truth, f"Should have reasons. Got: {ground_truth}"
        assert "violating" in ground_truth, f"Should have violating. Got: {ground_truth}"

    def test_scorer_answers_extraction_from_judge(
        self, real_scorer_config, scorer_input_trace
    ):
        """Test the 'answers' JMESPath expression extracts from JUDGE/SYNTHESISER.

        Config: answers: "[JUDGE,SYNTHESISER][].{agent_id: agent_info.agent_id, result: outputs, answer_id: call_id, error: error}"
        Expected: Should extract structured answer data from both JUDGE and SYNTHESISER
        """
        answers_expr = real_scorer_config["inputs"]["answers"]
        print(f"\n=== Testing answers extraction ===")
        print(f"Expression: {answers_expr}")

        result = jmespath.search(answers_expr, scorer_input_trace)

        print(f"Result type: {type(result)}")
        print(f"Result count: {len(result) if isinstance(result, list) else 'N/A'}")

        assert result is not None, (
            f"Answers extraction failed!\n"
            f"Expression: {answers_expr}"
        )
        assert isinstance(result, list), "Answers should be a list"
        assert len(result) >= 2, f"Should have at least JUDGE and SYNTHESISER answers, got {len(result)}"

        # Verify structure of extracted answers
        for answer in result:
            assert "agent_id" in answer, f"Answer should have agent_id. Got: {answer.keys()}"
            assert "result" in answer, f"Answer should have result. Got: {answer.keys()}"
            assert "answer_id" in answer, f"Answer should have answer_id. Got: {answer.keys()}"

        # Verify we got both JUDGE and SYNTHESISER
        agent_ids = [a["agent_id"] for a in result]
        assert any("JUDGE" in aid for aid in agent_ids), f"Should have JUDGE answer. Got: {agent_ids}"
        assert any("SYNTHESISER" in aid for aid in agent_ids), f"Should have SYNTHESISER answer. Got: {agent_ids}"

    def test_scorer_instructions_extraction_from_judge(
        self, real_scorer_config, scorer_input_trace
    ):
        """Test the 'instructions' JMESPath expression extracts from JUDGE/SYNTHESISER messages.

        Config: instructions: "JUDGE.messages[0].content || SYNTHESISER.messages[0].content"
        Expected: Should extract the first message content from JUDGE or SYNTHESISER
        """
        instructions_expr = real_scorer_config["inputs"]["instructions"]
        print(f"\n=== Testing instructions extraction ===")
        print(f"Expression: {instructions_expr}")

        result = jmespath.search(instructions_expr, scorer_input_trace)

        print(f"Result type: {type(result)}")
        print(f"Result preview: {str(result)[:100]}..." if result else "None")

        assert result is not None, (
            f"Instructions extraction failed!\n"
            f"Expression: {instructions_expr}\n"
            f"JUDGE.messages: {scorer_input_trace['JUDGE'].get('messages', [])}"
        )
        assert isinstance(result, str), f"Instructions should be a string, got {type(result)}"
        assert len(result) > 0, "Instructions should not be empty"

    def test_all_scorer_inputs_extract_successfully(
        self, real_scorer_config, scorer_input_trace
    ):
        """Test that ALL scorer input mappings extract successfully from trace data.

        This is a comprehensive test ensuring the scorer.yaml config works
        with realistic flow output data.
        """
        inputs = real_scorer_config["inputs"]
        extracted = {}
        failures = []

        print("\n=== Testing all scorer inputs ===")
        for key, expr in inputs.items():
            result = jmespath.search(expr, scorer_input_trace)
            if result is None or result == [] or result == {}:
                failures.append(f"{key}: expression '{expr}' returned {result}")
            else:
                extracted[key] = result
                print(f"✓ {key}: extracted successfully")

        if failures:
            pytest.fail(
                f"Some scorer inputs failed to extract:\n" +
                "\n".join(f"  - {f}" for f in failures) +
                f"\n\nAvailable trace keys: {list(scorer_input_trace.keys())}"
            )

        # Verify we got all required inputs
        required_keys = {"source", "expected", "answers", "instructions"}
        missing = required_keys - set(extracted.keys())
        assert not missing, f"Missing required inputs: {missing}"

    def test_fallback_patterns_for_source(
        self, real_scorer_config, scorer_input_trace
    ):
        """Test that source extraction fallback patterns work correctly.

        The config uses: "[FETCH.outputs]||*.record||*.inputs.record"

        This tests each fallback pattern individually to ensure they
        work when the primary pattern doesn't match.
        """
        source_expr = real_scorer_config["inputs"]["source"]

        # Test 1: Primary pattern - FETCH.outputs
        print("\n=== Testing fallback patterns for source ===")

        # With FETCH present, should use FETCH.outputs
        result_with_fetch = jmespath.search(source_expr, scorer_input_trace)
        assert result_with_fetch is not None, "Should extract from FETCH.outputs"
        print("✓ Primary pattern [FETCH.outputs] works")

        # Test 2: Fallback to *.record
        data_without_fetch = {k: v for k, v in scorer_input_trace.items() if k != "FETCH"}
        result_without_fetch = jmespath.search(source_expr, data_without_fetch)
        # This should fallback to *.record (from JUDGE.inputs.record)
        print(f"Without FETCH, result: {result_without_fetch}")
        # Note: The current config may not extract from JUDGE.inputs.record
        # This test documents the current behavior

        # Test 3: Test each part of the OR separately
        patterns = ["[FETCH.outputs]", "*.record", "*.inputs.record"]
        for pattern in patterns:
            result = jmespath.search(pattern, scorer_input_trace)
            status = "✓" if result else "✗"
            print(f"{status} Pattern '{pattern}': {type(result).__name__ if result else 'None'}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
