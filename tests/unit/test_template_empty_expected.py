"""Test that LLMCore template filling fails when expected variables render empty.

This reproduces a production bug where the score.jinja2 template's
'expected' variable was passed in inputs but rendered as empty in the
final message when called through LLMCore.

Bug details:
- Trace call_id: 27e06641-dd18-4f7b-bbab-9b4ab3dd29aa
- expected was present in BaseRecord from JMESPath transform
- Template rendered <BEGIN EXPECTED ANSWER KEY POINTS><END EXPECTED ANSWER KEY POINTS>
  with no content between the markers
- This should have FAILED but didn't
"""

import json
from pathlib import Path

import pytest

from buttermilk._core.llm_core import LLMCore
from buttermilk._core.types import BaseRecord
from buttermilk.utils.templating import load_template


@pytest.fixture
def judge_trace():
    """Load real judge trace from production bug."""
    fixture_path = (
        Path(__file__).parent.parent
        / "integration/fixtures/tja_judge_trace_minneapolis_shooting.json"
    )
    with open(fixture_path) as f:
        traces = json.load(f)
    return traces[0]  # BigQuery returns array with single result


@pytest.mark.anyio
async def test_llmcore_fills_template_with_expected_from_record(judge_trace):
    """Test that LLMCore._fill_template() renders 'expected' from BaseRecord.

    This tests one layer higher than load_template() - it tests LLMCore's
    _fill_template() method which is what the actual rescore pipeline uses.

    The bug: When JMESPath transform adds 'expected' to the BaseRecord,
    LLMCore should pass it to the template. But something in the pipeline
    strips it out before rendering.
    """
    # Extract real data from trace
    trace_inputs = json.loads(judge_trace["inputs"])
    record_id = trace_inputs["inputs"]["record_id"]
    assert record_id == "minneapolis_shooting_sky_news"

    # Create BaseRecord simulating what JMESPath transform would produce
    # This is what the rescore pipeline creates after SQL query + JMESPath
    record = BaseRecord(
        record_id="test_scorer_trace",
        dataset_name="traces",
        # These fields come from JMESPath transform in rescore.yaml
        expected='{"reasons": ["- Violates by misgendering and using incorrect pronouns", "- Violates by publishing deadname"], "violating": true}',
        instructions=json.loads(judge_trace["messages"][0])["content"],
        source="Test source content",
        answers=[{"result": json.loads(judge_trace["outputs"])}],
    )

    # Create LLMCore configured for score template
    llm_core = LLMCore(
        model="gemini25pro", template="score", fail_on_unfilled_parameters=True
    )

    # Call _fill_template() - this is what process() calls internally
    # Pass record as inputs parameter since that's how processor mode works
    messages = await llm_core._fill_template(
        inputs=record,
        record=None,  # In processor mode, data comes from inputs
        context=[],
    )

    # Get the rendered message content
    rendered = messages[0].content

    # Verify 'expected' appears in rendered output
    assert "<BEGIN EXPECTED ANSWER KEY POINTS>" in rendered
    assert "<END EXPECTED ANSWER KEY POINTS>" in rendered

    # Extract the content between markers
    start = rendered.find("<BEGIN EXPECTED ANSWER KEY POINTS>") + len(
        "<BEGIN EXPECTED ANSWER KEY POINTS>"
    )
    end = rendered.find("<END EXPECTED ANSWER KEY POINTS>")
    key_points_content = rendered[start:end].strip()

    # THE BUG: key_points_content is empty when it should have content
    # The 'expected' field from BaseRecord should be passed to the template
    assert key_points_content, (
        f"KEY POINTS section should have content but got empty. Full message:\n{rendered}"
    )
    assert "Violates by misgendering" in rendered, (
        "Expected content should be in rendered output"
    )


@pytest.mark.anyio
async def test_jmespath_transform_preserves_expected_field():
    """Test that JMESPathTransform preserves 'expected' field from SQL output.

    This tests the next layer up - the JMESPath transform that maps SQL columns
    to BaseRecord fields. The bug might be here if 'expected' is being filtered out.
    """
    from buttermilk._core.types import BaseRecord
    from buttermilk.processors.jmespath_transform import JMESPathTransform

    # Load SQL output from fixture
    sql_output_path = (
        Path(__file__).parent.parent / "integration/fixtures/rescore_sql_output.json"
    )
    with open(sql_output_path) as f:
        sql_results = json.load(f)

    sql_row = sql_results[0]

    # Create BaseRecord from SQL output (simulating what BigQuery source does)
    # Extract record_id separately to avoid duplicate keyword argument
    record_id = sql_row.pop("record_id")
    input_record = BaseRecord(
        record_id=record_id,
        dataset_name="traces",
        **sql_row,  # All other SQL columns become extra fields
    )

    # Verify input has expected
    assert hasattr(input_record, "expected"), "Input record should have expected field"
    assert input_record.expected is not None, "expected should not be None"

    # Create JMESPathTransform with mappings from rescore.yaml
    transform = JMESPathTransform(
        mappings={
            "answers": "answers",
            "criteria": "criteria",
            "instructions": "instructions",
            "source": "source",
            "parent_call_id": "parent_call_id",
            "expected": "expected",  # This mapping should preserve expected
        }
    )

    # Run the transform
    output_records = []
    async for record in transform.process(input_record, processor_stage="test"):
        output_records.append(record)

    # Verify output
    assert len(output_records) == 1, "Should produce one output record"
    output_record = output_records[0]

    # THE BUG: Check if 'expected' survived the transform
    assert hasattr(output_record, "expected"), (
        "Output record should have expected field"
    )
    assert output_record.expected is not None, (
        f"expected should not be None, got: {output_record.expected}"
    )
    assert "Violates by misgendering" in output_record.expected, (
        "expected should have content"
    )


@pytest.mark.anyio
async def test_full_rescore_pipeline_preserves_expected():
    """Test complete rescore pipeline: SQL → JMESPath → LLMCore → rendered template.

    This is the full end-to-end test that reproduces the production bug.
    It runs the complete processor chain from rescore.yaml.
    """
    from buttermilk._core.llm_core import LLMCore
    from buttermilk._core.types import BaseRecord
    from buttermilk.processors.jmespath_transform import JMESPathTransform

    # Load SQL output from fixture
    sql_output_path = (
        Path(__file__).parent.parent / "integration/fixtures/rescore_sql_output.json"
    )
    with open(sql_output_path) as f:
        sql_results = json.load(f)

    sql_row = sql_results[0].copy()

    # STEP 1: Create initial BaseRecord from SQL (what BigQuery source does)
    record_id = sql_row.pop("record_id")
    sql_record = BaseRecord(record_id=record_id, dataset_name="traces", **sql_row)

    # Verify SQL data has expected
    assert hasattr(sql_record, "expected"), "SQL record should have expected"
    assert sql_record.expected is not None, "SQL expected should not be None"

    # STEP 2: Run through JMESPathTransform (first processor in rescore.yaml)
    jmespath_transform = JMESPathTransform(
        mappings={
            "answers": "answers",
            "criteria": "criteria",
            "instructions": "instructions",
            "source": "source",
            "parent_call_id": "parent_call_id",
            "expected": "expected",
        }
    )

    transformed_records = []
    async for record in jmespath_transform.process(
        sql_record, processor_stage="jmespath"
    ):
        transformed_records.append(record)

    assert len(transformed_records) == 1
    transformed_record = transformed_records[0]

    # Verify JMESPath output has expected
    assert hasattr(transformed_record, "expected"), (
        "Transformed record should have expected"
    )
    assert transformed_record.expected is not None, (
        "Transformed expected should not be None"
    )

    # STEP 3: Test what LLMCore would pass to the template
    # We can't actually call LLM without BM singleton, so just test _fill_template directly
    llm_core = LLMCore(
        model="gemini25pro", template="score", fail_on_unfilled_parameters=True
    )

    # Call _fill_template which extracts fields from record and renders template
    # This is what process_with_llm calls internally (line 355 in llm_core.py)
    messages = await llm_core._fill_template(
        inputs=transformed_record,  # In processor mode, record is passed as inputs
        record=None,
        context=[],
    )

    # Extract the result from _fill_template
    result = type("obj", (object,), {"messages": messages})()

    # Check the rendered messages
    assert len(result.messages) > 0, "Should have rendered messages"
    rendered = result.messages[0].content

    # THE BUG: Verify 'expected' appears in rendered output
    assert "<BEGIN EXPECTED ANSWER KEY POINTS>" in rendered
    assert "<END EXPECTED ANSWER KEY POINTS>" in rendered

    start = rendered.find("<BEGIN EXPECTED ANSWER KEY POINTS>") + len(
        "<BEGIN EXPECTED ANSWER KEY POINTS>"
    )
    end = rendered.find("<END EXPECTED ANSWER KEY POINTS>")
    key_points_content = rendered[start:end].strip()

    # This is the production bug - expected renders empty despite being in the pipeline
    assert key_points_content, (
        f"KEY POINTS section should have content but got empty. Full message:\n{rendered}"
    )
    assert "Violates by misgendering" in rendered, (
        "expected content should be in rendered output"
    )


def test_score_template_with_empty_expected_should_fail():
    """Test that score template FAILS when 'expected' is empty string.

    Fail-fast principle: Empty expected should cause template rendering to fail
    when fail_on_unfilled_parameters=True.
    """
    untrusted_inputs = {
        "expected": "",  # Empty string
        "instructions": "Test instructions",
        "source": "Test source",
        "answers": [{"result": {"prediction": True, "reasons": ["test"]}}],
    }

    # This SHOULD raise an error because expected is empty
    # and we have fail_on_unfilled_parameters=True
    with pytest.raises(Exception, match="unfilled|expected"):
        rendered, unfilled_vars, _ = load_template(
            template="score",
            parameters={"fail_on_unfilled_parameters": True},
            untrusted_inputs=untrusted_inputs,
        )


def test_score_template_without_expected_should_fail():
    """Test that score template FAILS when 'expected' is missing entirely.

    Fail-fast principle: Missing expected should cause template rendering to fail
    when fail_on_unfilled_parameters=True.
    """
    untrusted_inputs = {
        # expected is missing entirely
        "instructions": "Test instructions",
        "source": "Test source",
        "answers": [{"result": {"prediction": True, "reasons": ["test"]}}],
    }

    # This SHOULD raise an error because expected is missing
    # and we have fail_on_unfilled_parameters=True
    with pytest.raises(Exception, match="unfilled|expected"):
        rendered, unfilled_vars, _ = load_template(
            template="score",
            parameters={"fail_on_unfilled_parameters": True},
            untrusted_inputs=untrusted_inputs,
        )
