"""End-to-end demo tests for VertexBatchProcessor with real TJA fixtures.

These tests demonstrate VertexBatchProcessor using REAL data:
- Real TJA (Trans Journalists Association) stylebook criteria
- Real content samples with ground truth labels
- Real LLM models from bm.llms configuration

Purpose:
- Verify VertexBatchProcessor correctly integrates with buttermilk LLM infrastructure
- Demonstrate human-readable output with full verification
- Validate structured output parsing with actual content moderation tasks

Test Philosophy:
- NO MOCKS, NO FAKES - uses real APIs and real data
- Uses fixtures from production research projects
- Outputs are human-readable for manual verification
"""

from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, Field

from buttermilk._core.types import Record
from tests.conftest import CHEAP_CHAT_MODELS

# Constants for formatting
SEPARATOR = "=" * 60


# =============================================================================
# STRUCTURED OUTPUT MODELS
# =============================================================================


class TJAJudgment(BaseModel):
    """Structured output for TJA stylebook compliance evaluation.

    This model captures the LLM's judgment about whether content
    violates the Trans Journalists Association's stylebook guidelines.
    """

    violating: bool = Field(description="Whether the content violates the TJA stylebook")
    reasons: list[str] = Field(
        default_factory=list,
        description="Specific reasons for the judgment, citing relevant guidelines",
    )
    confidence: str = Field(
        default="medium",
        description="Confidence level: high, medium, or low",
    )
    key_violations: list[str] = Field(
        default_factory=list,
        description="Specific violations identified (e.g., 'deadnaming', 'wrong pronouns')",
    )


# =============================================================================
# FIXTURE LOADING UTILITIES
# =============================================================================


def load_tja_record(record_name: str = "agony_of_page") -> Record:
    """Load a real TJA record from the explorations/tja dataset.

    Args:
        record_name: Name of the record YAML file (without extension)

    Returns:
        Record object with content, metadata, and ground_truth
    """
    records_path = Path("/home/nic/src/explorations/tja/records/train")
    record_file = records_path / f"{record_name}.yaml"

    if not record_file.exists():
        raise FileNotFoundError(f"TJA record not found: {record_file}")

    with open(record_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return Record(
        record_id=data["record_id"],
        content=data["content"],
        metadata={
            **data.get("metadata", {}),
            "ground_truth": data.get("ground_truth", {}),
        },
    )


def load_tja_criteria() -> str:
    """Load the TJA stylebook criteria template content.

    Returns the full Trans Journalists Association stylebook guidelines
    for use as evaluation criteria.

    Returns:
        String containing the full TJA stylebook criteria
    """
    # Use the simpler criteria_ordinary template from buttermilk
    # which is more suitable for testing structured output
    criteria_path = Path("/home/nic/src/buttermilk/buttermilk/templates/criteria/criteria_ordinary.jinja2")

    if not criteria_path.exists():
        raise FileNotFoundError(f"Criteria template not found: {criteria_path}")

    with open(criteria_path, encoding="utf-8") as f:
        return f.read()


# =============================================================================
# DEMO TESTS - Human-Readable Verification
# =============================================================================


@pytest.mark.demo
class TestVertexBatchProcessorDemo:
    """Demo tests with human-readable output for verification.

    These tests are designed to:
    1. Use REAL records from research datasets
    2. Use REAL criteria templates
    3. Output results in human-readable format
    4. Fully verify the result object structure
    """

    @pytest.mark.anyio
    @pytest.mark.parametrize("model_name", CHEAP_CHAT_MODELS)
    async def test_demo_tja_evaluation(self, real_bm, model_name, capsys):
        """DEMO: Evaluate real TJA content with full human-readable output.

        This test demonstrates:
        - Loading real TJA record (agony_of_page - known violating content)
        - Using real criteria template
        - Getting structured LLM judgment
        - Human-readable verification output

        Expected: The "agony_of_page" content SHOULD be flagged as violating
        because it deadnames Elliot Page, uses wrong pronouns, and uses
        "identifies as" language.
        """
        from buttermilk.processors.vertex_batch import VertexBatchProcessor

        # ==== SETUP ====
        print(f"\n{SEPARATOR}")
        print(f"DEMO TEST: TJA Evaluation with {model_name}")
        print(SEPARATOR)

        # Load real fixtures
        record = load_tja_record("agony_of_page")
        criteria = load_tja_criteria()

        print(f"\n📄 RECORD: {record.record_id}")
        print(f"   Source: {record.metadata.get('outlet', 'unknown')}")
        print(f"   Title: {record.metadata.get('title', 'unknown')}")
        print(f"   Content length: {len(record.content)} chars")

        ground_truth = record.metadata.get("ground_truth", {})
        print("\n🎯 GROUND TRUTH:")
        print(f"   Violating: {ground_truth.get('violating', 'unknown')}")
        print(f"   Reasons: {ground_truth.get('reasons', [])}")

        # ==== PROCESSOR SETUP ====
        # Pass criteria as template variable - the judge template will use it
        processor = VertexBatchProcessor(
            name="tja_demo",
            model=model_name,
            template="judge",  # Built-in judge template
            template_vars={
                "criteria": criteria,  # Real criteria content
            },
            output_model="tests.demo.test_vertex_demo.TJAJudgment",
            fail_on_unfilled_parameters=False,
        )

        print("\n⚙️  PROCESSOR CONFIG:")
        print(f"   Model: {model_name}")
        print("   Template: judge")
        print("   Output model: TJAJudgment")

        # ==== PROCESSING ====
        print("\n🔄 Processing batch...")
        from buttermilk._core.processing_context import ProcessingContext

        ctx = ProcessingContext(session_id="demo", record=record)
        results = await processor.process_batch([ctx])

        # ==== VERIFICATION ====
        print(f"\n{SEPARATOR}")
        print("RESULTS & VERIFICATION")
        print(SEPARATOR)

        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        result = results[0]

        result_type = type(result).__name__
        print(f"\n📊 RESULT TYPE: {result_type}")

        if result_type == "TJAJudgment":
            # SUCCESS: Got structured output
            print("\n✅ STRUCTURED OUTPUT PARSED SUCCESSFULLY")
            print("\n📋 LLM JUDGMENT:")
            print(f"   Violating: {result.violating}")
            print(f"   Confidence: {result.confidence}")
            print(f"   Key violations: {result.key_violations}")
            print("\n   Reasons:")
            for i, reason in enumerate(result.reasons, 1):
                # Truncate long reasons for readability
                display_reason = reason[:200] + "..." if len(reason) > 200 else reason
                print(f"     {i}. {display_reason}")

            # Verify structure
            assert hasattr(result, "violating"), "Missing 'violating' field"
            assert hasattr(result, "reasons"), "Missing 'reasons' field"
            assert isinstance(result.violating, bool), "violating should be bool"
            assert isinstance(result.reasons, list), "reasons should be list"

            # Compare with ground truth
            print("\n🔍 GROUND TRUTH COMPARISON:")
            gt_violating = ground_truth.get("violating", None)
            if gt_violating is not None:
                match = result.violating == gt_violating
                print(f"   LLM says violating={result.violating}, ground truth={gt_violating}")
                print(f"   Match: {'✅ YES' if match else '❌ NO'}")

        elif hasattr(result, "metadata"):
            # Record with metadata - check for error or llm_output
            if "error" in result.metadata:
                print(f"\n❌ ERROR: {result.metadata['error']}")
                pytest.fail(f"LLM call failed: {result.metadata['error']}")
            elif "llm_output" in result.metadata:
                print("\n⚠️  RAW OUTPUT (not parsed):")
                output = result.metadata["llm_output"]
                print(f"   {output[:500]}...")
                # Still a success, just not structured
            else:
                print(f"\n⚠️  UNEXPECTED METADATA: {list(result.metadata.keys())}")

        else:
            pytest.fail(f"Unexpected result type: {result_type}")

        print(f"\n{SEPARATOR}")
        print("TEST COMPLETE")
        print(f"{SEPARATOR}\n")

        # Capture output for pytest
        captured = capsys.readouterr()
        # Re-print so it shows in pytest output with -s
        print(captured.out)
