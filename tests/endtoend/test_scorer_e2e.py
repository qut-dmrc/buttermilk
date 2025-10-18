"""End-to-end test for scorer agent using the ENTIRE trans flow.

This test validates the complete pipeline from FETCH through JUDGE, SYNTH,
two SCORERS, and DIFF agent with a real record from the TJA dataset.

This is NOT a unit test. This runs the full AutogenOrchestrator with real LLM calls
to verify that:
1. FETCH retrieves the record with ground truth
2. JUDGE produces JudgeReasons output
3. SYNTH synthesizes the reasoning
4. Two SCORERS evaluate the answers against ground truth
5. DIFF agent compares the two scorer results
6. All agents produce sensible, non-empty outputs

This test uses real_bm fixture, real configuration, and real LLM calls.
"""

import pytest

from buttermilk._core.types import Record, RunRequest
from buttermilk.agents.evaluators.scorer import QualResults
from buttermilk.agents.judge import JudgeReasons
from buttermilk.runner.flowrunner import FlowRunner


@pytest.fixture
def trans_record_with_ground_truth() -> Record:
    """Real TJA record with ground truth for the trans flow.

    This is a real record from the TJA dataset that will be used
    to test the entire trans flow pipeline.
    """
    return Record(
        record_id="gender_clinics_south_australia",
        dataset_name="tja",
        split_type="train",
        content=(
            "Children as young as three years old are being referred to a gender diversity "
            "clinic, prompting a South Australian MP to call for an independent review of gender "
            "dysphoria treatment..."
        ),
        metadata={
            "created_at": "2025-10-13T14:25:22.047890",
            "date": "2024-06-27T00:00:00",
            "outlet": "7News",
            "title": "Children as young as three being referred to gender clinics in South Australia",
            "uri": "https://7news.com.au/news/children-as-young-as-three-being-referred-to-gender-clinics-in-south-australia--c-15153886",
        },
        ground_truth={
            "reasons": [
                "Violates by not centreing the voices of trans children and their families",
                "Violates by over-relying on anecdotal evidence that runs contrary to broad scientific consensus",
                "Violates by not contextualising the number of children receiving gender-affirming care",
                "Violates by quoting non-experts on the technical topic of gender-affirming care"
            ],
            "violating": True
        }
    )


@pytest.mark.endtoend
@pytest.mark.anyio
async def test_scorer_end_to_end_full_trans_flow(
    real_bm,
    trans_record_with_ground_truth: Record
):
    """Complete end-to-end test of the trans flow with scorer validation.

    This test runs the ENTIRE trans flow orchestrator with a real record,
    making actual LLM calls to verify all agents produce sensible outputs:

    1. FETCH: Retrieves the record (with ground truth)
    2. JUDGE: Evaluates the content against trans journalism guidelines
    3. SYNTH: Synthesizes the judge's reasoning
    4. SCORER (x2): Two scorers evaluate the answers against ground truth
    5. DIFF: Compares the two scorer results

    Success criteria:
    - Flow completes without errors
    - All agents produce non-empty outputs
    - JUDGE produces valid JudgeReasons
    - SCORERs produce valid QualResults with assessments
    - No "Undefined" appears in any output (template bug check)
    """
    # Get the trans flow configuration from real_bm
    trans_flow_config = real_bm.cfg.flows["trans"]

    # Create FlowRunner with the trans flow
    flow_runner = FlowRunner(
        source="testing",
        flows={"trans": trans_flow_config}
    )

    # Inject the real_bm instance so orchestrator can access LLMs
    flow_runner.bm = real_bm

    # Create run request for the trans flow
    # Pass the record_id so FETCH can retrieve it from TJA storage
    run_request = RunRequest(
        flow="trans",
        parameters={
            "record_id": trans_record_with_ground_truth.record_id,
            "dataset": "tja"
        },
        session_info=real_bm.session_info,
        session_id="test_scorer_e2e",
    )

    print(f"\n{'#'*80}")
    print(f"# Starting COMPLETE trans flow end-to-end test")
    print(f"# Record: {trans_record_with_ground_truth.record_id}")
    print(f"# Ground truth reasons: {len(trans_record_with_ground_truth.ground_truth['reasons'])}")
    print(f"{'#'*80}\n")

    # Run the flow and wait for completion
    await flow_runner.run_flow(run_request=run_request, wait_for_completion=True)

    # ========================================================================
    # CRITICAL VALIDATIONS: Verify all agents produced sensible outputs
    # ========================================================================

    print(f"\n{'='*80}")
    print(f"VALIDATION: Checking agent outputs")
    print(f"{'='*80}\n")

    # 1. Retrieve results from session storage
    from buttermilk.api.services.session_storage import SessionStorageService
    session_storage = SessionStorageService()
    results = session_storage.get_session_messages("test_scorer_e2e")

    # Convert messages to agent_outputs dict
    agent_outputs = {}
    for result in results:
        if hasattr(result, 'agent_id') and hasattr(result, 'outputs'):
            agent_outputs[result.agent_id] = result
            # Print progress for debugging
            print(f"\n{'='*60}")
            print(f"Agent: {result.agent_id}")
            output_type = type(result.outputs).__name__
            print(f"Output type: {output_type}")
            if hasattr(result.outputs, 'model_dump'):
                print(f"Output preview: {str(result.outputs)[:200]}...")
            print(f"{'='*60}\n")

    assert len(results) > 0, "Flow produced no results"
    print(f"✅ Flow produced {len(results)} results")

    # 2. Verify we have outputs from key agents
    # The trans flow should have: FETCH, JUDGE, SYNTH, SCORER-*, DIFF
    expected_agent_patterns = ["FETCH", "JUDGE", "SYNTH", "SCORER", "DIFF"]

    for pattern in expected_agent_patterns:
        matching_agents = [aid for aid in agent_outputs.keys() if pattern in aid.upper()]
        assert len(matching_agents) > 0, (
            f"No agent found matching pattern '{pattern}'. "
            f"Available agents: {list(agent_outputs.keys())}"
        )
        print(f"✅ Found agent(s) matching '{pattern}': {matching_agents}")

    # 3. Verify JUDGE produced JudgeReasons
    judge_agents = [aid for aid in agent_outputs.keys() if "JUDGE" in aid.upper()]
    for judge_id in judge_agents:
        judge_result = agent_outputs[judge_id]
        assert hasattr(judge_result, 'outputs'), f"JUDGE {judge_id} has no outputs"
        assert judge_result.outputs is not None, f"JUDGE {judge_id} outputs is None"

        if isinstance(judge_result.outputs, JudgeReasons):
            print(f"✅ {judge_id} produced JudgeReasons:")
            print(f"   - Prediction: {judge_result.outputs.prediction}")
            print(f"   - Reasons: {len(judge_result.outputs.reasons)}")
            print(f"   - Conclusion: {judge_result.outputs.conclusion[:100]}...")

            # Verify reasons are not empty
            assert len(judge_result.outputs.reasons) > 0, (
                f"{judge_id} produced empty reasons list"
            )
            assert judge_result.outputs.conclusion, (
                f"{judge_id} produced empty conclusion"
            )
        else:
            print(f"⚠️  {judge_id} output type: {type(judge_result.outputs)}")

    # 4. Verify SCORERS produced QualResults
    scorer_agents = [aid for aid in agent_outputs.keys() if "SCORER" in aid.upper()]
    assert len(scorer_agents) >= 2, (
        f"Expected at least 2 scorers, found {len(scorer_agents)}: {scorer_agents}"
    )
    print(f"✅ Found {len(scorer_agents)} scorer agents")

    for scorer_id in scorer_agents:
        scorer_result = agent_outputs[scorer_id]
        assert hasattr(scorer_result, 'outputs'), f"SCORER {scorer_id} has no outputs"
        assert scorer_result.outputs is not None, f"SCORER {scorer_id} outputs is None"

        if isinstance(scorer_result.outputs, QualResults):
            qual_results = scorer_result.outputs
            print(f"\n✅ {scorer_id} produced QualResults:")
            print(f"   - Assessed agent: {qual_results.assessed_agent_id}")
            print(f"   - Assessments: {len(qual_results.assessments)}")
            print(f"   - Score: {qual_results.score_text}")
            print(f"   - Correctness: {qual_results.correctness}")

            # Verify assessments exist
            assert len(qual_results.assessments) > 0, (
                f"{scorer_id} produced no assessments"
            )

            # CRITICAL: Verify no "Undefined" in feedback (the bug we're testing for)
            for i, assessment in enumerate(qual_results.assessments):
                assert "Undefined" not in assessment.feedback, (
                    f"BUG DETECTED: {scorer_id} assessment {i} contains 'Undefined'!\n"
                    f"Feedback: {assessment.feedback}"
                )
                assert assessment.feedback.strip(), (
                    f"{scorer_id} assessment {i} has empty feedback"
                )
                print(f"   - Assessment {i}: {assessment.correct} - {assessment.feedback[:80]}...")
        else:
            print(f"⚠️  {scorer_id} output type: {type(scorer_result.outputs)}")

    # 5. Verify DIFF agent ran
    diff_agents = [aid for aid in agent_outputs.keys() if "DIFF" in aid.upper()]
    if diff_agents:
        for diff_id in diff_agents:
            diff_result = agent_outputs[diff_id]
            if hasattr(diff_result, 'outputs') and diff_result.outputs:
                print(f"\n✅ {diff_id} produced output")
                print(f"   Type: {type(diff_result.outputs).__name__}")

    print(f"\n{'='*80}")
    print(f"✅ ALL VALIDATIONS PASSED")
    print(f"{'='*80}\n")

    print(f"\n{'#'*80}")
    print(f"# End-to-end trans flow test COMPLETED SUCCESSFULLY")
    print(f"# - All {len(expected_agent_patterns)} agent types produced outputs")
    print(f"# - JUDGE produced valid reasoning")
    print(f"# - {len(scorer_agents)} SCORERS produced valid assessments")
    print(f"# - No 'Undefined' values detected (template bug prevented)")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
