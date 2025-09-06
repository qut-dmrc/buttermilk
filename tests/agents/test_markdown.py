def test_judge_reasons_as_markdown():
    jr = JudgeReasons(
        conclusion="Non-violating",
        reasons=["Clear and factual", "No offensive language"],
        prediction=False,
        uncertainty="low",
        agent_name="JUDGE-gpt5",
        call_id="ABCD1234",
    )
    md = str(jr)
    assert "**JUDGE-gpt5 #ABCD1234**" in md
    assert "Non-violating" in md
    assert "- Clear and factual" in md
    assert "- No offensive language" in md
    assert "Prediction: No" in md
    assert "Uncertainty: Low" in md
