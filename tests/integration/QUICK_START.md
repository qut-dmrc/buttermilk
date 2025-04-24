# Buttermilk Agent Testing Quick Start

This guide provides quick solutions for the specific issues found in your logs.

## 1. For Pydantic Model Compatibility Issues (`BaseModel` has no attribute `__private_attributes__`)

```bash
# Test the differentiator agent directly to check for compatibility issues
bm-test-agent buttermilk.agents.differentiator.DifferentiatorAgent -v

# Or run the Pydantic compatibility test
pytest tests/integration/test_agent_process.py::TestDifferentiatorAgent::test_pydantic_model_compatibility -v
```

The error `'BaseModel' object has no attribute '__private_attributes__'` typically happens when:
- There's a mismatch between Pydantic v1 and v2 models
- Private attributes are being accessed directly instead of through the proper accessors
- The model structure doesn't match between agent implementations

## 2. For Weave Scorer Issues (`Error applying weave scorer to call CallRef`)

```bash
# Test the scorer agent specifically for weave integration issues
bm-test-agent buttermilk.agents.evaluators.scorer.LLMScorer \
  --role SCORER \
  --params '{"model": "mock-model", "template": "score"}' \
  --input '{"expected": "Expected answer", "answers": [{"agent_id": "judge-abc", "agent_name": "Test Judge", "answer_id": "test123"}]}' \
  --records '[{"content": "Test content", "data": {"ground_truth": "Expected answer"}}]' \
  -v

# Or run the specific test case
pytest tests/integration/test_agent_process.py::TestScorerAgent::test_scorer_listen_with_valid_message -v
```

These errors typically happen when:
- The weave trace ID is missing or malformed
- The format of the judge's output doesn't match what the scorer expects
- The ground truth data is missing from the records

## 3. For Rate Limiting Issues (`Error code: 429`)

```bash
# Test an agent with simulated rate limits
pytest tests/integration/test_agent_process.py::TestAPIRateLimit::test_llm_rate_limit_handling -v
```

This tests the retry mechanism and ensures your agents can recover from temporary API rate limits.

## 4. For Full Flow Testing

To test a complete, simplified version of your batch flow configuration, you can use:

```bash
# Test the autogen orchestrator with a simplified set of agents
pytest tests/integration/test_agent_process.py::TestAutogenGroupChat::test_orchestrator_run -v
```

This validates that the orchestrator can properly initialize, register agents, and execute steps in the flow.

## Adding Your Own Test Cases

You can easily extend the test framework for your specific needs:

1. Add a new test class in `test_agent_process.py`
2. Use the `AgentIntegrationTest` harness for agent testing
3. Mock any external dependencies (LLM clients, weave, etc.)
4. Run targeted tests for specific behaviors

See the full [README.md](./README.md) for more detailed information.
