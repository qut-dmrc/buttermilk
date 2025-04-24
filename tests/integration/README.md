# Buttermilk Integration Testing

This directory contains tools and utilities for testing individual components of the Buttermilk chatbot system in isolation, helping to detect issues before they appear in full batch runs.

## Issues Addressed

These tools help diagnose problems like:

1. **Pydantic Model Compatibility Issues**: `'BaseModel' object has no attribute '__private_attributes__'`
2. **API Rate Limiting**: Handling and recovering from rate limit errors (e.g., `Error code: 429`)
3. **Weave Scoring Errors**: `Error applying weave scorer to call CallRef`
4. **Malformed Responses**: Missing fields or unexpected response formats

## Integration Test Framework

The `test_agent_process.py` file contains a test framework for running isolated tests against individual agents using pytest. This helps validate that agents can process inputs correctly and handle common failure scenarios.

### Key Components

- `AgentIntegrationTest`: Base class for creating agent test harnesses
- Mock fixtures for LLM clients, weave, and other dependencies
- Test cases for specific agents and scenarios

## Command-Line Agent Tester

For quick diagnostics without writing a test case, use the `bm-test-agent` command-line tool:

```bash
# Test the scorer agent
uv run python -m buttermilk.runner.agent_tester buttermilk.agents.evaluators.scorer.LLMScorer \
  --role SCORER \
  --name "Test Scorer" \
  --params '{"model": "gemini2flash", "template": "score"}' \
  --input '{"prompt": "Evaluate this response", "expected": "The expected answer", "answers": [{"agent_id": "judge-abc", "agent_name": "Test Judge", "answer_id": "test123"}]}'

# Test with JSON records
bm-test-agent buttermilk.agents.evaluators.scorer.LLMScorer \
  --records '[{"content": "Test content", "data": {"ground_truth": "Expected answer"}}]'

# Test an agent directly without the adapter layer
bm-test-agent buttermilk.agents.flowcontrol.sequencer.Sequencer \
  --no-adapter \
  --input-type conductor_request \
  --input '{"participants": {"JUDGE": {"config": "xyz"}, "SCORER": {"config": "abc"}}, "task": "Assess this content"}'

# Enable verbose logging
bm-test-agent buttermilk.agents.llm.LLMAgent -v
```

## Best Practices

1. **Isolated Testing**: Test one agent at a time to pinpoint issues
2. **Mock Dependencies**: Use mock fixtures for external services 
3. **Test Edge Cases**: Include tests for rate limiting, malformed inputs, etc.
4. **Ensure Pydantic Compatibility**: Test with both Pydantic v1 and v2 models
5. **Verify Adapter Integration**: Test both with and without the AutogenAgentAdapter


### Pydantic Version Conflicts

The `test_pydantic_model_compatibility` test checks for compatibility with different Pydantic versions:

```python
# Run this test to check compatibility
pytest tests/integration/test_agent_process.py::TestDifferentiatorAgent::test_pydantic_model_compatibility -v
```

## Adding New Test Cases

To add new test cases:

1. Create a new test class in `test_agent_process.py`
2. Add fixture methods for any required test context
3. Implement test methods that exercise the agent's functionality
4. Use assertions to verify expected behavior

Example:

```python
@pytest.mark.anyio
class TestMyAgent:
    @pytest.fixture
    async def my_agent_test(self, mock_dependencies):
        # Setup code
        yield test_harness
        # Cleanup code
        
    async def test_my_agent_functionality(self, my_agent_test):
        # Test code
        assert result.is_expected_value
```
