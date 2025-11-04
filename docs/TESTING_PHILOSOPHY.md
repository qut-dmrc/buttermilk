# Buttermilk Testing Philosophy

## Core Principle: Mock Only at the Boundary

We mock only at system boundaries—where our code interfaces with external systems. The "inside" of our application should be tested with real logic and plain assertions.

### System Boundaries (OK to Mock)

- **Network**: HTTP calls, API requests, websockets
- **Filesystem**: File I/O, directory operations
- **Time**: System clock, delays, timeouts
- **Randomness**: Random number generation
- **Environment**: Environment variables, system properties
- **Processes**: External process execution

### Our Code (NEVER Mock)

- Anything in `buttermilk.*`
- Our business logic
- Our data transformations
- Our agent implementations
- Our internal APIs

## Why This Philosophy?

### Problems with Over-Mocking

```python
# ❌ BAD: Testing the mock, not the code
@patch("buttermilk.agents.llm.LLMAgent._process")
def test_agent(mock_process):
    mock_process.return_value = "mocked result"
    agent = LLMAgent()
    result = agent.run()
    assert result == "mocked result"  # Proves nothing about LLMAgent!
```

### Benefits of Boundary-Only Mocking

```python
# ✅ GOOD: Testing real behavior, mocking only external calls
@respx.mock
def test_agent():
    # Mock only the external HTTP call
    respx.post("https://api.openai.com/v1/chat").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Paris"}}]})
    )

    # Test real agent logic
    agent = LLMAgent(model="gpt-4")
    result = agent.answer("What is the capital of France?")

    # Assert on actual behavior
    assert "Paris" in result
    assert agent.token_count > 0  # Real logic ran
```

## Practical Patterns

### 1. Network Boundaries

```python
# Use respx for httpx
import respx

@respx.mock
async def test_api_call():
    respx.get("https://api.example.com/data").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    result = await fetch_external_data()
    assert result["status"] == "ok"
```

### 2. Filesystem Boundaries

```python
# Use tmp_path fixture
def test_file_processing(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    result = process_file(test_file)
    assert result == "PROCESSED: test content"
```

### 3. Time Boundaries

```python
# Use freezegun
from freezegun import freeze_time

@freeze_time("2024-01-01")
def test_timestamp():
    record = create_record()
    assert record.timestamp == "2024-01-01T00:00:00"
```

### 4. Simple Test Doubles for Complex Systems

```python
# Instead of complex mocks, use simple test doubles
class FakeLLM:
    """Simple test double for LLM interactions."""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    async def generate(self, prompt):
        self.calls.append(prompt)
        for pattern, response in self.responses.items():
            if pattern in prompt:
                return response
        return "default response"

# Use in tests
async def test_agent_with_fake():
    fake_llm = FakeLLM(responses={
        "capital of France": "Paris",
        "population": "2.2 million"
    })

    agent = Agent(llm=fake_llm)
    result = await agent.research("Tell me about Paris")

    assert "Paris" in result
    assert len(fake_llm.calls) == 2  # Verify interactions
```

## Test Categories

### Unit Tests

- Test individual functions/methods
- No external dependencies
- Mock only boundaries if needed
- Fast execution (< 100ms)

### Integration Tests

- Test component interactions
- Mock external services at boundaries
- May use test databases/queues
- Moderate execution (< 5s)

### End-to-End Tests

- Test complete workflows
- Minimal mocking (only flaky external services)
- Use real services when possible
- Slower execution (OK to take 30s+)

## Red Flags in Tests

If you see these patterns, the test needs refactoring:

1. **Mocking our own code**

   ```python
   @patch("buttermilk._core.something")  # ❌ Our code!
   ```

1. **Complex mock setup**

   ```python
   mock = MagicMock()
   mock.method.return_value.attribute.side_effect = ...  # ❌ Too complex!
   ```

1. **Testing mock behavior**

   ```python
   mock.assert_called_with(...)  # ❌ Testing the mock, not the code!
   ```

1. **Mocking data transformations**

   ```python
   @patch("transform_data")
   def test(mock_transform):
       mock_transform.return_value = {"transformed": True}  # ❌ Not testing logic!
   ```

## Good Test Checklist

✅ Tests real code execution paths ✅ Mocks only external system boundaries ✅ Uses simple test doubles over complex mocks ✅ Assertions verify actual behavior ✅ Tests remain valid when implementation changes ✅ Tests are readable and maintainable

## Migration Strategy

When fixing existing tests:

1. **Identify boundary**: What external system is involved?
1. **Move mock to boundary**: Mock only the external call
1. **Test real logic**: Let the actual code run
1. **Verify behavior**: Assert on outcomes, not mock calls

Remember: **Every mock is a liability**. The fewer mocks, the more confidence in your tests.
