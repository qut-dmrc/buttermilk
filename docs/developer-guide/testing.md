# Testing Guide

This guide covers testing strategies, best practices, and common patterns for Buttermilk development.

## Table of Contents
- [Overview](#overview)
- [Testing Philosophy](#testing-philosophy)
- [Test Categories](#test-categories)
- [Setting Up Tests](#setting-up-tests)
- [Writing Tests](#writing-tests)
- [Running Tests](#running-tests)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)

## Overview

Buttermilk uses a test-driven development approach with pytest as the primary testing framework. Our testing strategy emphasizes:

- **Test-first development**: Write failing tests before implementing features
- **Comprehensive coverage**: Unit, integration, and end-to-end tests
- **Real-world scenarios**: Tests that reflect actual usage patterns
- **Fast feedback**: Quick test execution for development workflow

## Testing Philosophy

### Test-Driven Development (TDD)

**The TDD Cycle:**
1. **Red**: Write a failing test that demonstrates the problem
2. **Green**: Write minimal code to make the test pass
3. **Refactor**: Improve the code while keeping tests passing

**Example:**
```python
def test_agent_processes_content():
    # Red: This test will fail initially
    agent = ContentAnalyzer(config)
    result = agent.process("Test content")
    assert result.sentiment == "neutral"

# Now implement ContentAnalyzer.process() to make this pass
```

### Test Categories

**Unit Tests (Fast, Isolated):**
- Test individual functions/methods
- Mock external dependencies
- Execute quickly (< 1 second each)

**Integration Tests (Real Dependencies):**
- Test component interactions
- Use real services where possible
- Test end-to-end workflows

**System Tests (Full Application):**
- Test complete user scenarios
- Use real configuration and data
- Validate entire pipelines

## Setting Up Tests

### Prerequisites

**Authentication:**
```bash
# Required for LLM access in tests
gcloud auth login
gcloud auth application-default login
```

**Environment:**
```bash
# Install test dependencies
uv install

# Set up test environment
export GOOGLE_CLOUD_PROJECT=your-test-project
export BUTTERMILK_ENV=test
```

### Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_agents.py      # Agent-specific tests
│   ├── test_config.py      # Configuration tests
│   └── test_utils.py       # Utility function tests
├── integration/            # Integration tests
│   ├── test_flows.py       # Flow execution tests
│   ├── test_llm_integration.py  # LLM service tests
│   └── test_storage.py     # Storage backend tests
├── system/                 # System tests
│   ├── test_api.py         # API endpoint tests
│   └── test_cli.py         # CLI command tests
├── fixtures/               # Test data and fixtures
│   ├── config/            # Test configurations
│   ├── data/              # Sample data files
│   └── __init__.py        # Fixture definitions
└── conftest.py            # pytest configuration
```

## Writing Tests

### Unit Tests

**Agent Testing:**
```python
import pytest
from unittest.mock import Mock, patch
from buttermilk.agents.llm import LLMAgent
from buttermilk._core.types import Record

class TestLLMAgent:
    @pytest.fixture
    def agent_config(self):
        return {
            "agent_id": "test_agent",
            "role": "ANALYZER",
            "description": "Test agent",
            "parameters": {"model": "gpt-4"},
            "tools": []
        }
    
    @pytest.fixture
    def agent(self, agent_config):
        return LLMAgent(**agent_config)
    
    def test_agent_initialization(self, agent):
        assert agent.agent_id == "test_agent"
        assert agent.role == "ANALYZER"
    
    @patch('buttermilk.agents.llm.LLMClient')
    def test_process_record(self, mock_llm_client, agent):
        # Arrange
        mock_llm_client.return_value.generate.return_value = "Test response"
        record = Record(id="test_1", content="Test content")
        
        # Act
        result = agent.process(record)
        
        # Assert
        assert result.content == "Test response"
        mock_llm_client.return_value.generate.assert_called_once()
```

**Configuration Testing:**
```python
import pytest
from omegaconf import DictConfig
from buttermilk._core.config import FlowConfig

class TestFlowConfig:
    def test_valid_config_creation(self):
        config_dict = {
            "name": "test_flow",
            "description": "Test flow",
            "agents": [{"name": "agent1", "type": "LLMAgent"}]
        }
        config = FlowConfig(**config_dict)
        assert config.name == "test_flow"
        assert len(config.agents) == 1
    
    def test_invalid_config_raises_error(self):
        with pytest.raises(ValueError):
            FlowConfig(name="", description="Invalid flow")
```

### Integration Tests

**Flow Execution:**
```python
import pytest
from buttermilk.runner.flowrunner import FlowRunner
from buttermilk._core.types import Record

class TestFlowExecution:
    @pytest.fixture
    def flow_runner(self):
        return FlowRunner.from_config("test_flow")
    
    @pytest.mark.asyncio
    async def test_complete_flow_execution(self, flow_runner):
        # Arrange
        record = Record(id="test_1", content="Test content for analysis")
        
        # Act
        result = await flow_runner.run_record(record)
        
        # Assert
        assert result.success
        assert result.outputs
        assert len(result.outputs) > 0
    
    @pytest.mark.asyncio
    async def test_flow_with_multiple_agents(self, flow_runner):
        # Test that agents are executed in correct order
        record = Record(id="test_1", content="Multi-agent test")
        result = await flow_runner.run_record(record)
        
        # Verify all agents produced output
        agent_names = [output.agent_name for output in result.outputs]
        assert "preprocessor" in agent_names
        assert "analyzer" in agent_names
        assert "validator" in agent_names
```

**LLM Integration:**
```python
import pytest
from buttermilk.llm.client import LLMClient

class TestLLMIntegration:
    @pytest.fixture
    def llm_client(self):
        return LLMClient(model="gemini-pro")
    
    @pytest.mark.asyncio
    async def test_llm_generation(self, llm_client):
        # This test requires real LLM access
        response = await llm_client.generate("What is 2+2?")
        assert response
        assert "4" in response
    
    @pytest.mark.asyncio
    async def test_llm_with_system_prompt(self, llm_client):
        system_prompt = "You are a helpful math tutor."
        response = await llm_client.generate(
            "What is 2+2?", 
            system_prompt=system_prompt
        )
        assert response
        assert len(response) > 0
```

### System Tests

**API Testing:**
```python
import pytest
from fastapi.testclient import TestClient
from buttermilk.api.flow import create_app

class TestAPI:
    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_flow_execution_endpoint(self, client):
        payload = {
            "text": "Test content",
            "flow": "test_flow"
        }
        response = client.post("/flow/test_flow", json=payload)
        assert response.status_code == 200
        assert "results" in response.json()
```

### Test Data and Fixtures

**Using Fixtures:**
```python
import pytest
from pathlib import Path

@pytest.fixture
def sample_data():
    return {
        "records": [
            {"id": "1", "content": "Sample content 1"},
            {"id": "2", "content": "Sample content 2"}
        ]
    }

@pytest.fixture
def test_config_path():
    return Path(__file__).parent / "fixtures" / "config" / "test_config.yaml"

@pytest.fixture
def mock_llm_response():
    return {
        "content": "Mock LLM response",
        "metadata": {"tokens": 100, "model": "mock-model"}
    }
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/unit/test_agents.py

# Run specific test function
uv run pytest tests/unit/test_agents.py::test_agent_initialization
```

### Test Categories

```bash
# Run only unit tests
uv run pytest tests/unit/

# Run only integration tests
uv run pytest tests/integration/

# Run only system tests
uv run pytest tests/system/
```

### Test Markers

```bash
# Run tests marked as slow
uv run pytest -m slow

# Skip slow tests
uv run pytest -m "not slow"

# Run scheduled tests
uv run pytest -m scheduled

# Run tests that require authentication
uv run pytest -m auth_required
```

### Coverage Reports

```bash
# Run tests with coverage
uv run pytest --cov=buttermilk

# Generate HTML coverage report
uv run pytest --cov=buttermilk --cov-report=html

# Set coverage threshold
uv run pytest --cov=buttermilk --cov-fail-under=80
```

### Parallel Test Execution

```bash
# Run tests in parallel
uv run pytest -n auto

# Run with specific number of workers
uv run pytest -n 4
```

## Common Patterns

### Mocking External Services

**LLM Service Mocking:**
```python
from unittest.mock import AsyncMock, patch

@patch('buttermilk.llm.client.LLMClient')
async def test_agent_with_mocked_llm(mock_llm_class):
    # Setup mock
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "Mocked response"
    mock_llm_class.return_value = mock_llm
    
    # Test
    agent = LLMAgent(config)
    result = await agent.process(record)
    
    # Verify
    assert result.content == "Mocked response"
    mock_llm.generate.assert_called_once()
```

**Storage Service Mocking:**
```python
@patch('buttermilk.storage.client.StorageClient')
def test_storage_operations(mock_storage_class):
    mock_storage = Mock()
    mock_storage.save.return_value = True
    mock_storage_class.return_value = mock_storage
    
    # Test storage operations
    storage = StorageService(config)
    result = storage.save_results(data)
    
    assert result is True
    mock_storage.save.assert_called_once_with(data)
```

### Configuration Testing

**Hydra Configuration Testing:**
```python
import pytest
from hydra import compose, initialize_config_store
from omegaconf import DictConfig

def test_flow_configuration():
    with initialize_config_store(config_path="../conf"):
        cfg = compose(config_name="test_flow")
        assert isinstance(cfg, DictConfig)
        assert cfg.flow.name == "test_flow"
        assert len(cfg.agents) > 0
```

### Async Testing

**Async Test Patterns:**
```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_agent_processing():
    agent = AsyncAgent(config)
    result = await agent.process(record)
    assert result.success

@pytest.mark.asyncio
async def test_concurrent_processing():
    agent = AsyncAgent(config)
    records = [Record(id=f"test_{i}") for i in range(5)]
    
    # Process concurrently
    tasks = [agent.process(record) for record in records]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 5
    assert all(result.success for result in results)
```

### Error Testing

**Exception Testing:**
```python
def test_agent_handles_invalid_input():
    agent = LLMAgent(config)
    
    with pytest.raises(ValueError, match="Invalid input"):
        agent.process(None)

def test_agent_handles_llm_errors():
    agent = LLMAgent(config)
    
    with patch.object(agent, 'llm_client') as mock_llm:
        mock_llm.generate.side_effect = Exception("LLM error")
        
        with pytest.raises(Exception, match="LLM error"):
            agent.process(record)
```

## Test Organization

### Test Naming

```python
# Good test names
def test_agent_processes_valid_content():
    pass

def test_agent_raises_error_on_invalid_input():
    pass

def test_flow_executes_agents_in_correct_order():
    pass

# Bad test names
def test_agent():
    pass

def test_1():
    pass

def test_stuff():
    pass
```

### Test Structure (AAA Pattern)

```python
def test_something():
    # Arrange - Set up test data and conditions
    agent = LLMAgent(config)
    record = Record(id="test", content="test content")
    
    # Act - Execute the code being tested
    result = agent.process(record)
    
    # Assert - Verify the results
    assert result.success
    assert result.content == "expected output"
```

### Test Parametrization

```python
@pytest.mark.parametrize("input_text,expected_sentiment", [
    ("I love this!", "positive"),
    ("This is terrible", "negative"),
    ("It's okay", "neutral"),
])
def test_sentiment_analysis(input_text, expected_sentiment):
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze(input_text)
    assert result.sentiment == expected_sentiment
```

## Performance Testing

### Benchmarking

```python
import pytest
import time

def test_agent_performance():
    agent = LLMAgent(config)
    record = Record(id="test", content="test content")
    
    start_time = time.time()
    result = agent.process(record)
    end_time = time.time()
    
    processing_time = end_time - start_time
    assert processing_time < 5.0  # Should complete within 5 seconds
    assert result.success
```

### Load Testing

```python
@pytest.mark.slow
def test_agent_handles_concurrent_requests():
    agent = LLMAgent(config)
    records = [Record(id=f"test_{i}") for i in range(100)]
    
    # Process all records
    results = []
    for record in records:
        result = agent.process(record)
        results.append(result)
    
    # Verify all succeeded
    assert len(results) == 100
    assert all(result.success for result in results)
```

## Troubleshooting

### Common Test Issues

**Authentication Errors:**
```bash
# Re-authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login

# Verify authentication
gcloud auth list
```

**Import Errors:**
```bash
# Install in development mode
uv install -e .

# Check Python path
uv run python -c "import sys; print(sys.path)"
```

**Configuration Errors:**
```bash
# Check test configuration
uv run pytest --collect-only

# Debug configuration loading
uv run python -c "
from hydra import compose, initialize_config_store
with initialize_config_store(config_path='../conf'):
    cfg = compose(config_name='test_flow')
    print(cfg)
"
```

### Debug Test Failures

**Verbose Output:**
```bash
# Run with maximum verbosity
uv run pytest -vvv

# Show local variables in tracebacks
uv run pytest --tb=long

# Drop into debugger on failure
uv run pytest --pdb
```

**Logging in Tests:**
```python
import logging

def test_with_logging():
    # Enable logging for debugging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    logger.debug("Starting test")
    # ... test code ...
    logger.debug("Test completed")
```

### Test Environment Issues

**Environment Variables:**
```bash
# Set test environment
export BUTTERMILK_ENV=test
export GOOGLE_CLOUD_PROJECT=test-project

# Check environment
uv run python -c "import os; print(os.environ.get('BUTTERMILK_ENV'))"
```

**Resource Cleanup:**
```python
@pytest.fixture
def temp_directory():
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)
```

## Best Practices

### Test Quality

1. **Test Behavior, Not Implementation**
   - Focus on what the code does, not how it does it
   - Test public interfaces, not internal methods

2. **Use Descriptive Test Names**
   - Test names should explain what is being tested
   - Include expected behavior and conditions

3. **Keep Tests Independent**
   - Each test should be able to run independently
   - Don't rely on test execution order

4. **Use Fixtures for Common Setup**
   - Avoid duplicating setup code
   - Use pytest fixtures for reusable test data

5. **Test Edge Cases**
   - Test with empty inputs
   - Test with malformed data
   - Test error conditions

### Test Organization

1. **Group Related Tests**
   - Use classes to group related test methods
   - Organize tests by functionality

2. **Use Meaningful Assertions**
   - Assert specific values, not just existence
   - Include helpful error messages

3. **Mock External Dependencies**
   - Don't let tests depend on external services
   - Use mocks to isolate units under test

4. **Test at the Right Level**
   - Unit tests for individual components
   - Integration tests for component interactions
   - System tests for end-to-end workflows

## Continuous Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.12
    
    - name: Install dependencies
      run: |
        pip install uv
        uv install
    
    - name: Run tests
      run: |
        uv run pytest --cov=buttermilk --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: uv run pytest
        language: system
        pass_filenames: false
        always_run: true
```

## Conclusion

Testing is a critical part of Buttermilk development. By following these guidelines and patterns, you'll create robust, maintainable tests that help ensure the reliability and quality of the codebase.

Remember:
- Write tests first (TDD)
- Test behavior, not implementation
- Keep tests fast and independent
- Use appropriate test categories
- Mock external dependencies
- Write clear, descriptive test names

Good testing practices lead to better code design and increased confidence in changes.