# MCP Endpoint Testing Strategy

## Overview

This document outlines the testing strategy for Buttermilk's MCP (Model Context Protocol) endpoints. The testing approach includes unit tests, integration tests, and manual testing tools.

## Testing Levels

### 1. Unit Tests (`tests/api/test_mcp_endpoints.py`)

**Purpose**: Test MCP endpoint logic in isolation with mocked dependencies.

**Coverage**:
- Request/response models (MCPToolResponse, request classes)
- Endpoint routing and parameter validation
- Error handling for invalid inputs
- Helper function logic (`run_single_agent`)

**Benefits**:
- Fast execution (no external dependencies)
- Reliable (no flaky external calls)
- Easy to debug
- Good for TDD development

**Limitations**:
- Doesn't test real agent execution
- May miss integration issues

### 2. Integration Tests (`tests/api/test_mcp_integration.py`)

**Purpose**: Test MCP endpoints with real Buttermilk components but mocked LLM calls.

**Coverage**:
- Full request flow through FastAPI → FlowRunner → Agent
- Real agent instantiation and configuration
- Actual orchestrator setup (with mocked LLM responses)
- Performance characteristics

**Benefits**:
- Tests real component integration
- Catches configuration issues
- Tests actual agent lifecycle
- No LLM costs or delays

**Limitations**:
- Slower than unit tests
- More complex setup
- May still miss LLM-specific issues

### 3. Manual Testing (`scripts/test_mcp.py`)

**Purpose**: Interactive testing and debugging tools.

**Features**:
- Test server with minimal setup
- Manual endpoint testing
- Development debugging tools

## Test Configuration Strategy

### Problem: Buttermilk Configuration Complexity

Buttermilk uses Hydra for configuration management, which creates challenges for testing:

1. **Interpolation Dependencies**: Configs reference other configs (e.g., `${flows}`)
2. **Storage Conflicts**: Multiple storage configurations cause conflicts
3. **LLM Dependencies**: Real agents require LLM API configurations

### Solution: Layered Testing Approach

#### 1. Unit Tests with Dependency Injection
```python
# Override dependencies for isolated testing
app.dependency_overrides[get_flows] = lambda: mock_flow_runner
```

#### 2. Integration Tests with Minimal Config
```python
# Minimal configuration that avoids Hydra complexity
test_config = {
    "flows": {"test_flow": minimal_flow_config},
    "agents": {"judge": minimal_agent_config}
}
```

#### 3. Mock Real Components Where Needed
```python
# Mock LLM calls but use real agent structure
with patch('buttermilk.agents.judge.Judge._process') as mock_process:
    mock_process.return_value = mock_response
```

## Running Tests

### Quick Test Commands

```bash
# Run unit tests only (fast)
uv run python scripts/test_mcp.py unit

# Run integration tests (medium speed)  
uv run python scripts/test_mcp.py integration

# Run all tests
uv run python scripts/test_mcp.py all

# Start test server for manual testing
uv run python scripts/test_mcp.py server

# Test endpoints against running server
uv run python scripts/test_mcp.py manual
```

### Detailed Pytest Commands

```bash
# Unit tests with verbose output
uv run python -m pytest tests/api/test_mcp_endpoints.py -v

# Integration tests excluding slow tests
uv run python -m pytest tests/api/test_mcp_integration.py -v -m "not slow"

# All MCP tests
uv run python -m pytest tests/api/ -k "mcp"

# Run with coverage
uv run python -m pytest tests/api/ --cov=buttermilk.api.mcp
```

## Test Data Strategy

### Mock Responses
Tests use realistic mock responses that match expected agent outputs:

```python
mock_judge_response = {
    "toxicity_score": 0.2,
    "reasoning": "Content analysis indicates low toxicity due to...",
    "criteria_met": ["criteria_ordinary"],
    "timestamp": "2024-01-01T12:00:00Z"
}
```

### Test Inputs
Standardized test inputs for consistent testing:

```python
standard_test_inputs = {
    "judge": {
        "text": "This is a test message about climate policy.",
        "criteria": "toxicity",
        "model": "gpt4o",
        "flow": "tox"
    },
    "synthesize": {
        "text": "Original policy text...",
        "criteria": "clarity",
        "model": "gpt4o", 
        "flow": "tox"
    }
}
```

## Test Environment Setup

### Dependencies Required
- `pytest` - Test framework
- `pytest-asyncio` - Async test support  
- `fastapi[test]` - FastAPI test client
- `requests` - For manual endpoint testing

### Configuration Files
- `tests/api/test_mcp_endpoints.py` - Unit tests
- `tests/api/test_mcp_integration.py` - Integration tests
- `scripts/test_mcp.py` - Test runner and utilities

## Debugging Failed Tests

### Common Issues and Solutions

#### 1. Configuration Errors
```
Error: InterpolationKeyError: Interpolation key 'flows' not found
```
**Solution**: Use mocked dependencies in unit tests, minimal config in integration tests.

#### 2. Agent Instantiation Failures
```
Error: Agent class not found or misconfigured
```
**Solution**: Check mock agent configurations match real agent structure.

#### 3. Async Test Issues
```
Error: RuntimeError: This event loop is already running
```
**Solution**: Ensure tests are marked with `@pytest.mark.anyio` and use proper async fixtures.

### Debug Tools

#### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Test Individual Components
```python
# Test just the helper function
result = await run_single_agent("judge", "tox", test_params, mock_runner)
```

#### Use Test Server for Interactive Debugging
```bash
# Start test server
python scripts/test_mcp.py server

# Test endpoints with curl
curl -X POST http://localhost:8001/mcp/tools/judge \
  -H "Content-Type: application/json" \
  -d '{"text": "test", "criteria": "toxicity", "model": "gpt4o", "flow": "tox"}'
```

## Future Testing Enhancements

### 1. End-to-End Tests
- Test complete workflows from HTTP request to agent response
- Include real LLM calls in CI/CD pipeline (with cost limits)

### 2. Performance Tests
- Load testing with multiple concurrent requests
- Memory usage profiling
- Response time benchmarking

### 3. Contract Tests
- Validate MCP protocol compliance
- Test with actual MCP clients
- Schema validation tests

### 4. Error Scenario Tests
- Network failures
- Timeout handling
- Invalid agent configurations

## CI/CD Integration

### Test Categories for CI
1. **Fast Tests** (always run): Unit tests
2. **Medium Tests** (PR builds): Integration tests  
3. **Slow Tests** (nightly): Real LLM tests, performance tests

### Test Markers
```python
@pytest.mark.unit      # Fast unit tests
@pytest.mark.integration  # Integration tests
@pytest.mark.slow      # Slow/expensive tests
@pytest.mark.manual    # Manual testing only
```

This testing strategy ensures MCP endpoints are reliable, maintainable, and properly integrated with the Buttermilk system while avoiding the complexity of full system configuration during development.