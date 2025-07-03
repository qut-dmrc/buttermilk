# Buttermilk Runtime Debugging Infrastructure

A comprehensive, flow-agnostic debugging and testing framework for the Buttermilk project. Designed to be modular, extensible, and HASS-researcher friendly.

## Overview

This debugging infrastructure provides systematic runtime debugging capabilities for Buttermilk flows, agents, and configurations. It addresses common issues like:

- Enhanced RAG agent initialization errors
- Configuration validation problems  
- Import dependency issues
- Flow execution failures
- Type checking errors

## Key Components

### 1. MCP Client Testing (`mcp_client.py`)
Flow-agnostic testing client for MCP endpoints:
- Health checks and API availability testing
- Flow execution testing with timeout handling
- Agent-specific testing capabilities
- Comprehensive flow validation reports

```python
from buttermilk.debug.mcp_client import MCPFlowTester

client = MCPFlowTester()
health = client.health_check()
result = client.test_osb_vector_query("test query")
```

### 2. GCP Log Analysis (`gcp_logs.py`) 
Real-time log analysis using gcloud CLI:
- Daemon startup issue detection
- Agent error analysis
- Flow performance monitoring
- Real-time log streaming

```python
from buttermilk.debug.gcp_logs import GCPLogAnalyzer

analyzer = GCPLogAnalyzer()
startup_analysis = analyzer.analyze_daemon_startup()
```

### 3. Error Capture System (`error_capture.py`)
Enhanced error capture for runtime debugging:
- Type checking diagnostics
- Safe isinstance handling for subscripted generics
- Context-aware error reporting
- Stack trace analysis

```python
from buttermilk.debug.error_capture import safe_isinstance_check, capture_enhanced_rag_errors

# Safe type checking
is_list = safe_isinstance_check(obj, list)  # Instead of isinstance(obj, List[str])

# Enhanced error capture
@capture_enhanced_rag_errors()
def my_function():
    pass
```

### 4. Configuration Validator (`config_validator.py`)
Comprehensive configuration validation:
- YAML syntax validation
- Type-specific storage config validation
- Agent configuration checking
- Dependency availability verification

```python
from buttermilk.debug.config_validator import validate_configuration

report = validate_configuration()
if report.is_valid:
    print("Configuration is valid!")
```

### 5. Debug CLI (`cli.py`)
Unified command-line interface for all debugging tools:

```bash
# Test daemon startup
python -m buttermilk.debug.cli test-startup --flow osb

# Test MCP queries
python -m buttermilk.debug.cli test-mcp-query --flow osb --query "test"

# Analyze GCP logs
python -m buttermilk.debug.cli analyze-logs --minutes-back 30

# Validate configuration
python -m buttermilk.debug.cli validate-config --verbose

# Complete diagnostic
python -m buttermilk.debug.cli diagnose-issue --comprehensive
```

## Quick Start

### 1. Basic Health Check
```bash
python -m buttermilk.debug.cli diagnose-issue --flow osb
```

### 2. Configuration Validation
```bash
python -m buttermilk.debug.cli validate-config --verbose
```

### 3. Real-time Log Monitoring
```bash
python -m buttermilk.debug.cli stream-logs
```

### 4. Comprehensive Testing
```bash
python -m buttermilk.debug.cli test-flow-comprehensive --flow osb
```

## Architecture Principles

### Flow-Agnostic Design
All debugging tools are designed to work with any Buttermilk flow:
- OSB (Online Safety Benchmark)
- Trans (Translation flows)
- Tox (Toxicity detection)
- Custom flows

### Modular Components
Each debugging component can be used independently:
- Import only what you need
- No tight coupling between components
- Easy to extend and modify

### HASS-Researcher Friendly
Designed for accessibility and clarity:
- Clear error messages with suggestions
- Structured output formats (JSON, readable text)
- Comprehensive documentation
- Progressive disclosure of complexity

### Production Ready
Built for ongoing use, not just debugging:
- Robust error handling
- Performance considerations
- Configurable logging levels
- Resource cleanup

## Common Use Cases

### 1. Enhanced RAG Agent Issues
```bash
# Diagnose Enhanced RAG agent errors
python -m buttermilk.debug.cli diagnose-issue --flow osb

# Check for type checking issues
python -c "from buttermilk.debug.error_capture import analyze_type_checking_errors; print(analyze_type_checking_errors())"
```

### 2. Configuration Problems
```bash
# Validate all configuration files
python -m buttermilk.debug.cli validate-config --verbose

# Check specific storage configs
python -c "from buttermilk.debug.config_validator import validate_configuration; print(validate_configuration())"
```

### 3. Flow Execution Issues
```bash
# Test flow end-to-end
python -m buttermilk.debug.cli test-mcp-query --flow osb --query "test"

# Monitor real-time execution
python -m buttermilk.debug.cli stream-logs --filter 'textPayload:"osb"'
```

### 4. Dependency Issues
```bash
# Check for missing dependencies
python -m buttermilk.debug.cli validate-config

# Test specific imports
python -c "from buttermilk.debug.error_capture import TypeCheckingDiagnostics; print('Import successful')"
```

## Integration with Existing Code

### Adding Error Capture to Agents
```python
from buttermilk.debug.error_capture import ErrorCapture

class MyAgent(Agent):
    def __init__(self, **data):
        super().__init__(**data)
        self._error_capture = ErrorCapture()
    
    async def process(self, message):
        with self._error_capture.capture_context("process", agent_type="my_agent"):
            # Your processing logic
            pass
```

### Using Safe Type Checking
```python
from buttermilk.debug.error_capture import safe_isinstance_check

# Instead of: isinstance(obj, List[str])  # Causes runtime error
# Use:
is_list = safe_isinstance_check(obj, list)  # Safe
```

### Adding Configuration Validation
```python
from buttermilk.debug.config_validator import ConfigValidator

validator = ConfigValidator()
report = validator.validate_all()

if not report.is_valid:
    for error in report.errors:
        logger.error(f"Config error: {error.message}")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Missing optional dependencies
   - Solution: Run `validate-config` to check dependencies
   - Install missing packages or use graceful fallbacks

2. **Type Checking Errors**: Subscripted generics in isinstance
   - Solution: Use `safe_isinstance_check` instead
   - Replace `isinstance(obj, List[str])` with `isinstance(obj, list)`

3. **Configuration Errors**: Invalid YAML or missing fields
   - Solution: Run `validate-config --verbose` for detailed suggestions
   - Check file paths and required fields

4. **Flow Execution Timeouts**: Long-running operations
   - Solution: Use `stream-logs` to monitor real-time progress
   - Check GCP logs for performance bottlenecks

### Debug Commands Reference

| Command | Purpose | Example |
|---------|---------|---------|
| `diagnose-issue` | Complete diagnostic | `diagnose-issue --flow osb --comprehensive` |
| `test-startup` | Test daemon startup | `test-startup --flow osb --timeout 60` |
| `test-mcp-query` | Test specific query | `test-mcp-query --flow osb --query "test"` |
| `analyze-logs` | Analyze GCP logs | `analyze-logs --minutes-back 30` |
| `stream-logs` | Real-time log streaming | `stream-logs --filter 'textPayload:"error"'` |
| `validate-config` | Configuration validation | `validate-config --verbose` |

## Extension Points

### Adding New Validation Rules
```python
class CustomConfigValidator(ConfigValidator):
    def _validate_custom_component(self, config):
        # Your custom validation logic
        pass
```

### Adding New Test Cases
```python
class CustomMCPTester(MCPFlowTester):
    def test_custom_flow(self, flow_name):
        # Your custom testing logic
        pass
```

### Adding New Error Handlers
```python
class CustomErrorCapture(ErrorCapture):
    def _handle_custom_error(self, error):
        # Your custom error handling
        pass
```

## Best Practices

1. **Always validate configuration first** before debugging runtime issues
2. **Use real-time log streaming** for long-running operations
3. **Capture full context** when reporting issues
4. **Test incrementally** from basic health checks to comprehensive tests
5. **Document issues** using structured validation reports
6. **Follow CLAUDE.md principles** for any modifications

## Related Files

- `/workspaces/buttermilk/CLAUDE.md` - Development principles
- `/workspaces/buttermilk/docs/ARCHITECTURE.md` - System architecture
- `/workspaces/buttermilk/osb_flow_proof.py` - End-to-end flow validation
- `/workspaces/buttermilk/conf/` - Configuration files