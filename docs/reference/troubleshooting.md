# Troubleshooting Guide

This comprehensive guide helps you diagnose and resolve common issues with Buttermilk.

## Table of Contents
- [Quick Diagnostics](#quick-diagnostics)
- [Installation Issues](#installation-issues)
- [Configuration Problems](#configuration-problems)
- [Authentication Errors](#authentication-errors)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [API Problems](#api-problems)
- [Testing Issues](#testing-issues)
- [Getting Help](#getting-help)

## Quick Diagnostics

### System Health Check

```bash
# Check Python version
python3 --version

# Check uv installation
uv --version

# Verify Buttermilk installation
uv run python -c "import buttermilk; print('✅ Buttermilk installed')"

# Check authentication
gcloud auth list

# Test basic configuration
uv run python -m buttermilk.runner.cli --help
```

### Environment Check

```bash
# Check environment variables
echo "Python: $(python3 --version)"
echo "GCP Project: ${GOOGLE_CLOUD_PROJECT}"
echo "Environment: ${BUTTERMILK_ENV}"

# Check working directory
pwd
ls -la conf/

# Check permissions
ls -la ~/.config/gcloud/
```

## Installation Issues

### uv Command Not Found

**Problem:** `bash: uv: command not found`

**Solution:**
```bash
# Install uv
pip install uv

# Or using pipx
pipx install uv

# Verify installation
uv --version
```

### Python Version Issues

**Problem:** Python 3.10+ required

**Solution:**
```bash
# Check current version
python3 --version

# Install Python 3.12 (Ubuntu/Debian)
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.12 python3.12-pip

# Use pyenv for version management
curl https://pyenv.run | bash
pyenv install 3.12.0
pyenv global 3.12.0
```

### Dependency Installation Failures

**Problem:** Package installation fails

**Solution:**
```bash
# Clean uv cache
uv cache clean

# Update uv
pip install --upgrade uv

# Reinstall dependencies
rm -rf .venv
uv install

# Install with verbose output
uv install --verbose
```

### System Dependencies Missing

**Problem:** Missing system libraries

**Solution:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
  build-essential \
  python3-dev \
  libssl-dev \
  libffi-dev \
  pkg-config

# macOS
brew install openssl libffi

# CentOS/RHEL
sudo yum install -y \
  gcc \
  python3-devel \
  openssl-devel \
  libffi-devel
```

## Configuration Problems

### Flow Not Found

**Problem:** `Flow 'my_flow' not found`

**Solution:**
```bash
# List available flows
uv run python -m buttermilk.runner.cli --info searchpath

# Check configuration directories
ls -la conf/flows/

# Verify flow file exists
ls -la conf/flows/my_flow.yaml

# Check flow configuration
uv run python -m buttermilk.runner.cli flow=my_flow --cfg job
```

### Configuration Validation Errors

**Problem:** `Key 'X' is not in struct`

**Solution:**
```bash
# Use + prefix for new keys
uv run python -m buttermilk.runner.cli +new_key=value

# Check configuration structure
uv run python -m buttermilk.runner.cli --cfg job | head -50

# Validate configuration file
uv run python -c "
import yaml
with open('conf/flows/my_flow.yaml') as f:
    config = yaml.safe_load(f)
    print('✅ Valid YAML')
"
```

### Interpolation Errors

**Problem:** `Interpolation key 'X' not found`

**Solution:**
```bash
# Enable debug output
export HYDRA_FULL_ERROR=1

# Check what's available
uv run python -m buttermilk.runner.cli --cfg job | grep -A 10 -B 10 "interpolation_key"

# Fix missing dependencies
uv run python -m buttermilk.runner.cli llms=lite  # Ensure llms config is loaded
```

### Override Syntax Errors

**Problem:** `Could not override 'X'. No match in defaults`

**Solution:**
```bash
# Wrong: trying to override non-existent key
uv run python -m buttermilk.runner.cli flows=osb

# Correct: add new key
uv run python -m buttermilk.runner.cli +flows=osb

# Check available override options
uv run python -m buttermilk.runner.cli --help
```

## Authentication Errors

### Google Cloud Authentication

**Problem:** `Default credentials not found`

**Solution:**
```bash
# Authenticate with user account
gcloud auth login

# Set application default credentials
gcloud auth application-default login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Verify authentication
gcloud auth list
gcloud auth application-default print-access-token
```

### Service Account Issues

**Problem:** Service account permissions

**Solution:**
```bash
# Check service account
gcloud auth list

# Activate service account
gcloud auth activate-service-account --key-file=path/to/key.json

# Set quota project
gcloud auth application-default set-quota-project YOUR_PROJECT_ID

# Verify permissions
gcloud projects get-iam-policy YOUR_PROJECT_ID
```

### API Access Issues

**Problem:** API access denied

**Solution:**
```bash
# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable bigquery.googleapis.com

# Check API status
gcloud services list --enabled

# Verify credentials have required roles
gcloud projects get-iam-policy YOUR_PROJECT_ID --flatten="bindings[].members" --format="table(bindings.role,bindings.members)"
```

## Runtime Errors

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'buttermilk'`

**Solution:**
```bash
# Install in development mode
uv install -e .

# Check Python path
uv run python -c "import sys; print(sys.path)"

# Verify installation
uv run python -c "import buttermilk; print(buttermilk.__file__)"
```

### Agent Initialization Failures

**Problem:** Agent fails to initialize

**Solution:**
```bash
# Check agent configuration
uv run python -m buttermilk.runner.cli flow=my_flow --cfg job | grep -A 20 agents

# Enable debug logging
uv run python -m buttermilk.runner.cli flow=my_flow +debug=true

# Test agent individually
uv run python -c "
from buttermilk.agents.llm import LLMAgent
agent = LLMAgent(agent_id='test', role='TEST', description='Test agent')
print('✅ Agent created successfully')
"
```

### LLM Connection Issues

**Problem:** LLM API calls fail

**Solution:**
```bash
# Check model configuration
uv run python -m buttermilk.runner.cli --cfg job | grep -A 10 llms

# Test LLM connection
uv run python -c "
from buttermilk.llm.client import LLMClient
client = LLMClient(model='gemini-pro')
print('✅ LLM client created')
"

# Check API quotas
gcloud logging read 'resource.type="consumed_api"' --limit=10
```

### Memory Issues

**Problem:** Out of memory errors

**Solution:**
```bash
# Check memory usage
htop
free -h

# Reduce batch size
uv run python -m buttermilk.runner.cli run=batch flow=my_flow run.batch_size=1

# Limit workers
uv run python -m buttermilk.runner.cli run=batch flow=my_flow run.max_workers=1

# Monitor memory usage
uv run python -m buttermilk.runner.cli run=batch flow=my_flow +profile=true
```

## Performance Issues

### Slow Execution

**Problem:** Flows take too long to execute

**Diagnosis:**
```bash
# Enable profiling
uv run python -m buttermilk.runner.cli run=console flow=my_flow +profile=true

# Check resource usage
uv run python -m buttermilk.runner.cli run=console flow=my_flow +monitor=true

# Time specific operations
time uv run python -m buttermilk.runner.cli run=console flow=my_flow +text="test"
```

**Solutions:**
```bash
# Use faster models
uv run python -m buttermilk.runner.cli flow=my_flow llms=lite

# Reduce token limits
uv run python -m buttermilk.runner.cli flow=my_flow agents.0.max_tokens=500

# Enable parallel processing
uv run python -m buttermilk.runner.cli run=batch flow=my_flow run.parallel=true
```

### Rate Limiting

**Problem:** API rate limits exceeded

**Solution:**
```bash
# Add delays between requests
uv run python -m buttermilk.runner.cli run=batch flow=my_flow run.delay=2

# Reduce concurrency
uv run python -m buttermilk.runner.cli run=batch flow=my_flow run.max_concurrent=1

# Check rate limits
gcloud logging read 'protoPayload.methodName="google.cloud.aiplatform.v1.PredictionService.Predict"' --limit=10
```

### Resource Leaks

**Problem:** Memory/connection leaks

**Solution:**
```bash
# Check for unclosed connections
lsof -p $(pgrep -f buttermilk)

# Monitor resource usage over time
watch -n 5 'ps aux | grep buttermilk'

# Enable connection pooling
uv run python -m buttermilk.runner.cli flow=my_flow +connection_pool=true
```

## API Problems

### Server Won't Start

**Problem:** FastAPI server fails to start

**Solution:**
```bash
# Check port availability
netstat -tulpn | grep :8000

# Kill existing processes
pkill -f "buttermilk.runner.cli"

# Start with different port
uv run python -m buttermilk.runner.cli run=api server.port=8080

# Check logs
uv run python -m buttermilk.runner.cli run=api +verbose=true
```

### API Endpoint Errors

**Problem:** 404 or 500 errors

**Solution:**
```bash
# Check available endpoints
curl -X GET http://localhost:8000/docs

# Test health endpoint
curl -X GET http://localhost:8000/health

# Check server logs
tail -f /tmp/buttermilk_*.log

# Enable debug mode
uv run python -m buttermilk.runner.cli run=api +debug=true
```

### WebSocket Connection Issues

**Problem:** WebSocket connections fail

**Solution:**
```bash
# Check session creation
curl -X GET http://localhost:8000/api/session

# Test WebSocket connection
wscat -c ws://localhost:8000/ws/SESSION_ID

# Check CORS settings
curl -H "Origin: http://localhost:3000" -H "Access-Control-Request-Method: GET" -H "Access-Control-Request-Headers: X-Requested-With" -X OPTIONS http://localhost:8000/api/session
```

## Testing Issues

### Test Failures

**Problem:** Tests fail unexpectedly

**Solution:**
```bash
# Run tests with verbose output
uv run pytest -v

# Run specific test
uv run pytest tests/test_specific.py::test_function -v

# Check test environment
uv run pytest --collect-only

# Enable debug output
uv run pytest --tb=long --capture=no
```

### Authentication in Tests

**Problem:** Tests fail due to authentication

**Solution:**
```bash
# Verify authentication
gcloud auth list
gcloud auth application-default print-access-token

# Set test project
export GOOGLE_CLOUD_PROJECT=test-project

# Use service account for CI
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Skip auth-required tests
uv run pytest -m "not auth_required"
```

### Mock Issues

**Problem:** Mocking not working correctly

**Solution:**
```python
# Use proper mock paths
from unittest.mock import patch

# Wrong: mocking the import
@patch('buttermilk.llm.client.LLMClient')

# Correct: mocking the actual usage
@patch('buttermilk.agents.llm.LLMAgent.llm_client')

# Use MagicMock for complex objects
from unittest.mock import MagicMock
mock_client = MagicMock()
mock_client.generate.return_value = "test response"
```

## Common Error Messages

### "extra_forbidden" Error

**Problem:** Pydantic validation error

**Solution:**
```python
# Don't change model validation settings
# Instead, fix the data structure

# Wrong approach
class MyModel(BaseModel):
    model_config = ConfigDict(extra="allow")  # Don't do this

# Correct approach
class MyModel(BaseModel):
    field1: str
    field2: Optional[str] = None
    # Add all expected fields
```

### "No module named" Error

**Problem:** Import path issues

**Solution:**
```bash
# Check installation
uv run python -c "import buttermilk; print('OK')"

# Fix import paths
# Wrong: from buttermilk.agents import Agent
# Correct: from buttermilk._core.agent import Agent

# Check available modules
uv run python -c "import buttermilk; print(dir(buttermilk))"
```

### "Connection refused" Error

**Problem:** Service not available

**Solution:**
```bash
# Check service status
curl -I http://localhost:8000/health

# Verify service is running
ps aux | grep buttermilk

# Check firewall/network
telnet localhost 8000

# Use correct host/port
uv run python -m buttermilk.runner.cli run=api server.host=0.0.0.0 server.port=8000
```

## Debug Commands

### Configuration Debugging

```bash
# Show resolved configuration
uv run python -m buttermilk.runner.cli --cfg job

# Show configuration search paths
uv run python -m buttermilk.runner.cli --info searchpath

# Show Hydra configuration
uv run python -m buttermilk.runner.cli --cfg hydra

# Validate configuration
uv run python -c "
from hydra import compose, initialize_config_store
try:
    with initialize_config_store(config_path='conf'):
        cfg = compose(config_name='config')
        print('✅ Configuration valid')
except Exception as e:
    print(f'❌ Configuration error: {e}')
"
```

### Runtime Debugging

```bash
# Enable all debug output
export HYDRA_FULL_ERROR=1
export BUTTERMILK_DEBUG=1

# Run with profiling
uv run python -m buttermilk.runner.cli run=console flow=my_flow +profile=true

# Monitor system resources
uv run python -m buttermilk.runner.cli run=console flow=my_flow +monitor=true

# Enable verbose logging
uv run python -m buttermilk.runner.cli run=console flow=my_flow logging.level=DEBUG
```

### Network Debugging

```bash
# Check API connectivity
curl -v http://localhost:8000/health

# Test WebSocket
wscat -c ws://localhost:8000/ws/test-session

# Check DNS resolution
nslookup aiplatform.googleapis.com

# Test SSL/TLS
openssl s_client -connect aiplatform.googleapis.com:443
```

## Getting Help

### Self-Diagnosis Checklist

Before asking for help, check:

- [ ] Python version is 3.10+
- [ ] uv is installed and working
- [ ] Google Cloud authentication is set up
- [ ] Configuration files exist and are valid
- [ ] Required APIs are enabled
- [ ] Network connectivity is working
- [ ] Sufficient disk space and memory
- [ ] No conflicting processes running

### Log Collection

```bash
# Collect system information
uv run python -m buttermilk.runner.cli --info system > system_info.txt

# Collect configuration
uv run python -m buttermilk.runner.cli --cfg job > config_dump.yaml

# Collect logs
cp /tmp/buttermilk_*.log ./

# Create diagnostic archive
tar -czf buttermilk_diagnostic.tar.gz system_info.txt config_dump.yaml *.log
```

### Community Resources

- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share ideas
- **Documentation**: Check [user guides](../user-guide/) and [developer guides](../developer-guide/)
- **Email Support**: [nic@suzor.com](mailto:nic@suzor.com)

### Creating Bug Reports

When reporting bugs, include:

1. **System Information**:
   - Operating system
   - Python version
   - uv version
   - Buttermilk version

2. **Configuration**:
   - Relevant configuration files
   - Command that failed
   - Environment variables

3. **Error Output**:
   - Complete error messages
   - Stack traces
   - Log files

4. **Steps to Reproduce**:
   - Minimal example
   - Expected vs actual behavior
   - Screenshots if relevant

### Template for Bug Reports

```markdown
## Bug Report

### Environment
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.12.0]
- uv: [e.g., 0.1.0]
- Buttermilk: [e.g., 0.3.3]

### Command
```bash
uv run python -m buttermilk.runner.cli run=console flow=my_flow
```

### Expected Behavior
[Describe what should happen]

### Actual Behavior
[Describe what actually happens]

### Error Output
```
[Paste complete error message]
```

### Configuration
```yaml
[Paste relevant configuration]
```

### Additional Context
[Any other relevant information]
```

## Performance Optimization

### Memory Optimization

```bash
# Reduce batch size
uv run python -m buttermilk.runner.cli run=batch flow=my_flow run.batch_size=1

# Limit concurrent operations
uv run python -m buttermilk.runner.cli run=batch flow=my_flow run.max_concurrent=1

# Use streaming for large datasets
uv run python -m buttermilk.runner.cli run=batch flow=my_flow data.streaming=true
```

### Speed Optimization

```bash
# Use lighter models
uv run python -m buttermilk.runner.cli flow=my_flow llms=lite

# Reduce token limits
uv run python -m buttermilk.runner.cli flow=my_flow agents.0.max_tokens=500

# Enable caching
uv run python -m buttermilk.runner.cli flow=my_flow +cache=true
```

### Network Optimization

```bash
# Use connection pooling
uv run python -m buttermilk.runner.cli flow=my_flow +connection_pool=true

# Adjust timeout settings
uv run python -m buttermilk.runner.cli flow=my_flow run.timeout=300

# Enable retry logic
uv run python -m buttermilk.runner.cli flow=my_flow run.max_retries=3
```

Remember: When troubleshooting, always start with the simplest explanation and work your way up to more complex issues. Most problems have simple solutions once you identify the root cause.