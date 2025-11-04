# Boundary Mocking Guide

Quick reference for mocking at system boundaries in Buttermilk tests.

## Network Boundaries

### HTTP with respx (Recommended)

```python
import respx
import httpx

@respx.mock
async def test_api_call():
    # Mock specific endpoint
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": "Response"}}]
        })
    )

    # Your code makes real httpx calls
    result = await my_function_that_calls_api()
    assert "Response" in result
```

### Multiple Endpoints

```python
@respx.mock
async def test_multiple_apis():
    # Setup multiple mocks
    respx.get("https://api.github.com/user").mock(
        return_value=httpx.Response(200, json={"login": "user"})
    )
    respx.post(re.compile(r"https://api\.example\.com/.*")).mock(
        return_value=httpx.Response(201, json={"created": True})
    )

    # Test your code
    result = await fetch_and_create()
    assert result["user"] == "user"
    assert result["created"] is True
```

## Filesystem Boundaries

### Using tmp_path Fixture

```python
def test_file_operations(tmp_path):
    # Create test files
    test_file = tmp_path / "data.json"
    test_file.write_text('{"key": "value"}')

    # Test your file processing
    result = process_json_file(test_file)
    assert result["key"] == "value"

    # Check output files
    output = tmp_path / "output.txt"
    assert output.exists()
```

### Mocking open()

```python
from unittest.mock import mock_open, patch

def test_read_file():
    mock_data = "file content"
    with patch("builtins.open", mock_open(read_data=mock_data)):
        result = read_config_file("/etc/config")
        assert result == "file content"
```

## Time Boundaries

### Using freezegun

```python
from freezegun import freeze_time
from datetime import datetime

@freeze_time("2024-01-15 10:30:00")
def test_timestamp():
    record = create_timestamped_record()
    assert record.created_at == datetime(2024, 1, 15, 10, 30, 0)

# Time progression
def test_timeout():
    with freeze_time("2024-01-01") as frozen_time:
        start = datetime.now()
        frozen_time.tick(delta=timedelta(seconds=30))
        end = datetime.now()
        assert (end - start).seconds == 30
```

## Environment Variables

### Using monkeypatch

```python
def test_env_config(monkeypatch):
    # Set environment variables
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("DEBUG", "true")

    # Test configuration loading
    config = load_config()
    assert config.api_key == "test-key"
    assert config.debug is True
```

### Multiple Environment Changes

```python
def test_environment(monkeypatch):
    env_vars = {
        "DATABASE_URL": "postgresql://test",
        "REDIS_URL": "redis://localhost",
        "SECRET_KEY": "test-secret"
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    config = initialize_app()
    assert config.database_url == "postgresql://test"
```

## Randomness

### Controlling Random

```python
import random

def test_random_selection():
    random.seed(42)  # Deterministic randomness

    result = select_random_items(["a", "b", "c", "d"])
    assert result == ["c", "a"]  # Always same with seed 42
```

### Mocking random functions

```python
from unittest.mock import patch

@patch("random.choice")
def test_random_choice(mock_choice):
    mock_choice.return_value = "specific_value"

    result = get_random_option()
    assert result == "specific_value"
```

## External Processes

### Mocking subprocess

```python
from unittest.mock import patch, MagicMock

@patch("subprocess.run")
def test_shell_command(mock_run):
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="command output",
        stderr=""
    )

    result = run_external_tool("ls")
    assert result == "command output"
    mock_run.assert_called_once_with(["ls"], capture_output=True, text=True)
```

## Cloud Services

### Google Cloud Storage

```python
from unittest.mock import patch, MagicMock

@patch("google.cloud.storage.Client")
def test_gcs_upload(mock_client):
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    upload_to_gcs("file.txt", "bucket-name")

    mock_blob.upload_from_filename.assert_called_once_with("file.txt")
```

### BigQuery

```python
@patch("google.cloud.bigquery.Client")
def test_bigquery_insert(mock_client):
    mock_table = MagicMock()
    mock_client.return_value.get_table.return_value = mock_table
    mock_client.return_value.insert_rows_json.return_value = []

    result = insert_records([{"id": 1, "name": "test"}])
    assert result is True
```

## Complex External Systems

### Simple Test Doubles Instead of Mocks

```python
class FakeOpenAIClient:
    """Test double for OpenAI API."""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    async def create(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        self.calls.append(prompt)

        for pattern, response in self.responses.items():
            if pattern in prompt:
                return {"choices": [{"message": {"content": response}}]}

        return {"choices": [{"message": {"content": "default"}}]}

# Use in tests
async def test_with_fake_client():
    client = FakeOpenAIClient(responses={
        "weather": "It's sunny",
        "news": "Breaking news"
    })

    agent = Agent(client=client)
    result = await agent.query("What's the weather?")

    assert "sunny" in result
    assert len(client.calls) == 1
```

## Pytest Fixtures for Common Boundaries

```python
# conftest.py
import pytest
import respx
from freezegun import freeze_time

@pytest.fixture
def mock_api():
    """Pre-configured API mocks."""
    with respx.mock:
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
        )
        yield respx

@pytest.fixture
def frozen_time():
    """Frozen time at a specific date."""
    with freeze_time("2024-01-01"):
        yield

@pytest.fixture
def temp_config(tmp_path, monkeypatch):
    """Temporary config with isolated filesystem and env."""
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path))
    config_file = tmp_path / "config.yaml"
    config_file.write_text("key: value")
    return config_file
```

## Key Principles

1. **Mock as close to the boundary as possible** - Don't mock your wrapper, mock the actual external call
1. **Use appropriate tools** - respx for HTTP, freezegun for time, etc.
1. **Keep mocks simple** - Complex mock setups indicate design problems
1. **Prefer test doubles over mocks** for complex external systems
1. **Never mock your own code** - Only mock external dependencies
