# API Reference

This guide covers the Buttermilk REST API for running flows programmatically, managing sessions, and integrating with external systems.

## Base URL

When running locally:
```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication for local development. For production deployments, authentication should be configured through your cloud provider's authentication mechanisms.

## Content Types

The API supports both JSON and HTML responses based on the `Accept` header:

- `application/json` - JSON responses (default)
- `text/html` - HTML templates for browser requests

## Core Endpoints

### 1. Flow Execution

#### Run a Flow
```http
POST /flow/{flow_name}
```

Execute a specific flow with input data.

**Path Parameters:**
- `flow_name` (string) - Name of the flow to execute

**Request Body:**
```json
{
  "text": "Your input text here",
  "prompt": "Optional custom prompt",
  "record_id": "optional_record_id",
  "metadata": {
    "source": "api",
    "user_id": "user123"
  }
}
```

**Alternative Request Formats:**
```json
// Simple text input
{
  "q": "Your text content here"
}

// URL input
{
  "uri": "https://example.com/article"
}

// Custom parameters
{
  "flow": "custom_flow",
  "parameters": {
    "model": "gpt-4",
    "temperature": 0.7
  }
}
```

**Response (200 OK):**
```json
{
  "id": "flow_run_123",
  "flow_name": "trans_clean",
  "status": "completed",
  "results": [
    {
      "agent": "analyzer",
      "output": "Analysis results here",
      "metadata": {
        "model": "gemini-pro",
        "tokens_used": 1250,
        "processing_time": 2.3
      }
    }
  ],
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:30:15Z"
}
```

**Example Requests:**
```bash
# Basic text analysis
curl -X POST http://localhost:8000/flow/trans_clean \
  -H "Content-Type: application/json" \
  -d '{"text": "This is content to analyze"}'

# With custom prompt
curl -X POST http://localhost:8000/flow/trans_clean \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Article content here",
    "prompt": "Analyze this for sentiment and bias"
  }'

# URL input
curl -X POST http://localhost:8000/flow/content_analyzer \
  -H "Content-Type: application/json" \
  -d '{"uri": "https://example.com/article"}'
```

### 2. Flow Management

#### List Available Flows
```http
GET /api/flows
```

Get a list of all available flows.

**Response (200 OK):**
```json
[
  {
    "name": "trans_clean",
    "description": "Analysis of trans-related content",
    "version": "1.0.0",
    "agents": ["analyzer", "validator"],
    "status": "active"
  },
  {
    "name": "tox_allinone",
    "description": "Toxicity detection flow",
    "version": "1.0.0",
    "agents": ["toxicity_detector"],
    "status": "active"
  }
]
```

#### Get Flow Details
```http
GET /api/flows/{flow_name}
```

Get detailed information about a specific flow.

**Path Parameters:**
- `flow_name` (string) - Name of the flow

**Response (200 OK):**
```json
{
  "name": "trans_clean",
  "description": "Analysis of trans-related content",
  "version": "1.0.0",
  "agents": [
    {
      "name": "analyzer",
      "type": "LLMAgent",
      "model": "gemini-pro",
      "description": "Primary content analyzer"
    }
  ],
  "configuration": {
    "max_tokens": 4000,
    "temperature": 0.7,
    "timeout": 30
  },
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### 3. Session Management

#### Create Session
```http
GET /api/session
```

Create a new session for WebSocket connections.

**Query Parameters:**
- `session_id` (string, optional) - Existing session ID to validate

**Response (200 OK):**
```json
{
  "sessionId": "550e8400-e29b-41d4-a716-446655440000",
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Example:**
```bash
curl -X GET http://localhost:8000/api/session
```

### 4. WebSocket Connection

#### Connect to WebSocket
```http
GET /ws/{session_id}
```

Establish a WebSocket connection for real-time flow execution.

**Path Parameters:**
- `session_id` (string) - Session ID from `/api/session`

**WebSocket Messages:**

**Client to Server:**
```json
// Start a flow
{
  "type": "run_flow",
  "flow_name": "trans_clean",
  "data": {
    "text": "Content to analyze"
  }
}

// Send user response
{
  "type": "manager_response",
  "message": "User input or response"
}

// General UI message
{
  "type": "ui_message",
  "message": "User interface message"
}
```

**Server to Client:**
```json
// Agent announcement
{
  "type": "agent_announcement",
  "agent": "analyzer",
  "action": "joined",
  "timestamp": "2024-01-15T10:30:00Z"
}

// Agent execution result
{
  "type": "agent_trace",
  "agent": "analyzer",
  "result": "Analysis complete",
  "metadata": {
    "tokens_used": 1250,
    "processing_time": 2.3
  }
}

// Flow completion
{
  "type": "flow_complete",
  "flow_name": "trans_clean",
  "status": "completed",
  "results": [...]
}
```

### 5. Data Management

#### Get Records
```http
GET /api/records
```

Retrieve records for a specific flow.

**Query Parameters:**
- `flow` (string) - Flow name
- `include_scores` (boolean, optional) - Include summary scores
- `limit` (integer, optional) - Maximum number of records (default: 100)
- `offset` (integer, optional) - Pagination offset (default: 0)

**Response (200 OK):**
```json
[
  {
    "id": "record_123",
    "name": "Example Record",
    "content": "Record content here",
    "metadata": {
      "created_at": "2024-01-15T10:30:00Z",
      "dataset": "osb",
      "word_count": 156
    },
    "summary_scores": {
      "off_shelf_accuracy": 0.75,
      "custom_average": 0.635,
      "total_evaluations": 8
    }
  }
]
```

#### Get Record Details
```http
GET /api/flows/{flow_name}/records/{record_id}
```

Get detailed information about a specific record.

**Path Parameters:**
- `flow_name` (string) - Flow name
- `record_id` (string) - Record ID

**Response (200 OK):**
```json
{
  "id": "record_123",
  "name": "Example Record",
  "content": "Full record content here",
  "metadata": {
    "created_at": "2024-01-15T10:30:00Z",
    "dataset": "osb",
    "word_count": 156,
    "char_count": 892
  }
}
```

#### Get Record Scores
```http
GET /api/flows/{flow_name}/records/{record_id}/scores
```

Get evaluation scores for a specific record.

**Path Parameters:**
- `flow_name` (string) - Flow name
- `record_id` (string) - Record ID

**Response (200 OK):**
```json
{
  "record_id": "record_123",
  "off_shelf_results": {
    "GPT-4": {
      "correct": true,
      "score": 0.85,
      "label": "TOXIC",
      "confidence": 0.92
    },
    "Claude-3": {
      "correct": false,
      "score": 0.42,
      "label": "SAFE",
      "confidence": 0.78
    }
  },
  "custom_results": {
    "Judge-GPT4": {
      "step": "judge",
      "score": 0.88,
      "model": "gpt-4-0613",
      "criteria": "community_guidelines"
    }
  },
  "summary": {
    "off_shelf_accuracy": 0.75,
    "custom_average_score": 0.635,
    "total_evaluations": 8
  }
}
```

### 6. Monitoring and Health

#### Health Check
```http
GET /health
```

Check the health status of the API.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "0.3.3",
  "uptime": 3600,
  "active_sessions": 5,
  "available_flows": 3
}
```

#### Metrics
```http
GET /metrics
```

Get system metrics (Prometheus format).

**Response (200 OK):**
```
# HELP buttermilk_flows_total Total number of flows executed
# TYPE buttermilk_flows_total counter
buttermilk_flows_total{flow="trans_clean"} 42
buttermilk_flows_total{flow="tox_allinone"} 28

# HELP buttermilk_active_sessions Number of active WebSocket sessions
# TYPE buttermilk_active_sessions gauge
buttermilk_active_sessions 5
```

## Error Handling

The API uses standard HTTP status codes:

- `200 OK` - Success
- `400 Bad Request` - Invalid request data
- `401 Unauthorized` - Authentication required
- `404 Not Found` - Resource not found
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error

**Error Response Format:**
```json
{
  "error": {
    "code": "FLOW_NOT_FOUND",
    "message": "Flow 'invalid_flow' not found",
    "details": {
      "available_flows": ["trans_clean", "tox_allinone"]
    }
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Default limit**: 100 requests per minute per IP
- **Flow execution limit**: 10 concurrent flows per session
- **WebSocket connections**: 5 concurrent connections per IP

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705315800
```

## Integration Examples

### Python Client

```python
import requests
import json

# Basic flow execution
def run_flow(flow_name, text):
    url = f"http://localhost:8000/flow/{flow_name}"
    data = {"text": text}
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    
    return response.json()

# Usage
result = run_flow("trans_clean", "Content to analyze")
print(json.dumps(result, indent=2))
```

### JavaScript Client

```javascript
// Basic flow execution
async function runFlow(flowName, text) {
  const response = await fetch(`http://localhost:8000/flow/${flowName}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text })
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return await response.json();
}

// Usage
runFlow('trans_clean', 'Content to analyze')
  .then(result => console.log(result))
  .catch(error => console.error('Error:', error));
```

### WebSocket Client

```javascript
// WebSocket connection
const sessionResponse = await fetch('http://localhost:8000/api/session');
const session = await sessionResponse.json();

const ws = new WebSocket(`ws://localhost:8000/ws/${session.sessionId}`);

ws.onopen = () => {
  // Start a flow
  ws.send(JSON.stringify({
    type: 'run_flow',
    flow_name: 'trans_clean',
    data: { text: 'Content to analyze' }
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Received:', message);
  
  if (message.type === 'flow_complete') {
    console.log('Flow completed:', message.results);
  }
};
```

## Best Practices

### 1. Use Appropriate HTTP Methods
- `GET` for retrieving data
- `POST` for creating/executing flows
- `PUT` for updating resources
- `DELETE` for removing resources

### 2. Handle Errors Gracefully
```python
try:
    result = requests.post(url, json=data)
    result.raise_for_status()
    return result.json()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
        print("Flow not found")
    else:
        print(f"HTTP error: {e}")
except requests.exceptions.RequestException as e:
    print(f"Request error: {e}")
```

### 3. Use Sessions for Multiple Requests
```python
session = requests.Session()
session.headers.update({'User-Agent': 'MyApp/1.0'})

# Multiple requests with the same session
flows = session.get('http://localhost:8000/api/flows').json()
for flow in flows:
    result = session.post(f'http://localhost:8000/flow/{flow["name"]}', 
                         json={'text': 'test'})
```

### 4. Implement Proper Timeout Handling
```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Configure retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("http://", adapter)
session.mount("https://", adapter)

# Use with timeout
response = session.post(url, json=data, timeout=30)
```

## Advanced Features

### Batch Processing
```python
# Submit multiple flows for batch processing
batch_data = [
    {"text": "Content 1"},
    {"text": "Content 2"},
    {"text": "Content 3"}
]

for item in batch_data:
    result = requests.post(f'http://localhost:8000/flow/{flow_name}', 
                          json=item)
    print(f"Processed: {result.json()}")
```

### Streaming Responses
```python
# For long-running flows, use streaming
response = requests.post(url, json=data, stream=True)
for line in response.iter_lines():
    if line:
        update = json.loads(line)
        print(f"Progress: {update}")
```

## Deployment Considerations

### Environment Variables
```bash
export BUTTERMILK_API_HOST=0.0.0.0
export BUTTERMILK_API_PORT=8000
export BUTTERMILK_LOG_LEVEL=INFO
export GOOGLE_CLOUD_PROJECT=your-project-id
```

### Docker Deployment
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uv", "run", "python", "-m", "buttermilk.runner.cli", "run=api"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: buttermilk-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: buttermilk-api
  template:
    metadata:
      labels:
        app: buttermilk-api
    spec:
      containers:
      - name: buttermilk-api
        image: buttermilk:latest
        ports:
        - containerPort: 8000
        env:
        - name: GOOGLE_CLOUD_PROJECT
          value: "your-project-id"
```

## Further Reading

- [Flow Configuration](configuration.md)
- [Running Flows](flows.md)
- [WebSocket Integration](../reference/websocket-integration.md)
- [CLI Reference](cli-reference.md)
- [Troubleshooting](../reference/troubleshooting.md)