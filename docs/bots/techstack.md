# Buttermilk Technology Stack & Architecture

## Core Technologies

### Language & Runtime
- **Python 3.10+** (3.12 recommended)
  - Modern async/await support
  - Type hints throughout
  - Rich ecosystem for data science

### Package Management
- **uv**: Fast, reliable Python package manager
  - Always use: `uv run python ...`
  - Dev dependencies: `uv add --dev <package>`
  - Lock file ensures reproducibility

### Key Dependencies

#### Data Validation & Configuration
- **Pydantic v2**: Data validation and settings
  - Strict type checking
  - Automatic validation
  - JSON schema generation
  - **NEVER** change `extra="forbid"` to suppress errors

- **Hydra-core + OmegaConf**: Configuration management
  - Hierarchical configuration composition
  - Command-line overrides
  - Interpolation support (`${llms.general}`)
  - **ONLY** use YAML configs, no manual dicts

#### Async & Concurrency
- **asyncio**: Core async support
  - All I/O operations async
  - Concurrent agent execution
  - Non-blocking LLM calls

- **autogen-core**: Agent communication runtime
  - Message routing via topics
  - Agent lifecycle management
  - Background processing

#### AI/ML Integration
- **LLM Providers**:
  - Google (Gemini models)
  - Anthropic (Claude models)
  - OpenAI (GPT models)
  - Local models (Llama)

- **Model Configuration**: Loaded from GCP Secret `models.json`
  - Centralized model management
  - Environment-specific configs
  - Secure credential storage

#### Web & API
- **FastAPI**: REST API framework
  - Auto-generated OpenAPI docs
  - WebSocket support
  - Async request handling

- **httpx**: Async HTTP client
  - Connection pooling
  - Retry logic
  - Timeout management

#### Testing
- **pytest + pytest-asyncio**: Testing framework
  - Async test support
  - Fixtures for common setups
  - Parallel test execution

- **weave**: Tracing and debugging
  - LLM call tracking
  - Performance monitoring
  - Cost analysis

## Architecture Patterns

### Agent-Based Architecture

#### Core Components
1. **Agent** (`buttermilk._core.agent.Agent`)
   - Base class for all processing units
   - Async `_process()` method
   - State management
   - Tool definition support

2. **Orchestrator** (`buttermilk._core.orchestrator.Orchestrator`)
   - Manages flow execution
   - Routes messages between agents
   - Handles shared state/context

3. **Contract** (`buttermilk._core.contract`)
   - Pydantic models for messages
   - Type-safe communication
   - Validated data flow

#### Agent Types
- **LLMAgent**: Interfaces with language models
- **HostAgent**: Coordinates group chats
- **StructuredLLMHostAgent**: Tool-based coordination
- **Custom Agents**: Domain-specific processors

### Tool Definition System (Issue #83)

#### Components
- **@tool decorator**: Define callable tools
- **@MCPRoute**: Expose via MCP protocol
- **AgentToolDefinition**: Structured tool metadata
- **UnifiedRequest**: Consolidated request format

#### Benefits
- Type-safe tool calling
- Automatic schema generation
- MCP protocol support
- Backward compatibility

### Configuration Architecture

#### Hydra Composition
```yaml
defaults:
  - _self_
  - local              # Environment settings
  - llms: lite        # Model configs
  - flows: []         # Flow definitions

# Package directives for placement
- /agents@agents: researcher
- llms@bm.llms: full
```

#### Key Patterns
- **Interpolation**: `${llms.general}` references
- **Overrides**: `+flows=[osb,trans]` additions
- **Composition**: Build from multiple files
- **Validation**: Early fail with clear errors

### Data Flow Architecture

1. **YAML** → **Hydra** → **OmegaConf**
2. **OmegaConf** → **Pydantic** validation
3. **AgentInput** → **Agent** → **AgentOutput**
4. **AgentTrace** captures execution history

### Storage Architecture

#### Storage Types
- **Local**: File-based storage
- **BigQuery**: Structured data warehouse
- **GCS**: Object storage for large files
- **Vector DBs**: For RAG applications

#### Factory Pattern
- `StorageFactory.create_config()` for validation
- Discriminated unions for type safety
- Environment-specific configurations

## Design Decisions

### Why These Choices?

#### Python 3.10+
- Balance of stability and modern features
- Strong HASS researcher familiarity
- Rich data science ecosystem

#### Pydantic v2
- Performance improvements over v1
- Better validation messages
- Stronger type safety

#### Hydra over Alternatives
- Composable configurations
- Command-line flexibility
- Proven in ML research

#### Async-First
- Better resource utilization
- Responsive user experience
- Natural fit for I/O-heavy workloads

### Architecture Principles

#### Modularity
- Small, focused components
- Clear interfaces
- Composition over inheritance

#### Type Safety
- Pydantic models everywhere
- Type hints required
- Runtime validation

#### Configuration-Driven
- Behavior defined in YAML
- No hardcoded values
- Environment-specific overrides

#### Error Handling
- Fail fast with clear messages
- Validation at boundaries
- No defensive programming

## Integration Points

### External Services
- **Google Cloud Platform**
  - BigQuery for data warehouse
  - Secret Manager for credentials
  - Cloud Storage for artifacts

- **LLM Providers**
  - API-based integration
  - Async clients
  - Rate limiting/retry logic

### Protocol Support
- **MCP (Model Context Protocol)**
  - Tool discovery
  - Standardized calling
  - IDE integration potential

- **WebSocket**
  - Real-time flow monitoring
  - Interactive debugging
  - UI communication

## Performance Considerations

### Optimization Strategies
- **Async I/O**: Non-blocking operations
- **Connection Pooling**: Reuse HTTP connections
- **Batch Processing**: Group operations
- **Lazy Loading**: Load resources on-demand

### Monitoring
- **Structured Logging**: JSON-formatted logs
- **Trace Context**: Request correlation
- **Metrics Collection**: Performance tracking
- **Cost Tracking**: LLM usage monitoring

## Security Considerations

### Credential Management
- **Never commit secrets**
- Use environment variables
- GCP Secret Manager integration
- Service account authentication

### Data Protection
- Input validation everywhere
- SQL injection prevention
- Rate limiting on APIs
- Audit logging

## Future Architecture

### Planned Enhancements
- GraphQL API for complex queries
- Distributed agent execution
- Plugin marketplace
- Visual flow builder

### Scaling Considerations
- Horizontal scaling via queues
- Stateless agent design
- Cloud-native deployment
- Kubernetes support

Remember: Every technical decision should support HASS researchers' needs. Performance optimizations should never compromise usability or reproducibility.