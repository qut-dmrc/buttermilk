# Flow Architecture: Trans & Tox Flows Developer Guide

Dense technical reference for trans.yaml and tox.yaml flows in Buttermilk. For HASS researchers working with content moderation and bias evaluation.


## Flow Configurations

### Trans Flow (trans.yaml)
```yaml
orchestrator: buttermilk.orchestrators.groupchat.AutogenOrchestrator
data: tja_train (journalism training dataset)
agents: [judge, synth, differences]
observers: [spy, owl, scorer, fetch, host/sequencer]
criteria: trans (advocacy/journalism quality criteria)
```

### Tox Flow (tox.yaml)  
```yaml
orchestrator: buttermilk.orchestrators.groupchat.AutogenOrchestrator
data: drag (toxicity dataset)
agents: [judge, synth, differences] 
observers: [owl, scorer, host/sequencer]
criteria: hate (multiple toxicity criteria)
```

## Execution Path (API Default)

1. **CLI Entry**: `cli.py` @hydra.main loads config from `conf/`
2. **API Mode**: Creates FastAPI app via `create_fastapi_app(bm, flow_runner)`
3. **FlowRunner**: Receives a RunRequest and then instantiates orchestrator using `OrchestratorFactory.create_orchestrator()`
4. **AutogenOrchestrator**: Manages agent lifecycle and internal pub/sub communication via `SingleThreadedAgentRuntime`
5. **Session Management**: `SessionManager` handles WebSocket connections and cleanup for frontend clients

## Agent Class Hierarchy

```
Agent (ABC) <- AgentConfig
├── LLM-based agents (judge, synth, differences)
├── UIAgent <- Agent
│   ├── CLIUserAgent (console interaction)
│   └── WebSocketAgent (API interaction)
├── FlowControl agents
│   ├── HostAgent (conductor/sequencer role)
│   ├── ExplorerAgent (adaptive flow control)
│   └── LLMHostAgent (LLM-driven conductor)
└── Observer agents (spy, owl, scorer, fetch)
```

## Host/UI Options

### Host Agents (conductor role)
- **host/sequencer**: Basic sequential host (default)
- **host/explorer**: Adaptive exploration host  
- **host/assistant**: LLM-driven assistant host

### UI Options
- **console**: CLIUserAgent for terminal interaction
- **web**: WebSocket-based for browser/API clients
- **batch**: No UI for automated processing

## Component Flow

1. **Orchestrator Setup**: `AutogenOrchestrator._setup()` registers agents with `SingleThreadedAgentRuntime`
2. **Agent Registration**: Each agent becomes `AutogenAgentAdapter` wrapping Buttermilk `Agent`
3. **Topic Subscription**: Agents subscribe to main topic (`{bm.name}-{job}-{uuid}`) and role topics
4. **Message Flow**: Host agent sends `ConductorRequest` and `AgentInput`; other agents respond via internal pub/sub (managed by Autogen)
5. **Session Cleanup**: `SessionManager` handles resource cleanup and timeout

### Agent-based data processing

- Agent is the key abstraction for processing data
- Agents are stateful and can be configured to save data sent over internal Autogen pub/sub groupchat
- Agents are explicitly invoked with a `AgentInput` object that provides parameters and data, often including a `Record` item
- Subclasses of Agent must implement a `_process()` method that expects AgentInput and returns AgentOutput
- The invoke method is the normal interface point; it accepts an `AgentInput` and returns an `AgentTrace` that includes the `AgentOutput` and additional tracing information for full observability and logging. 
- `Observer` Agents are not invoked directly, but have their own logic that is usually triggered in their `_listen` events.
