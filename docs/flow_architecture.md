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
3. **FlowRunner**: Instantiates orchestrator using `OrchestratorFactory.create_orchestrator()`
4. **AutogenOrchestrator**: Manages agent lifecycle via `SingleThreadedAgentRuntime`
5. **Session Management**: `SessionManager` handles WebSocket connections and cleanup

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
4. **Message Flow**: Host agent sends `ConductorRequest`, other agents respond via internal pub/sub (managed by Autogen)
5. **Session Cleanup**: `SessionManager` handles resource cleanup and timeout

### Agent vs Observer Roles
- **Agents**: Core processing (judge content, synthesize, find differences)
- **Observers**: Monitoring, scoring, data collection, flow control

## Configuration Merge Order
1. Flow defaults (`conf/flows/{flow}.yaml`)
2. Agent defaults (`conf/agents/{agent}.yaml`) 
3. Runtime parameters (CLI args, API requests)
4. Session-specific overrides

## Development Notes
- Fresh orchestrator instance per flow run (no state leakage)
- Session isolation via unique topic IDs
- Async cleanup with timeout protection
- Weave tracing for debugging/monitoring
- Resource tracking for memory management
- Each substantive data processing operation loggged to BigQuery (AgentTrace object; spy agent uploads all AgentTrace objects in groupchat)


## Quick Reference
- **Start flow**: `FlowRunner.run_flow(RunRequest)`
- **Agent factory**: `OrchestratorFactory.create_orchestrator()`
- **Session handling**: `SessionManager.get_or_create_session()`
- **Cleanup**: `FlowRunContext.cleanup()` -> `SessionResources.cleanup()`

