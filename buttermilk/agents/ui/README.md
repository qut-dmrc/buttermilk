# UI Agents for Buttermilk

This package provides a flexible UI abstraction layer that allows different user interface implementations to be used interchangeably with any flow.

## Architecture

The UI system follows a proxy pattern:

```
┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│                │      │                │      │                │
│  Flow Config   │ uses │   UIProxyAgent │ uses │  Concrete UI   │
│  (YAML file)   │─────▶│   (Proxy)      │─────▶│  Implementation│
│                │      │                │      │                │
└────────────────┘      └────────────────┘      └────────────────┘
                                                        ▲
                                                        │
                                                        │
                                                 ┌──────┴───────┐
                                                 │  UI Registry  │
                                                 └──────────────┘
```

1. **UIProxyAgent**: Acts as an intermediary between the flow orchestrator and the actual UI implementation. It dynamically connects to a specific UI implementation at runtime based on configuration.

2. **UI Registry**: Maintains a mapping of UI types to their implementations, allowing dynamic registration/lookup.

3. **Concrete UI Implementations**:
   - `WebUIAgent`: Handles web interface interactions via WebSockets
   - `ConsoleUIAgent`: Provides a terminal-based interface
   - `SlackThreadChatUIAgent`: Integrates with Slack for chat-based interaction

## Usage

### In Flow Configuration

Use the `interface/proxy` agent in your flow configuration:

```yaml
defaults:
  - /agents@observers:
    - spy
    - owl
    - host/sequencer
    - interface/proxy  # Use the proxy that can switch UI implementations
```

### Command Line

When running a flow, specify the UI type using the `ui_type` parameter:

```bash
# Run with web UI (default)
python -m buttermilk.runner.cli ui=api flows=trans

# Run with console UI
python -m buttermilk.runner.cli ui=api flows=trans ui_type=console

# Run with Slack UI
python -m buttermilk.runner.cli ui=api flows=trans ui_type=slack
```

### In Code

When creating a `FlowRunner` instance, set the `ui_type` parameter:

```python
from buttermilk.runner.flowrunner import FlowRunner

# Create FlowRunner with console UI
flow_runner = FlowRunner(
    bm=bm,
    flows=flows,
    ui="api",
    ui_type="console"
)
```

## Adding New UI Implementations

1. Create a new UI implementation class:

```python
# my_custom_ui.py
from buttermilk.agents.ui.generic import UIAgent

class MyCustomUIAgent(UIAgent):
    """Custom UI implementation."""
    
    async def _process(self, *, inputs, cancellation_token=None, **kwargs):
        # Implementation...
        pass
        
    # Other required methods...
```

2. Register the implementation:

```python
# In your app's startup code:
from buttermilk.agents.ui.registry import register_ui
from my_package.my_custom_ui import MyCustomUIAgent

register_ui("my_custom", MyCustomUIAgent)
```

3. Use it in your flow:

```bash
python -m buttermilk.runner.cli ui=api flows=trans ui_type=my_custom
```

## Benefits

- **Consistency**: Same flow configuration works with any UI implementation
- **Flexibility**: Switch UIs at runtime without changing flow configurations
- **Extensibility**: Easily add new UI implementations without modifying flows
- **Simplicity**: Clear separation between flow logic and UI concerns
