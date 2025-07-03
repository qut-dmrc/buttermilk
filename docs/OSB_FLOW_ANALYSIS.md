# OSB Flow Configuration Analysis

## Issue Summary
The OSB flow agents aren't being loaded when running the API because the `flows` configuration group isn't specified in the command line.

## Root Cause
1. The OSB flow configuration (`conf/flows/osb.yaml`) correctly defines agents using Hydra defaults
2. The flow has 4 agents (researcher, policy_analyst, fact_checker, explorer) and 1 observer (llm_host)
3. The API run configuration (`conf/run/api.yaml`) references `flows: ${flows}` 
4. However, there's no default `flows` configuration loaded, so `${flows}` interpolation fails
5. This results in the FlowRunner having an empty flows dictionary

## Configuration Structure

### OSB Flow (`conf/flows/osb.yaml`)
```yaml
defaults:
  - _self_
  - /agents/rag@osb.agents: 
    - ragzot
    - researcher
    - policy_analyst
    - fact_checker
    - explorer
  - /agents@osb.observers:
    - host/llm_host
  # ... storage configs ...

osb:
  orchestrator: buttermilk.orchestrators.groupchat.AutogenOrchestrator
  name: "OSB Interactive Flow"
  agents: {}     # Empty, but filled by defaults
  observers: {}  # Empty, but filled by defaults
```

### API Run Config (`conf/run/api.yaml`)
```yaml
run:
  _target_: buttermilk.runner.flowrunner.FlowRunner
  mode: api
  flows: ${flows}  # References a 'flows' key that doesn't exist by default
```

## Solution

To run the API with the OSB flow, you must specify the flows configuration group:

```bash
# Correct command
uv run python -m buttermilk.runner.cli run=api flows=osb

# Or to run multiple flows
uv run python -m buttermilk.runner.cli run=api flows=osb,rag,test
```

## Why This Happens

1. Hydra's configuration composition doesn't automatically load all available flows
2. The `flows` configuration group must be explicitly specified
3. The empty `agents: {}` and `observers: {}` in the OSB flow are correct - they get populated by the defaults mechanism
4. The issue is NOT with the OSB flow configuration itself, but with how it's loaded

## Verification

Running the test script shows:
- OSB flow loads correctly with 4 agents and 1 observer when accessed directly
- The agents are properly configured with all their parameters
- The issue only occurs when the flows configuration group isn't specified

## Recommendations

1. Update documentation to clarify that flows must be explicitly specified when running the API
2. Consider adding a default flows configuration or improving error messages
3. The current OSB flow configuration is correct and doesn't need changes