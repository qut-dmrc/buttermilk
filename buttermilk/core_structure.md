# Buttermilk Core Structure

This document outlines the recommended organization for core objects in Buttermilk to avoid circular dependencies while maintaining logical structure.

## Overview

The Buttermilk framework appears to be built around several core concepts:
- Agents (that perform tasks)
- Messages (for communication between agents)
- Orchestrators (that coordinate agent execution)
- Data types (for representing various information)
- Configuration (for customizing behavior)

## Dependency Hierarchy

Based on the code review, here is a recommended dependency hierarchy that avoids circular imports:

```
Level 1: Core Types, Constants, Exceptions
Level 2: Configuration Classes 
Level 3: Message Contracts
Level 4: Base Agent
Level 5: Specialized Agents (LLMAgent, etc.)
Level 6: Orchestrators
```

## File Organization

### Level 1: Core Types (`types.py`)

This file should contain fundamental data structures and utility types with minimal dependencies:
- Basic data classes (`Record`, `MediaObj`)
- Type definitions
- Constants
- Simple helpers

**Dependencies**: None or minimal standard library

### Level 1: Exceptions (`exceptions.py`) 

This file should contain custom exceptions:
- `ProcessingError`
- `FatalError`
- Other custom exceptions

**Dependencies**: None or minimal standard library

### Level 2: Configuration (`config.py`)

Configuration classes that define how agents and systems are configured:
- `AgentConfig`
- `AgentVariants`
- `DataSourceConfig`
- `SaveInfo` 
- `Project`

**Dependencies**: 
- `types.py`
- `exceptions.py`

### Level 3: Message Contracts (`contract.py`)

Message types for communication between agents and orchestrators:
- `FlowMessage`
- `AgentInput`
- `AgentTrace`
- `ManagerMessage`
- Other message types

**Dependencies**:
- `types.py`
- `config.py`
- `exceptions.py`

### Level 4: Base Agent (`agent.py`)

The abstract base agent class:
- `Agent`
- `AgentOutput`
- `buttermilk_handler` decorator

**Dependencies**:
- `types.py`
- `config.py`
- `contract.py`
- `exceptions.py`

### Level 5: Specialized Agents

Classes that build upon the base Agent:
- `LLMAgent` in `llm.py`
- Other agent types

**Dependencies**:
- `types.py`
- `config.py`
- `contract.py` 
- `exceptions.py`
- `agent.py`

### Level 6: Orchestrator (`orchestrator.py`)

Classes that coordinate agent execution:
- `Orchestrator`
- `OrchestratorProtocol`

**Dependencies**:
- All of the above

## Implementation Recommendations

1. **Move Core Types to the Top of Dependency Chain**: Ensure `Record`, `SessionInfo`, and other fundamental types are defined in `types.py` without circular imports.

2. **Avoid Direct Imports of Higher-Level Components**: Lower-level modules should never import from higher-level modules.

3. **Use Forward References for Type Hints**: When a type hint would create a circular dependency, use string literals for the type (e.g., `"AgentConfig"` instead of `AgentConfig`).

4. **Isolate Third-Party Dependencies**: Wrap third-party libraries to isolate their interfaces and make them easier to replace or mock.

5. **Create Adapter Pattern for External Tools**: Use adapters to integrate with external tools rather than direct dependencies.

## Additional Recommendations

1. **Create a Core Constants Module**: Move constants to a dedicated `constants.py` file that can be imported by any module.

2. **Use Dependency Injection**: Pass dependencies to constructors rather than importing directly.

3. **Consider a Registry Pattern**: For dynamic loading of agents and tools.

4. **Make Import Paths Consistent**: Use absolute imports from the package root rather than relative imports where possible to improve clarity.

5. **Document Import Requirements**: Add comments above classes indicating their position in the dependency hierarchy.
