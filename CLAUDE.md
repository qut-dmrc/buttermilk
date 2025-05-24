# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Buttermilk Project Development Guidelines
Primary Goal: Build reproducible, traceable, and HASS-researcher-friendly tools.

* HASS-Centric Design: Prioritize usability, clarity, extensibility, and accessibility for Humanities, Arts, and Social Sciences (HASS) researchers.
* Reproducibility & Traceability: Ensure that experiments, data processing, and results are reproducible and traceable. Design for robust logging and versioning.
* Modularity: Maintain a modular architecture. Core components are Agent, Orchestrator, and Contract. Prefer creating new Agent/Orchestrator subclasses over modifying core components.
* Asynchronous Operations: Embrace async/await for I/O-bound tasks, LLM calls, and concurrent operations to ensure responsiveness.
* Pay down tech debt: The project is still in "very early stages". There is left-over code that does not align with these principles or the direction of the project. You should seek confirmation but not hesitate to suggest removing or refactoring code that creates unecessary complexity and maintenance overhead.
* You might find a lot of linting errors. If you've been asked to complete a task, you should only fix the critical errors or problems you introduced. Don't go aruond looking for other stuff to do, although you should make a note of problems you noticed but didn't fix in your final report.

## Development Commands

- **Install dependencies**: `uv install` (installs main + ml + dev groups by default)
- **Python command**: `uv run python` (all Python commands should use this prefix)
- **Run all tests**: `uv run python -m pytest`
- **Run single test**: `uv run python -m pytest tests/path/to/test_file.py::test_function`
- **Integration tests**: Tests marked with `@pytest.mark.integration` are excluded by default
- **Lint**: `uv run python -m ruff buttermilk`
- **Format**: `uv run python -m black buttermilk`
- **Type checking**: Uses mypy (configured in pyproject.toml)
- **Main CLI**: `uv run python buttermilk/runner/cli.py`

## Architecture Overview

Buttermilk is an AI and data pipeline framework for HASS (Humanities, Arts, Social Sciences) scholars. The core architecture follows a dependency hierarchy to avoid circular imports:

### Core Concepts
- **Flows**: Complete research pipelines
- **Records**: Immutable data pieces with unique IDs and rich metadata
- **Agents**: Specialized components that perform specific tasks (LLM interactions, data collection, etc.) Primary __call__ method takes AgentInput and returns AgentOutput.
- **Orchestrators**: Coordinate Flow execution and manage Agent interactions

### Dependency Hierarchy (buttermilk/_core/)
```
Level 1: types.py, exceptions.py, constants.py (core types, no dependencies)
Level 2: config.py (configuration classes)
Level 3: contract.py (message contracts for inter-component communication)
Level 4: agent.py (base Agent class)
Level 5: llms.py, specialized agents (build on base Agent)
Level 6: orchestrator.py (coordinates everything)
```

### Configuration System
- Uses **Hydra** for configuration management
- Configuration files in `conf/` directory (YAML format)
- Main config: `conf/testing.yaml`
- Supports command-line overrides for experimentation
- Modular configs: separate files for agents, data sources, flows, LLMs, etc.

### Key Modules
- **buttermilk/_core/**: Core framework components
- **buttermilk/agents/**: Specialized agent implementations
- **buttermilk/runner/**: CLI and execution logic
- **buttermilk/api/**: FastAPI back-end API interface
- **conf/**: Hydra configuration files
- **templates/**: Jinja2 templates for prompts and formatting

### Execution Modes
The CLI supports multiple modes via Hydra configuration:
- `console`: Interactive CLI mode
- `api`: FastAPI web server
- `batch`: Create batch jobs in queue
- `chat`: Web UI
- `slackbot`: Slack integration

### Key Technologies & Practices:
* Pydantic: Use for all data models in _core.contract. Provide description for fields.
* Async: Use async/await for I/O and concurrency.
* Hydra: For YAML-based configuration (conf).
* Type Hints: Mandatory for all signatures and variables.
* Docstrings: Google-style for all public APIs.
* Logging: Use centralized logger for key events and errors.

### Testing Structure
- Unit tests in `tests/` with parallel execution (pytest-xdist)
- Integration tests marked separately
- Test fixtures in `conftest.py`
- 600s timeout, async support via pytest-anyio


### Core Principles
- **HASS-Centric Design**: Prioritize usability, clarity, extensibility, and accessibility for HASS researchers
- **Reproducibility & Traceability**: Ensure experiments and data processing are reproducible and traceable
- **Modularity**: Maintain modular architecture with core components: Agent, Orchestrator, Contract
- **Asynchronous Operations**: Use async/await for I/O-bound tasks and LLM calls

### Core Abstractions
- **Agent**: Processing logic (`_process` yields `AgentTrace`)
- **Orchestrator**: Flow management (`run` method)
- **Contract**: Pydantic models for all data exchange (`AgentInput`, `AgentTrace`)
- **UI**: Modular UI components for console, web, slack, and batch interfaces

### Key Technologies
- **Pydantic**: All data models in `_core.contract` with field descriptions
- **Async**: Use async/await for I/O and concurrency
- **Hydra**: YAML-based configuration in `conf/`
- **Type Hints**: Mandatory for all signatures and variables
- **Docstrings**: Google-style for all public APIs
- **Logging**: Use centralized logger for key events and errors

### Code Style
- Line length: 150 characters (Black + Ruff)
- Use Google-style docstrings
- Prefer relative imports
- Target Python 3.11+

### Stability Guidelines
- **Prioritize Stability**: Core contracts and base interfaces should be stable
- **Extend First**: Prefer creating new subclasses over modifying core components
- **Justify Core Changes**: Significant refactoring requires strong justification
- **Pay down tech debt**: Project is in early stages - suggest removing/refactoring code that creates unnecessary complexity

## Instructions for Bots
- Simple, elegant code, always
- Use libraries, don't reinvent the wheel
- Follow existing patterns, unless they're problematic
- Don't add defensive coding; ask for input when stuck with unexpected behavior
- Document for posterity, not just this task
- Unit tests are good. Look at `conftest.py` for major fixtures and follow existing patterns
- Fix only critical errors or problems you introduced - don't look for additional work


# Buttermilk Project Development Guidelines
Primary Goal: Build reproducible, traceable, and HASS-researcher-friendly tools.

* HASS-Centric Design: Prioritize usability, clarity, extensibility, and accessibility for Humanities, Arts, and Social Sciences (HASS) researchers.
* Reproducibility & Traceability: Ensure that experiments, data processing, and results are reproducible and traceable. Design for robust logging and versioning.
* Modularity: Maintain a modular architecture. Core components are Agent, Orchestrator, and Contract.
* Asynchronous Operations: Embrace async/await for I/O-bound tasks, LLM calls, and concurrent operations to ensure responsiveness.
* Pay down tech debt: The project is still in "very early stages". There is left-over code that does not align with these principles or the direction of the project. You should seek confirmation but not hesitate to suggest removing or refactoring code that creates unecessary complexity and maintenance overhead.

## Instructions for bots
```md
- Simple, elegant code, always.
- Use libraries, don't reinvent the wheel.
- Follow existing patterns, unless they're stupid. If they're stupid, talk to the User and offer suggestions before proceeding.
- Don't add defensive coding; ask the User for input when you get stuck with unexpected behavior.
- Document for posterity, not for just this task.
- Unit tests are good. They're in `tests/`. Look at `conftest.py` for major fixtures and follow existing patterns.

Project info:
- all configuration uses Hydra, with files in `conf/`. Main config is `testing.yaml`.
- Main entry point is `cli.py`, using RunRequest() for the job and FlowRunner() for the executor. 
- Python command is `uv run python`. Dependencies managed by `uv`.
- Test command is `uv run python -m pytest`
``` 

## Key project info

- all configuration uses Hydra, with files in `conf/`. Main config is `testing.yaml`.
- Main entry point is `cli.py`, using RunRequest() for the job and FlowRunner() for the executor. 
- Python command is `uv run python`. Dependencies are managed by `uv`.
- Test command is `uv run python -m pytest`

## Coding Conventions & Best Practices:

Core Abstractions:
* Agent: Processing logic (_process yields AgentTrace).
* Orchestrator: Flow management (run method).
* Contract: Pydantic models for all data exchange (e.g., AgentInput, AgentTrace).
* UI: modular UI components to seamless swap between console, web, slack, and batch interfaces.

Separation of Concerns:
* buttermilk._core: Contains fundamental, runtime-agnostic components.

Key Technologies & Practices:
* Pydantic: Use for all data models in _core.contract. Provide description for fields.
* Async: Use async/await for I/O and concurrency.
* Hydra: For YAML-based configuration (conf).
* Type Hints: Mandatory for all signatures and variables.
* Docstrings: Google-style for all public APIs.
* Logging: Use centralized logger for key events and errors.

Structural Changes vs. Stability:
* Prioritize Stability: Core contracts (AgentInput, AgentTrace) and base interfaces (Agent._process, Orchestrator.run) should be stable.
* Extend First: Prefer creating new Agent/Orchestrator subclasses over modifying core components.
* Justify Core Changes: Significant refactoring of core elements requires strong justification based on project goals (HASS usability, reproducibility, extensbility).
* You might find a lot of linting errors. If you've been asked to complete a task, you should only fix the critical errors or problems you introduced. Don't go aruond looking for other stuff to do, although you should make a note of problems you noticed but didn't fix in your final report.