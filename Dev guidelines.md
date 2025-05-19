# Buttermilk Project Development Guidelines
Primary Goal: Build reproducible, traceable, and HASS-researcher-friendly tools.

* HASS-Centric Design: Prioritize usability, clarity, extensibility, and accessibility for Humanities, Arts, and Social Sciences (HASS) researchers.
* Reproducibility & Traceability: Ensure that experiments, data processing, and results are reproducible and traceable. Design for robust logging and versioning.
* Modularity: Maintain a modular architecture. Core components are Agent, Orchestrator, and Contract.
* Asynchronous Operations: Embrace async/await for I/O-bound tasks, LLM calls, and concurrent operations to ensure responsiveness.
* Pay down tech debt: The project is still in "very early stages". There is left-over code that does not align with these principles or the direction of the project. You should seek confirmation but not hesitate to suggest removing or refactoring code that creates unecessary complexity and maintenance overhead.

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