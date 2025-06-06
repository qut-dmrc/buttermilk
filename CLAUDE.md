# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Buttermilk Project Development Guidelines
Primary Goal: Build reproducible, traceable, and HASS-researcher-friendly tools.

* HASS-Centric Design: Prioritize usability, clarity, extensibility, and accessibility for Humanities, Arts, and Social Sciences (HASS) researchers.
* Reproducibility & Traceability: Ensure that experiments, data processing, and results are reproducible and traceable. Design for robust logging and versioning.
* Modularity: Maintain a modular architecture. Core components are Agent, Orchestrator, and Contract. Prefer creating new Agent/Orchestrator subclasses over modifying core components.
* Fewer classes the better for later extensibility: prioritise using the same fundamental base classes throughout the codebase. For example, use AgentTrace or AgentOutputs throughout, rather than re-creating separate objects for specific purposes, like an object specifically for a particular frontend use case (within reason, obviously!)
* Store data in Buttermilk base classes: For example, AgentTrace is designed to contain all the information that is needed in addition to the AgentOutput data, in order to provide a reliable and fully observable trace. This means that AgentTrace objects can be easily re-created from BigQuery / GCS storage, and we should avoid excess conversion. Note that there are some data sources that are SQL views of the underlying Buttermilk object tables -- e.g. judge_scores is defined in judge_scores.sql and is a view that includes the FLOWS_TABLE (which contains AgentTrace objects directly). Don't hard-code SQL, but you can create new views if required.
* Asynchronous Operations: Embrace async/await for I/O-bound tasks, LLM calls, and concurrent operations to ensure responsiveness.
* Pay down tech debt: The project is still in "very early stages". There is left-over code that does not align with these principles or the direction of the project. You should seek confirmation but not hesitate to suggest removing or refactoring code that creates unecessary complexity and maintenance overhead.
* You might find a lot of linting errors. If you've been asked to complete a task, you should only fix the critical errors or problems you introduced. Don't go aruond looking for other stuff to do, although you should make a note of problems you noticed but didn't fix in your final report.
* Never change the general agent initialisation interface just to fix one agent
* Don't write disposable test scripts; write pytest unittests instead in /tests


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

## Configuration Hints

* To see the actual hydra config in use, use `-c job`, like: `uv run python -m buttermilk.runner.cli -c job +flow=tox +run=batch "+flows=[tox]" "run.mode=batch_run"`