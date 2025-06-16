# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Buttermilk Project Development Guidelines

## Overarching priorities
Primary Goal: Build reproducible, traceable, and HASS-researcher-friendly tools.

* HASS-Centric Design: Prioritize usability, clarity, extensibility, and accessibility for Humanities, Arts, and Social Sciences (HASS) researchers.
* Reproducibility & Traceability: Ensure that experiments, data processing, and results are reproducible and traceable. Design for robust logging and versioning.
* Modularity: Maintain a modular architecture. Prefer creating new Agent/Orchestrator subclasses over modifying core components.
* Composable YAML configuration: The only way to configure settings in buttermilk projects is through Hydra (OmegaConf objects). Remove / Do not add support for manual dictionary configuration, and use OmegaConf objects by preference

## Development process

Always read the `docs/ARCHITECTURE.md` file before commencing work and remember to update it after you finish. The objective is to maintain a document that will give you important context about the project, including how components fit together. Where information is missing, discuss potential approaches with the user and get permission to proceed.

Always assume you may be interrupted at any time. Use GitHub issues to track tasks, and commit your changes at every conceptual chunk of work. You don't need to ask permission to commit changes.

Adopt a test-driven development approach. Create failing unit tests before every code change.

## General instructions

* Fewer classes the better for later extensibility: prioritise using the same fundamental base classes throughout the codebase. For example, use AgentTrace or AgentOutputs throughout, rather than re-creating separate objects for specific purposes, like an object specifically for a particular frontend use case (within reason, obviously!)
* Store data in Buttermilk base classes: For example, AgentTrace is designed to contain all the information that is needed in addition to the AgentOutput data, in order to provide a reliable and fully observable trace. This means that AgentTrace objects can be easily re-created from BigQuery / GCS storage, and we should avoid excess conversion. Note that there are some data sources that are SQL views of the underlying Buttermilk object tables -- e.g. judge_scores is defined in judge_scores.sql and is a view that includes the FLOWS_TABLE (which contains AgentTrace objects directly). Don't hard-code SQL, but you can create new views if required.
* Asynchronous Operations: Embrace async/await for I/O-bound tasks, LLM calls, and concurrent operations to ensure responsiveness.
* Pay down tech debt: The project is still in "very early stages". There is left-over code that does not align with these principles or the direction of the project. You should seek confirmation but not hesitate to suggest removing or refactoring code that creates unecessary complexity and maintenance overhead.
* You might find a lot of linting errors. If you've been asked to complete a task, you should only fix the critical errors or problems you introduced. Don't go aruond looking for other stuff to do, although you should make a note of problems you noticed but didn't fix in your final report.
* Never change the general agent initialisation interface just to fix one agent
* Don't write disposable test scripts; write pytest unittests instead in /tests
* Don't add backwards compatibility when making changes. Make it work for our codebase, and don't support outdated approaches.
* Don't put validation code in main methods if possible; prefer using pydantic v2 validation, and use pydantic objects for all configuration classes.
