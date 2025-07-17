# Buttermilk development guide

This file provides essential guidance to LLM Agents when working with code in this repository.

## Buttermilk Core Principles
- **HASS-Centric Design**: Prioritize usability for Humanities, Arts, and Social Sciences researchers
- **Reproducibility & Traceability**: Ensure experiments and results are reproducible and traceable
- **Modularity**: Prefer creating new Agent/Orchestrator subclasses over modifying core components
- **Composable YAML Configuration**: Use Hydra (OmegaConf objects) exclusively for configuration

## Adaptive Workflow
At the beginning of every task, read the essential documents in the `doc/bots/` folder, starting with `doc/bots/README.md`. If you try to read or edit another document before reading these, something BAD will happen.

## Critical Documentation and Workflow

Maintain the `docs/bots/` folder with essential information for robot developers.
- Use [README.md](docs/bots/README.md) to index and link to documentation and external tools.
- Use markdown formatting
- Update documents every time there is a key change (but not minor issues)
- Include ALL important information developers need to understand, but REMOVE minor details or information that is highly specific to a particular task or scenario.
- Be CONCISE to save tokens.
- If you find conflicting information, ask the user for clarification, and then update the documents.


### Before Making Any Code Changes
1. **STOP**: Understand the full problem scope before proposing solutions. Check github issues for relevant past work and discussion; create a new issue if you cannot find an existing one.
2. **ANALYZE**: Map the system architecture and identify root causes
3. **PLAN**: Use github issues to track problems and document your plan with clear phases and validation criteria
4. **TEST**: Write failing tests that capture expected behavior
5. **IMPLEMENT**: Make minimal changes that solve the root cause
6. **DOCUMENT**: ALWAYS document your code with clear docstrings and comments
7. **VALIDATE**: Ensure no regressions and all success criteria are met
8. **COMMIT** and **UPDATE GITHUB ISSUES**: Commit your changes in logical chunks and document each step in the appropriate github issue.
