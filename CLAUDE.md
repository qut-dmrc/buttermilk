# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Process Guidelines

### Before Making Any Code Changes
1. **STOP**: Understand the full problem scope before proposing solutions
2. **ANALYZE**: Map the system architecture and identify root causes  
3. **PLAN**: Create documentation (issues/plans) with clear phases and validation criteria
4. **TEST**: Write failing tests that capture expected behavior
5. **IMPLEMENT**: Make minimal changes that solve the root cause
6. **VALIDATE**: Ensure no regressions and all success criteria are met

### Red Flags That Indicate You're Moving Too Fast
- Proposing config changes without understanding data flow
- Making multiple small fixes instead of one root cause fix
- Suggesting "try this" without a systematic plan
- Modifying code without first writing tests that demonstrate the problem
- **CRITICAL**: Changing Pydantic model validation settings (like `extra="forbid"` to `extra="allow"`) when encountering validation errors
- **CRITICAL**: Making "quick fixes" to suppress errors instead of understanding why they occur

### Never Make These Superficial Fixes
- Changing `extra="forbid"` to `extra="allow"` in Pydantic models when encountering "extra_forbidden" errors
- Adding try/except blocks to suppress validation errors without understanding the root cause
- Modifying type annotations to be more permissive (like `Any`) to bypass type checking
- Adding `# type: ignore` comments without understanding why the type checker is failing

### When You Encounter Validation Errors
1. **Trace the Data Flow**: Understand how data flows from YAML → Hydra → Pydantic models
2. **Identify the Conversion Point**: Find where the conversion should happen (field validators, factory methods, etc.)
3. **Fix at the Right Level**: Apply fixes where the data transformation should logically occur
4. **Test the Fix**: Ensure the fix works for the actual use case, not just the immediate error

### When Debugging Complex Issues
- Create GitHub issues with comprehensive analysis
- Write failing tests before implementing fixes
- Validate architectural assumptions with code inspection
- Make one minimal change that addresses the root cause

### Development Discipline Principles
- **Analysis Before Action**: Understand systems before changing them
- **Documentation-Driven Development**: Write issues/plans before coding
- **Test-First Debugging**: Failing tests → fix → passing tests
- **Minimal Viable Fixes**: Smallest change that solves root cause
- **Embrace "I Don't Know Yet"**: Systematic analysis beats guessing
- **Always Commit and Document**: Commit changes immediately and update GitHub issues with progress/solutions

## GitHub Repository Configuration

The default GitHub repository for this project is `qut-dmrc/buttermilk`. When using the `gh` CLI tool, commands will default to this repository. You can use commands like:
- `gh issue list`
- `gh issue view <number>`
- `gh pr list`
- `gh pr create`

Without needing to specify `--repo qut-dmrc/buttermilk` each time.

## Buttermilk Project Development Guidelines

### Overarching priorities
Primary Goal: Build reproducible, traceable, and HASS-researcher-friendly tools.

* HASS-Centric Design: Prioritize usability, clarity, extensibility, and accessibility for Humanities, Arts, and Social Sciences (HASS) researchers.
* Reproducibility & Traceability: Ensure that experiments, data processing, and results are reproducible and traceable. Design for robust logging and versioning.
* Modularity: Maintain a modular architecture. Prefer creating new Agent/Orchestrator subclasses over modifying core components.
* Composable YAML configuration: The only way to configure settings in buttermilk projects is through Hydra (OmegaConf objects). Remove / Do not add support for manual dictionary configuration, and use OmegaConf objects by preference

## Development process

Always read the `docs/ARCHITECTURE.md` file before commencing work and remember to update it after you finish. The objective is to maintain a document that will give you important context about the project, including how components fit together. Where information is missing, discuss potential approaches with the user and get permission to proceed.

Always assume you may be interrupted at any time. Use GitHub issues to track tasks, and commit your changes at every conceptual chunk of work. You don't need to ask permission to commit changes.

### 🚨 CRITICAL: Git Commits and GitHub Issues 🚨
**BEFORE COMMITTING ANY CHANGES:**
1. Check for related GitHub issues: `gh issue list --search "relevant keywords"`
2. If no issue exists, consider creating one: `gh issue create`
3. Commit your changes with a descriptive message
4. **IMMEDIATELY AFTER COMMITTING**: Update the GitHub issue with:
   - A comment describing what was changed
   - The commit hash
   - Any remaining work or follow-up tasks
   - Close the issue if the work is complete

**Example workflow:**
```bash
# Before starting work
gh issue list --search "initialize announcement"
# Work on the changes...
# After making changes
git add -A && git commit -m "refactor: Consolidate initialization methods"
gh issue comment <issue-number> --body "Completed refactoring of initialization methods in commit abc123"
```

This ensures work is preserved and documented for future reference.

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
* NEVER provide multiple formats for the same data (e.g., both camelCase and snake_case versions). Pick one format and use it consistently. This prevents ambiguity and confusion.
* Don't put validation code in main methods if possible; prefer using pydantic v2 validation, and use pydantic objects for all configuration classes.

## Development Memories

* Run python and pytest with `uv run ...`
* The BM() class, usually instatantiated as `bm`, is a critical component of all buttermilk projects. It handles logging and authentication, as well as providing many convenience methods that embed strong MLOps default dest practices. It is accessed with the get_bm() method and should be instantiated and singleton saved with set_bm() as the first substantive step of any run.
* Use `uv run python -m butttermilk.runner.cli ... -c job` to view the config as compiled
* don't create defensive code. Use validators on config and inputs and from then on, just assume athe best scenario and allow errors to percolate out. 
* don't add legacy code, we do not maintain backwards compatibility.