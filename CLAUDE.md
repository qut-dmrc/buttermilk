# CLAUDE.md

This file provides essential guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Process Guidelines

See `docs/developer-guide/llm-debugging.md` for tools, expected behavior, and success criteria.

### Before Making Any Code Changes
1. **STOP**: Understand the full problem scope before proposing solutions
2. **ANALYZE**: Map the system architecture and identify root causes  
3. **PLAN**: Create documentation (issues/plans) with clear phases and validation criteria
4. **TEST**: Write failing tests that capture expected behavior
5. **IMPLEMENT**: Make minimal changes that solve the root cause
6. **DOCUMENT**: ALWAYS document your code with clear docstrings and comments
7. **VALIDATE**: Ensure no regressions and all success criteria are met

### Red Flags That Indicate You're Moving Too Fast
- Proposing config changes without understanding data flow
- Making multiple small fixes instead of one root cause fix
- Suggesting "try this" without a systematic plan
- Modifying code without first writing tests that demonstrate the problem
- **CRITICAL**: Changing Pydantic model validation settings (like `extra="forbid"` to `extra="allow"`) when encountering validation errors
- **CRITICAL**: Making "quick fixes" to suppress errors instead of understanding why they occur

### Never Make Superficial Fixes
- Changing `extra="forbid"` to `extra="allow"` in Pydantic models when encountering "extra_forbidden" errors
- Adding try/except blocks to suppress validation errors without understanding the root cause
- Modifying type annotations to be more permissive (like `Any`) to bypass type checking
- Adding `# type: ignore` comments without understanding why the type checker is failing

### When You Encounter Validation Errors
1. **Trace the Data Flow**: Understand how data flows from YAML → Hydra → Pydantic models
2. **Identify the Conversion Point**: Find where the conversion should happen (field validators, factory methods, etc.)
3. **Fix at the Right Level**: Apply fixes where the data transformation should logically occur
4. **Test the Fix**: Ensure the fix works for the actual use case, not just the immediate error

### Development Discipline Principles
- **Analysis Before Action**: Understand systems before changing them
- **Documentation-Driven Development**: Write issues/plans before coding
- **Test-First Debugging**: Failing tests → fix → passing tests
- **Minimal Viable Fixes**: Smallest change that solves root cause
- **Embrace "I Don't Know Yet"**: Systematic analysis beats guessing
- **Always Commit and Document**: Commit changes immediately and update GitHub issues with progress/solutions

### Testing and Debugging Guidelines
- **Do not make miscellaneous test scripts; use the proper debug framework**

## Buttermilk Project Development Guidelines

### Core Principles
- **HASS-Centric Design**: Prioritize usability for Humanities, Arts, and Social Sciences researchers
- **Reproducibility & Traceability**: Ensure experiments and results are reproducible and traceable
- **Modularity**: Prefer creating new Agent/Orchestrator subclasses over modifying core components
- **Composable YAML Configuration**: Use Hydra (OmegaConf objects) exclusively for configuration

### Configuration
The LLM configuration is loaded from a GCP Secret named `models.json`. Authentication with GCP is required for tests to run correctly.

## Where to Find More Information

For detailed development guidance, see:
- `docs/developer-guide/contributing.md` - Development process and standards
- `docs/developer-guide/architecture.md` - System architecture 
- `docs/user-guide/configuration.md` - Configuration management
- `docs/reference/concepts.md` - Core concepts

## GitHub Repository Configuration

The default GitHub repository for this project is `qut-dmrc/buttermilk`. When using the `gh` CLI tool, commands will default to this repository. You can use commands like:
- `gh issue list`
- `gh issue view <number>`
- `gh pr list`
- `gh pr create`

Without needing to specify `--repo qut-dmrc/buttermilk` each time.

## 🚨 CRITICAL: Git Commits and GitHub Issues 🚨
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