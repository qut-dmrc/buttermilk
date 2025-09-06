# Buttermilk Bot Developer Knowledge Bank

Essential information for LLM chatbot developers working with the Buttermilk codebase. This knowledge bank provides the core concepts, patterns, and guidelines necessary for effective contribution.

## 📚 Core Documentation

### [goals.md](goals.md) - Project Philosophy & Objectives
- HASS-centric design principles
- Core objectives and values
- Target users and use cases

### [techstack.md](techstack.md) - Technology Stack & Architecture
- Key technologies (Python 3.10+, Hydra, Pydantic v2, AsyncIO)
- Architecture patterns and design decisions
- Agent/Orchestrator architecture
- Tool definition system

### [map.md](map.md) - Project Structure Map
- Directory structure and organization
- Key files and their purposes
- Where to find specific components

### [config.md](config.md) - Configuration System
- Hydra/OmegaConf configuration management
- YAML file structure and composition
- Common configuration patterns
- Interpolation and overrides

### [exploration-before-implementation.md](exploration-before-implementation.md) - CRITICAL: Rush-to-Code Prevention
- **MANDATORY READ**: Documented failure pattern and prevention protocol
- Required exploration phases before any implementation
- Red flag detection and recovery procedures
- Concrete search techniques and justification requirements

### [impact-analysis.md](impact-analysis.md) - CRITICAL: Shared Infrastructure Protection
- **MANDATORY READ**: Prevention of tunnel vision on shared infrastructure
- Impact analysis protocol for shared files and components
- High-risk file identification and protection strategies
- Targeted solutions vs. shared infrastructure modifications

### [development.md](development.md) - Development Workflow
- Systematic development approach (STOP → ANALYZE → PLAN → TEST → IMPLEMENT)
- GitHub workflow and issue tracking
- Commit standards and documentation
- Common anti-patterns to avoid

### [debugging.md](debugging.md) - The Golden Path to Debugging
- The single, authoritative guide to debugging and validation.
- Provides a simple, clear workflow for the most common tasks.

### [logs.md](logs.md) - Log Analysis Guide
- Finding and accessing debug logs
- Common log patterns and pre-built searches
- Building custom grep commands for log analysis
- Time-based filtering and performance tips
- Log file management commands

### [data-architecture.md](data-architecture.md) - Data Architecture & Schema Contracts
- Schema-data contract principles
- No defensive programming philosophy
- Data model structure and evolution
- Testing data integrity
- Common data handling pitfalls

### [standalone-tracing.md](standalone-tracing.md) - Standalone Tracing for Batch Processes
- Tracing outside orchestrator context
- StandaloneTraceContext usage
- Integration with DocProcessor and agents
- Examples for scripts and CLI tools

### [tester subagent](.claude/agents/tester.md) - Testing Specialist Subagent
- Comprehensive testing, validation, and verification agent (configured in .claude/agents/)
- Prevents standalone validation code creation
- Converts all testing needs into proper pytest tests
- Handles examples, demos, and test creation workflows

## 🚀 Quick Reference

### Essential Commands
```bash
# Run tests
uv run pytest

# View configuration
uv run python -m buttermilk.runner.cli -c job

# Run API server
uv run python -m buttermilk.runner.cli "+flows=[zot,osb,trans]" +run=api llms=full

# Debug flows
uv run python -m buttermilk.debug.ws_debug_cli test-connection
```

### 🚨 CRITICAL FAILURE MODE PREVENTION
**THREE DOCUMENTED PATTERNS - MANDATORY CHECKPOINTS:**

**RUSH-TO-CODE PATTERN:**
1. **BEFORE ANY IMPLEMENTATION**: Complete exploration phase (minimum 3 searches)
2. **SEARCH EXISTING CODE**: Find base classes, utilities, framework capabilities
3. **JUSTIFY NEW CODE**: Explain why existing solutions won't work
4. **IF YOU CAN'T JUSTIFY**: You're probably overengineering

**STANDALONE VALIDATION PATTERN:**
1. **NEVER** create standalone validation (files OR commands) outside proper pytest workflow
2. **RED FLAGS**: "Let me create a test...", "I'll verify this works...", "I'll use python -c..."
3. **ALWAYS** redirect to tester agent: `Task: tester - [describe need]`
4. **ENFORCEMENT**: Any standalone validation = immediately redirect to tester agent

**SHARED INFRASTRUCTURE TUNNEL VISION PATTERN:**
1. **BEFORE MODIFYING SHARED FILES**: Mandatory impact analysis (see `impact-analysis.md`)
2. **HIGH-RISK FILES**: `conftest.py`, `_core/*`, base classes, `__init__.py`
3. **RED FLAGS**: "I'll remove this from conftest.py...", "This shared file is causing problems..."
4. **ALWAYS** consider targeted solutions instead of modifying shared components

### 🚨 Debugging Quick Reference
**ALWAYS START HERE for debugging tasks:**
1. Read `debugging.md` FIRST (no exceptions)
2. Use WebSocket CLI to reproduce: `uv run python -m buttermilk.debug.ws_debug_cli start <flow>`
3. Check logs with focus: `python scripts/mcp_debug/buttermilk_logs.py search "pattern" 50`
4. ONLY read source code after understanding actual behavior
5. Keep outputs concise - extract relevant data only

### Key Principles
1. **HASS-Centric**: Prioritize humanities researchers' needs
2. **Reproducibility**: Ensure experiments are traceable
3. **Modularity**: Create new subclasses, don't modify core
4. **Configuration-First**: Use Hydra/YAML exclusively
5. **Test-Driven**: Write tests before implementation
6. **Documentation**: Keep docs synchronized with code

### Critical Reminders
- **ALWAYS** check GitHub issues before starting work
- **NEVER** make superficial fixes (e.g., changing `extra="forbid"`)
- **ALWAYS** trace data flow when debugging validation errors
- **NEVER** commit without updating relevant documentation
- **ALWAYS** use the provided debugging framework
- **NEVER** create standalone validation (files OR commands) - Use tester agent ONLY (violating this = restart task)

## 🔗 External Resources
- [Main Project README](/README.md)
- [Complete Documentation](/docs/README.md)
- [Architecture Guide](/docs/developer-guide/architecture.md)
- [Contributing Guide](/docs/developer-guide/contributing.md)

## 📋 Development Checklist
Before working on any task:
- [ ] Read all files in this directory
- [ ] **MANDATORY EXPLORATION**: Search codebase for existing solutions (minimum 3 searches)
- [ ] **MANDATORY EXPLORATION**: Check framework/autogen built-in capabilities
- [ ] **MANDATORY JUSTIFICATION**: Explain why new code is necessary vs. reusing existing
- [ ] Check GitHub issues for related work
- [ ] Understand the data flow and architecture
- [ ] Plan your approach systematically
- [ ] Write tests first
- [ ] Document your changes

Remember: When in doubt, analyze systematically rather than guess. Unknown unknowns are dangerous - surface them early through careful analysis.