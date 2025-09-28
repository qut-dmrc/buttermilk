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

### [config.md](config.md) - Configuration System
- Hydra/OmegaConf configuration management
- YAML file structure and composition
- Common configuration patterns
- Interpolation and overrides

### [debugging.md](debugging.md) - The Golden Path to Debugging
- The single, authoritative guide to debugging and validation
- Provides a simple, clear workflow for the most common tasks
- Validated debugging tools and infrastructure

### [data-architecture.md](data-architecture.md) - Data Architecture & Schema Contracts
- Schema-data contract principles
- No defensive programming philosophy
- Data model structure and evolution
- Testing data integrity

### [eval.md](eval.md) - Evaluation Tools
- Evaluation frameworks and methodologies
- Performance measurement tools
- Quality assessment patterns

### [FLOWS.md](FLOWS.md) - Buttermilk-Specific Flow Patterns
- Agent flow configurations
- Workflow orchestration patterns
- Flow debugging and validation

## 🤖 Specialized Agents

### [TEST_FIXER_AGENT.md](TEST_FIXER_AGENT.md) - Test Fixing Workflow
- Systematic test repair using ruff diagnostics
- Batch fixing procedures for test suites
- Health dashboard and priority management

### [AGENT-DEBUGGER.md](AGENT-DEBUGGER.md) - Debug Pipeline Manager
- Live system debugging specialist
- End-to-end validation protocols
- Infrastructure maintenance procedures

## 🚀 Quick Reference

### Essential Commands
```bash
# Run tests
uv run pytest

# View configuration
uv run python -m buttermilk.runner.cli -c job

# Run API server
uv run python -m buttermilk.runner.cli "+flows=[zot,osb,trans]" run=api llms=full

# Debug flows
uv run python -m buttermilk.debug.ws_debug_cli test-connection
```

### 🚨 CRITICAL FAILURE MODE PREVENTION
**DOCUMENTED PATTERNS - MANDATORY CHECKPOINTS:**

**1. RUSH-TO-CODE**: Complete exploration phase before any implementation
**2. STANDALONE VALIDATION**: Use proper pytest workflow, never create standalone validation
**3. SHARED INFRASTRUCTURE TUNNEL VISION**: Mandatory impact analysis before modifying shared files
**4. DEFENSIVE CODING**: Fix root causes, don't hide observability failures
**5. REPOSITORY POLLUTION**: Use GitHub issues, not repository documentation files

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

### Critical Reminders
- **ALWAYS** check GitHub issues before starting work
- **NEVER** create standalone validation (files OR commands) - Use existing test patterns
- **ALWAYS** trace data flow when debugging validation errors
- **ALWAYS** use the provided debugging framework

## 📋 Development Checklist
Before working on any task:
- [ ] **MANDATORY EXPLORATION**: Search codebase for existing solutions
- [ ] **MANDATORY EXPLORATION**: Check framework built-in capabilities
- [ ] **MANDATORY JUSTIFICATION**: Explain why new code is necessary
- [ ] Check GitHub issues for related work
- [ ] Understand the data flow and architecture
- [ ] Plan your approach systematically
- [ ] Write tests first
- [ ] Document your changes

Remember: When in doubt, analyze systematically rather than guess. Unknown unknowns are dangerous - surface them early through careful analysis.