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

### [development.md](development.md) - Development Workflow
- Systematic development approach (STOP → ANALYZE → PLAN → TEST → IMPLEMENT)
- GitHub workflow and issue tracking
- Commit standards and documentation
- Common anti-patterns to avoid

### [debugging.md](debugging.md) - Debugging Tools & Strategies
- Available debugging tools and commands
- Expected flow behavior
- Success criteria and troubleshooting
- Log analysis techniques

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

## 🔗 External Resources
- [Main Project README](/README.md)
- [Complete Documentation](/docs/README.md)
- [Architecture Guide](/docs/developer-guide/architecture.md)
- [Contributing Guide](/docs/developer-guide/contributing.md)

## 📋 Development Checklist
Before working on any task:
- [ ] Read all files in this directory
- [ ] Check GitHub issues for related work
- [ ] Understand the data flow and architecture
- [ ] Plan your approach systematically
- [ ] Write tests first
- [ ] Document your changes

Remember: When in doubt, analyze systematically rather than guess. Unknown unknowns are dangerous - surface them early through careful analysis.