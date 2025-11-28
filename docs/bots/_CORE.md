# Buttermilk Agent Instructions

## Mission

Buttermilk aims to make it easy for HASS scholars to use AI tools in a way that is understandable, traceable, and reproducible.

## Critical Failure Modes

**Six patterns to avoid**:

1. **Security**: NEVER commit real API keys, tokens, passwords, or secrets
2. **Rush-to-Code**: Explore existing code before implementing (see exploration-before-implementation.md)
3. **Standalone Validation**: No `python -c` testing or standalone scripts - use pytest in `tests/` only
4. **Shared Infrastructure Changes**: Don't break shared components (see impact-analysis.md)
5. **Defensive Coding**: Fix root causes, don't add workarounds
6. **Repository Pollution**: Use GitHub issues, not tracking markdown files

## Pre-Action Checklist

**Before creating files or running tests, verify**:

1. **Security**: Does this contain real credentials? If YES: STOP and use fake values
2. **Location**: Am I creating this in the correct directory? (tests/ for tests, docs/bots/ for general docs)
3. **Purpose**: Is this following proper conventions? (pytest for tests, GitHub issues for tracking)
4. **Alternatives**: Can I use existing files/tests instead?
5. **Shared infrastructure**: Am I modifying shared components? If YES: See impact-analysis.md
6. **Standalone validation**: Am I creating test files outside tests/ or using `python -c`? If YES: Use approved methods instead

## Testing Requirements

### Use Proper Testing Tools

**Approved methods**:
- Run existing tests: `uv run pytest tests/[relevant_module]`
- Extend existing tests in `tests/` directory
- Use debugging tools from `debugging.md`
- Use the `tester` agent via `Task: tester - [describe need]`

**Forbidden**:
- Creating standalone validation scripts anywhere outside `tests/`
- Using `python -c`, `uv run python -c`, or similar inline execution
- Creating "quick test" files or "verify" scripts
- Demo files, sample scripts, or examples outside proper test structure

### Testing Philosophy

**Mock only at system boundaries**:
- Network calls, filesystem, time, environment variables
- NEVER mock `buttermilk.*` code or business logic

See `docs/TESTING.md` for complete guide.

## Observability Requirements

**Research integrity depends on complete observability**. Logging, tracing, and data saving MUST work correctly.

**Never**:
- Add defensive code around tracing failures (`if weave_client is not None:`)
- Make observability optional when it should be required
- Silently continue when logging/saving fails
- Work around broken infrastructure instead of fixing it

**Always**:
- Fix the root cause when observability fails
- Fail fast if infrastructure cannot be fixed
- Ensure environment variables and configuration are correct

## Shared Infrastructure

**High-risk files requiring impact analysis**:
- `tests/conftest.py` - Affects ALL tests
- `buttermilk/_core/*` - Core infrastructure
- Any `__init__.py` - Module initialization
- Configuration files in `conf/`

**Before modifying shared files**:
1. Search for all imports and usage
2. Identify all dependents
3. Consider targeted solutions instead of global changes
4. Test with full suite to verify no regressions

See `docs/bots/impact-analysis.md` for complete protocol.

## Development Workflow

**Before making code changes**:

1. **Explore**: Search for existing solutions, check framework capabilities
2. **Analyze**: Map system architecture, identify root causes
3. **Plan**: Document approach with clear phases (use GitHub issues)
4. **Test**: Write tests in `tests/` directory using pytest
5. **Implement**: Make minimal changes that solve root cause
6. **Document**: Add clear docstrings and comments
7. **Validate**: Use approved validation methods (existing tests or debugging tools)
8. **Commit**: Commit logical chunks, update GitHub issues
9. **Reflect**: Review and update docs/bots if needed

See `docs/bots/DEVELOPMENT.md` for details.

## Technical Stack

**Configuration**: Hydra (composable YAML configs in `conf/`)
**LLM Backend**: Vertex AI (Gemini models)
**Vector Storage**: ChromaDB for RAG
**API**: FastAPI with WebSocket for multi-agent flows
**Testing**: pytest with real infrastructure via `real_bm` fixture

## Critical Rules

- **Verify success**: Always check command results (exit codes AND output text)
- **Use specified tools**: If asked to use Tool X, use Tool X - no substitutions
- **Debugging first**: Check `docs/bots/debugging.md` before reading source code
- **Flow completion**: Task complete means end-to-end flow execution with visible output
- **No implicit defaults**: All configuration must be explicit - fail fast if missing
- **Data contracts**: Schema changes require updating ALL components (see docs/bots/data-architecture.md)
- **No single-use scripts**: Always use proper tests following pytest conventions

## Debugging Workflow

**Before debugging**:
1. Read `docs/bots/debugging.md` for approved tools
2. Use golden path tools (e.g., `ws_debug_cli.py`, Playwright MCP)
3. Test tool connectivity before complex operations
4. Follow troubleshooting guidance if tools fail

**Output limits**:
- Keep outputs concise (max 15 lines for debugging excerpts)
- Extract key findings instead of dumping full responses
- Summarize patterns rather than listing all entries

## Red Flag Phrases

**If you catch yourself saying these, STOP and reconsider**:

**Standalone validation violations**:
- "Let me create a test to..." / "I'll write a quick test..."
- "Let me verify this works..." / "I'll test the serialization..."
- "I'll use python -c to verify..." / "Let me run a quick python command..."
- "I'll create an example..." / "Let me make a demo..."

**Defensive coding violations**:
- "Let me add a check for None and skip if it fails..."
- "I'll wrap this in try/except so failures don't break..."
- "I'll add a fallback when X is None..."

**Shared infrastructure violations**:
- "I'll just remove this from conftest.py to make my tests pass..."
- "I'll modify this base class to handle my use case..."

**Repository pollution violations**:
- "I'll create a README to document what was fixed..."
- "Let me add a status file to track this implementation..."

**Correct responses**:
- Standalone validation → Use `Task: tester` or approved testing methods
- Defensive coding → Fix the root cause instead
- Shared infrastructure → Use targeted solutions, see impact-analysis.md
- Documentation → Update GitHub issues, not tracking files

## Reflective Workflow

We continuously refine our workflow. Maintain `docs/bots/` with essential information for developers.

- Use INDEX.md to index and link documentation
- Update documents when key changes occur (e.g., changes to APIs, workflows, dependencies, or major features). Minor issues (e.g., typo fixes, formatting, small refactors) do not require documentation updates.
- Include important general information, remove task-specific details
- Be concise to save tokens
- If conflicting information found, ask for clarification and update docs

## File References

- Exploration before implementation: `docs/bots/exploration-before-implementation.md`
- Impact analysis for shared components: `docs/bots/impact-analysis.md`
- Debugging guide and tools: `docs/bots/debugging.md`
- Development workflow: `docs/bots/DEVELOPMENT.md`
- Testing guide: `docs/TESTING.md`
- Data architecture: `docs/bots/data-architecture.md`
- Configuration guide: `docs/configuration.md`
