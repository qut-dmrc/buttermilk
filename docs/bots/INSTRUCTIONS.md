<!-- This file should be read EVERY time. Keep it CONCISE and LIMITED to strictly required information. -->

Buttermilk aims to make it easy for HASS scholars to use AI tools in a way that is understandable, traceable, and reproducible.

**🚨 CRITICAL: READ [exploration-before-implementation.md](exploration-before-implementation.md) IMMEDIATELY IF YOU'RE ABOUT TO IMPLEMENT ANYTHING 🚨**

# 🚨 CRITICAL FAILURE MODES PREVENTION 🚨

**YOU HAVE TWO DOCUMENTED PATTERNS THAT MUST STOP:**
1. **RUSH-TO-CODE**: Jumping to implementation without exploration
2. **STANDALONE TEST SCRIPTS**: Creating single-use test files instead of proper pytest tests

## 🚨 MANDATORY TESTING CHECKPOINT: STOP BEFORE ANY TESTING OR VALIDATION 🚨

**BEFORE you test, validate, or verify ANY code behavior, you MUST:**

### ❌ NEVER CREATE THESE FILES:
- `test_*.py` files outside the `tests/` directory
- "Quick test scripts" or "validation files" in the project root
- Any file with names like: `test_something.py`, `verify_*.py`, `check_*.py`, `validate_*.py`

### 🚨 RED FLAG PHRASES - STOP IMMEDIATELY WHEN YOU USE THESE:
- "Let me create a test to..."
- "I'll write a quick test..."
- "Let me verify this works..."
- "I'll test the serialization..."
- "Let me check if this runs..."
- "I'll validate my implementation..."
- "Let me see if this works..."
- "I'll create a script to test..."

### ✅ CORRECT APPROACH - ALWAYS USE PYTEST:
- Create tests in `tests/` directory following existing structure
- Use proper pytest conventions with `def test_*()` functions
- Import required modules properly
- Use pytest fixtures and assertions
- Ensure tests integrate with CI/CD pipeline

### 📋 CONCRETE EXAMPLE:

**❌ WRONG (Standalone Script)**:
```python
# test_something.py in project root
import json
from my_module import MyClass
obj = MyClass()
result = obj.serialize()
print("Success!")
assert result["field"] == "expected"
```

**✅ CORRECT (Proper Pytest)**:
```python
# tests/unit/test_something.py
import pytest
from my_module import MyClass

def test_myclass_serialization():
    """Test that MyClass serializes correctly."""
    obj = MyClass()
    result = obj.serialize()
    
    assert result["field"] == "expected"
    assert isinstance(result, dict)
```

**ENFORCEMENT**: If you catch yourself about to create a test file, STOP and ask: "Am I creating this in the tests/ directory using proper pytest conventions?" If no, RESTART your approach.

## MANDATORY EXPLORATION CHECKPOINT
**BEFORE writing ANY code, you MUST:**
1. **EXPLORATION PHASE** (MINIMUM 3 searches):
   - Search for existing similar functionality in the codebase
   - Look for base classes, utilities, or patterns that solve the same problem
   - Check autogen documentation/examples for built-in solutions
   - Document what you found (even if "nothing relevant")

2. **VALIDATION CHECKPOINT**:
   - Can you reuse existing code instead of writing new?
   - Are you building something that already exists?
   - Did you check the framework's built-in capabilities?
   - **IF ANY ANSWER IS UNCLEAR, STOP AND EXPLORE MORE**

3. **IMPLEMENTATION JUSTIFICATION**:
   - Write 2-3 sentences explaining WHY your approach is necessary
   - Explain WHY simpler alternatives won't work
   - **IF YOU CAN'T JUSTIFY, YOU'RE PROBABLY OVERENGINEERING**

**RED FLAG PHRASES THAT MEAN YOU'RE RUSHING TO CODE:**
- "Let me implement..."
- "I'll create a custom..."
- "I'll build a new..."
- "Let me write..."

**RED FLAG PHRASES THAT MEAN YOU'RE ABOUT TO CREATE STANDALONE TESTS:**
- "Let me create a test..."
- "I'll write a test script..."
- "Let me verify this works..."
- "I'll test the implementation..."

**WHEN YOU CATCH YOURSELF USING THESE PHRASES: STOP. FOLLOW THE PROPER WORKFLOW.**

# CRITICAL RULES for 🤖 LLM Agents: 

* ALWAYS follow the workflow. No exceptions.
* **EXPLORATION BEFORE IMPLEMENTATION**: NEVER code without first searching for existing solutions
* ASK FOR HELP WHEN YOU NEED IT. 
* DO NOT try to fix errors you do not fully understand.
* STOP if you get stuck. No workarounds.
* EVERYTHING is VALIDATED on rigorous academic research. DO NOT MAKE substantive decisions on your own. NO ASSUMPTIONS. No single-use scripts, no dummy examples; use ONLY our validated examples.
* Prioritise reproducibility & traceability.
* Flexible, modular design aimed at future Humanities, Arts, and Social Sciences researchers.

Specific rules:
- VERIFY SUCCESS, never assume: ALWAYS check the results of commands (exit codes AND result text).
- When given a specific task with a specific method, you MUST use that exact method. Any deviation is a failure. Success is ONLY achieving the exact outcome requested using the exact method specified. NO SUBSTITUTIONS.
- If asked to use Tool X, use Tool X. Don't use Tool Y instead. Don't read code instead. Don't check logs instead. Use Tool X.
- Success = user's definition, not yours. If asked to SHOW something, you must visually demonstrate it, not just prove it exists.
- When you can't complete a task as specified, STOP immediately and say so. Don't waste time on workarounds.
- **DEBUGGING RULE**: For ANY debugging task, you MUST first check `docs/bots/debugging.md`. You MUST use the "golden path" tools documented there (e.g. `ws_debug_cli.py`, `buttermilk_logs.py`, Playwright MCP) BEFORE reading source code. Going straight to source code is a workflow violation.
- **OUTPUT RULE**: Keep outputs concise. When using debugging tools, focus on relevant excerpts. Avoid dumping entire JSON responses or log files. If output exceeds 50 lines, summarize key findings instead.
- **NO IMPLICIT DEFAULTS**: NEVER use implicit defaults or fallback behaviors. All configuration must be explicit. If something is missing, FAIL FAST with clear error messages. Do not infer, assume, or provide defaults.
- **DATA CONTRACTS**: Schema changes require updating ALL components atomically. See `docs/bots/data-architecture.md` for data contract principles. No defensive programming - trust schemas and let errors propagate.
- **NO SINGLE-USE SCRIPTS**: NEVER create standalone test scripts or dummy examples. Always write proper tests in the existing test suite using pytest conventions. Test files must be reusable, follow project structure, and integrate with the CI/CD pipeline. 

## CRITICAL FIRST STEPS
1. **For Debugging Tasks**: ALWAYS read `docs/bots/debugging.md` FIRST
2. **For Development Tasks**: Follow the 9-step workflow below
3. **For Research Tasks**: Use documented tools, not source code exploration

## TESTING WORKFLOW REMINDER
**When implementing tests (covered in detail above):**
- Use `uv run pytest tests/` to run the test suite
- Follow existing test patterns in the codebase
- Write tests that integrate with CI/CD pipeline

## WORKFLOW: Before Making Any Code Changes
1. **STOP**: Understand the full problem scope before proposing solutions. Read relevant documentation to understand the project goals and architecture. Check github issues for relevant past work and discussion; create a new issue if you cannot find an existing one.

2. **🔍 MANDATORY EXPLORATION PHASE** (CANNOT BE SKIPPED):
   - **Search existing codebase**: Use Grep/Glob to find similar functionality, base classes, utilities
   - **Check framework capabilities**: Search autogen docs, examine base classes in the project
   - **Document findings**: Write what you found, even if "no existing solution found"
   - **CHECKPOINT**: Can you reuse/extend existing code instead of writing from scratch?
   - **JUSTIFICATION**: Explain in 2-3 sentences WHY you need to build something new
   - **IF YOU CANNOT JUSTIFY BUILDING NEW CODE, STOP AND RECONSIDER**

3. **ANALYZE**: Map the system architecture and identify root causes

4. **PLAN**: Use github issues to track problems and document your plan with clear phases and validation criteria

5. **TEST**: Write unit tests in `tests/` directory using pytest conventions that capture expected behavior. **CRITICAL**: NO standalone test scripts anywhere else - violating this rule means starting over.

6. **IMPLEMENT**: Make minimal changes that solve the root cause

7. **DOCUMENT**: ALWAYS document your code with clear docstrings and comments

8. **VALIDATE**: Use the project's end-to-end debugging tools to ensure no regressions and all success criteria are met

9. **COMMIT** and **UPDATE GITHUB ISSUES**: Commit your changes in logical chunks and document each step in the appropriate github issue. If you're working independently, don't forget to file a pull request with your new changes!

10. **REFLECT**: Review your performance and update the `docs/bots` instructions if necessary to avoid mistakes.


## Key technical hints
- Repository: @qut-dmrc/buttermilk
- Owner: @nicsuzor
- Architecture: YAML configuration (/conf); FastAPI backend (buttermilk/api); Autogen-based LLM groupchat flows; web (/buttermilk/frontend/chat) and cli (/buttermilk/frontend/cli) frontends.
- Run python with `uv run ...`
- **Composable YAML Configuration**: Use Hydra (OmegaConf objects) exclusively for configuration
- Use heredocs when writing in shell to avoid escaping issues

## REFLECTIVE and EXPERIMENTAL workflow

We are CONTINUOSLY refining our workflow. Agents ONLY remember the information we provide. Maintain the `docs/bots/` folder with essential information for robot developers.
- Use [INDEX.md](docs/bots/INDEX.md) to index and link to documentation and external tools.
- Update documents every time there is a key change (but not minor issues)
- Include all GENERAL important information developers need to understand, but REMOVE minor details or information that is specific to a particular task or scenario.
- Be CONCISE to save tokens.
- If you find conflicting information, ask the user for clarification, and then update the documents.



