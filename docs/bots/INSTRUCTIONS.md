<!-- This file should be read EVERY time. Keep it CONCISE and LIMITED to strictly required information. -->

Buttermilk aims to make it easy for HASS scholars to use AI tools in a way that is understandable, traceable, and reproducible.

**🚨 CRITICAL: READ [exploration-before-implementation.md](exploration-before-implementation.md) IMMEDIATELY IF YOU'RE ABOUT TO IMPLEMENT ANYTHING 🚨**

# 🚨 CRITICAL FAILURE MODES PREVENTION 🚨

**YOU HAVE TWO DOCUMENTED PATTERNS THAT MUST STOP:**
1. **RUSH-TO-CODE**: Jumping to implementation without exploration
2. **STANDALONE VALIDATION**: Creating standalone validation code (files OR inline commands) instead of using proper pytest workflows

## 🚨 MANDATORY TESTING CHECKPOINT: STOP BEFORE ANY TESTING OR VALIDATION 🚨

**🛑 UNIVERSAL FILE CREATION CHECKPOINT 🛑**
**BEFORE creating ANY file (.py, .js, .md, etc.), you MUST ask yourself:**
1. **Location check**: Am I creating this in the correct directory? (tests/ for test files)
2. **Purpose check**: Is this following proper conventions? (pytest for tests)
3. **Alternative check**: Can I use existing files/tests instead?
4. **IF ANY ANSWER IS NO OR UNCLEAR: STOP and find the correct approach**

**BEFORE you test, validate, or verify ANY code behavior, you MUST:**

### ❌ NEVER CREATE STANDALONE VALIDATION (FILES OR COMMANDS):

**Forbidden File-Based Validation:**
- `test_*.py` files outside the `tests/` directory
- "Quick test scripts" or "validation files" in the project root
- Any file with names like: `test_something.py`, `verify_*.py`, `check_*.py`, `validate_*.py`
- `examples/*.py`, `demo_*.py`, or any standalone demonstration scripts

**Forbidden Command-Based Validation:**
- **Inline Python validation**: `uv run python -c "..."`, `python -c "..."`, or similar execution patterns
- **Bash-embedded test scripts**: Multi-line Python code in heredocs or command strings
- **"Quick verification" commands**: Any form of standalone Python execution for testing purposes

### 🚨 RED FLAG PHRASES - STOP IMMEDIATELY WHEN YOU USE THESE:

**🛑 CRITICAL: These phrases indicate you're about to violate workflow. STOP and use approved methods instead:**

**Standalone Validation Violations (ANY form of standalone testing/validation):**
- "Let me create a test to..." / "I'll write a quick test..." / "Let me create a simple test..."
- "I'll create a test script..." / "I'll make a test file..."
- "Let me verify this works..." / "I'll test the serialization..." / "Let me check if this runs..."
- "I'll validate my implementation..." / "Let me see if this works..." / "I need to test this..."
- "Let me make sure this works..." / "I'll run a quick test..." / "Let me execute this to check..."
- "I'll use python -c to verify..." / "I'll run python -c to test..." / "Let me execute this inline..."
- "I'll use uv run python -c..." / "Let me run a quick python command..."
- "I'll test this with a simple python execution..." / "I'll create a script to test..."
- "Let me create an example..." / "I'll make a demo..." / "Here's a sample script..."
- "I'll write a quick validation..." / "Let me create a verification..."

**🔧 WHEN YOU CATCH YOURSELF USING THESE PHRASES:**
1. **STOP immediately** - Do not proceed with file creation OR command execution
2. **REDIRECT to exemplar agent**: Use `Task: exemplar - [describe your testing/validation need]`
3. **Check the validation decision tree below** 
4. **Use approved validation methods only**
5. **Ask**: "Can I accomplish this goal using existing tests or debugging tools?"
6. **If unclear**: Ask for guidance rather than creating standalone validation

### 🤖 EXEMPLAR TESTING SUBAGENT AVAILABLE:
When you need ANY form of testing, validation, or verification, use:
```
Task: exemplar
```
This specialized agent handles ALL testing scenarios including:
- Creating examples and demonstrations as proper pytest tests
- Converting standalone validation needs into proper tests  
- Creating new test files or extending existing ones
- Replacing "quick verification" commands with systematic tests

### ✅ CORRECT APPROACH - ALWAYS USE PYTEST:
- Create tests in `tests/` directory following existing structure
- Use proper pytest conventions with `def test_*()` functions
- Import required modules properly
- Use pytest fixtures and assertions
- Ensure tests integrate with CI/CD pipeline

### 📋 CONCRETE EXAMPLES:

**❌ WRONG (Standalone File-Based Validation)**:
```python
# test_something.py in project root
import json
from my_module import MyClass
obj = MyClass()
result = obj.serialize()
print("Success!")
assert result["field"] == "expected"
```

**❌ WRONG (Standalone Command-Based Validation)**:
```bash
uv run python -c "
import logging
from buttermilk._core.execution_context import ExecutionContext
from buttermilk._core.log import logger

ctx = ExecutionContext()
logger.info('Test log message')
print('Logging test complete')
"
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

**ENFORCEMENT**: If you catch yourself about to create standalone validation (file OR command), STOP and ask: "Am I using proper pytest tests in the tests/ directory OR approved debugging tools?" If no, RESTART your approach.

## 🚨 POST-IMPLEMENTATION VALIDATION PROTOCOL 🚨

**When you need to validate or verify your implementation works:**

### ✅ APPROVED VALIDATION METHODS:
1. **Run existing tests**: Use `uv run pytest tests/[relevant_module]` to run existing test suites
2. **Use debugging tools**: Refer to `docs/bots/debugging.md` for validation tools
3. **Check with project's validation tools**: Use documented debugging and monitoring tools
4. **Extend existing tests**: Add test cases to existing test files in `tests/` directory
5. **Manual verification**: Use project's established debugging endpoints and tools

### ❌ FORBIDDEN VALIDATION METHODS (Both File-Based AND Command-Based):
- Creating any standalone validation scripts or files
- Creating "quick test" files anywhere outside tests/
- Writing verification code in project root or implementation directories
- Creating demo files, sample scripts, or proof-of-concept files
- **Using inline Python commands for validation**: `uv run python -c "..."`, `python -c "..."`, or similar
- **Executing Python code via Bash**: Multi-line Python in heredocs, command strings, or pipes
- **"Quick verification" commands**: Any form of standalone Python execution for testing purposes

### 🔄 VALIDATION DECISION TREE:

**🛑 CRITICAL CHECKPOINT: Before ANY testing/validation action, ask:**
- Am I about to create standalone validation in ANY form (file-based OR command-based)?
- Am I about to use `python -c`, `uv run python -c`, or similar inline execution?
- Am I about to create ANY form of standalone test code (file or command)?
- **IF YES TO ANY: STOP IMMEDIATELY and use approved methods below**

1. **Do existing tests cover this functionality?** 
   - YES: Run those tests with `uv run pytest tests/path/to/test.py`
   - NO: Go to step 2

2. **Can I add a test case to an existing test file?**
   - YES: Edit the existing test file in tests/ directory
   - NO: Go to step 3

3. **Do I need to create a completely new test?**
   - Create it in tests/ directory following pytest conventions
   - NOT in project root or implementation directories

4. **Do I just need to verify basic functionality?**
   - Use the project's debugging tools from docs/bots/debugging.md
   - Use existing API endpoints or monitoring tools
   - **NEVER** use inline Python execution for "quick verification"

**REMEMBER**: The urge to "quickly test" or "verify it works" is the most common trigger for workflow violations. This includes both creating test files AND using `python -c` for "simple checks". Resist both urges and use proper validation methods.

### 🆘 WHEN YOU HIT VALIDATION ROADBLOCKS:

**Common scenarios that trigger violations:**
- "The existing tests don't cover this exact case"
- "I need to test this quickly before moving on"
- "Let me just verify my changes work"
- "The debugging tools seem complex for this simple check"

**CORRECT responses to these scenarios:**
1. **No exact test coverage**: Add a test case to the most relevant existing test file
2. **Need quick verification**: Use debugging tools from `docs/bots/debugging.md` or run existing related tests
3. **Want to verify changes**: Run the full test suite or specific relevant test modules
4. **Debugging tools seem complex**: ASK FOR HELP rather than creating workarounds

**🛑 NEVER justify standalone validation with phrases like:**
- "This is just a simple check..." / "It's faster than writing a proper test..."
- "I just need to see if this works..." / "This is temporary validation..."
- "It's just a quick file..." / "This command is simpler..."
- "I'll delete it after..." / "It's just for testing..."

**✅ ALWAYS choose:**
- Proper pytest tests in tests/ directory
- Existing project debugging tools
- Running existing test suites
- Asking for guidance when stuck

## 📚 TESTING & VALIDATION PROTOCOL

**CRITICAL**: When you need ANY form of testing, validation, examples, or verification:

1. **DO NOT** create standalone scripts, files, or use inline commands
2. **IMMEDIATELY** invoke the exemplar testing subagent:
   ```
   Task: exemplar - [describe your testing/validation need]
   ```
3. The subagent will create proper pytest tests that serve as:
   - Living documentation and examples
   - Systematic validation of functionality
   - Reusable test cases for CI/CD

**WHY THIS MATTERS**:
- Examples in tests are verified by CI/CD
- They can't become outdated
- They provide actual test coverage
- They're discoverable by other developers

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
- "Let me create a simple test..."
- "I need to validate this..."

**WHEN YOU CATCH YOURSELF USING THESE PHRASES: STOP. FOLLOW THE PROPER WORKFLOW.**

## 🚨 PREVENTION SYSTEM: Pre-Action Verification 🚨

**UNIVERSAL RULE: Before performing ANY action that involves creating files or testing:**

### 📋 MANDATORY PRE-ACTION CHECKLIST:
**Ask yourself these questions BEFORE taking action:**

1. **Am I about to create standalone validation in ANY form?**
   - File creation: Where am I creating it? Is it in the correct directory?
   - Command execution: Am I using `python -c`, `uv run python -c`, or similar for validation?
   - For tests: MUST be in `tests/` directory with pytest conventions
   - **CRITICAL**: ALL forms of standalone validation (files AND commands) are FORBIDDEN

2. **Am I about to validate/test something?**
   - If YES: Check the validation decision tree in the previous section
   - Use existing tests or debugging tools FIRST
   - **NEVER** use standalone validation in ANY form (files OR commands)

3. **Am I using any red flag phrases?**
   - If YES: STOP immediately and use approved methods instead

4. **Can I accomplish this goal without creating new files?**
   - If MAYBE: Try existing methods first before creating anything new

**ENFORCEMENT: If you cannot answer these questions confidently with approved methods, STOP and ask for guidance.**

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
- **OUTPUT RULE**: Keep outputs concise. When using debugging tools, focus on relevant excerpts. Avoid dumping entire JSON responses or log files. If output exceeds 15 lines, summarize key findings instead. For debugging specifically: extract key findings, limit excerpts to 10-15 lines maximum, summarize patterns rather than listing entries, highlight specific errors only.
- **NO IMPLICIT DEFAULTS**: NEVER use implicit defaults or fallback behaviors. All configuration must be explicit. If something is missing, FAIL FAST with clear error messages. Do not infer, assume, or provide defaults.
- **DATA CONTRACTS**: Schema changes require updating ALL components atomically. See `docs/bots/data-architecture.md` for data contract principles. No defensive programming - trust schemas and let errors propagate.
- **NO SINGLE-USE SCRIPTS**: NEVER create standalone test scripts or dummy examples. Always write proper tests in the existing test suite using pytest conventions. Test files must be reusable, follow project structure, and integrate with the CI/CD pipeline. 

## CRITICAL FIRST STEPS
1. **For Debugging Tasks**: ALWAYS read `docs/bots/debugging.md` FIRST
2. **For Development Tasks**: Follow the 9-step workflow below
3. **For Research Tasks**: Use documented tools, not source code exploration

## DEBUGGING WORKFLOW ENFORCEMENT

**BEFORE using ANY debugging tool, you MUST:**

### ✅ CHECKPOINT: Tool Usage Verification
- **Verify command syntax**: Check documentation examples match tool's actual interface
- **Test basic connectivity**: Use `test-connection` commands before complex operations
- **Validate environment**: Ensure required processes are running before debugging
- **Follow troubleshooting**: If tools fail, consult the troubleshooting section FIRST

### ❌ COMMON DEBUGGING VIOLATIONS TO AVOID:
- Using tools without reading troubleshooting guidance first
- Proceeding when basic connectivity tests fail
- Ignoring tool-specific syntax requirements (e.g., `--wait` needs values)
- Dumping full outputs instead of extracting key findings
- Creating custom debugging scripts instead of using documented tools
- **Using standalone validation for debugging**: Files, `uv run python -c "..."`, or similar
- **Creating "quick debug scripts" or commands** when debugging tools seem difficult to use

### 🚨 DEBUGGING OUTPUT ENFORCEMENT:
When you find yourself about to paste:
- More than 15 lines of tool output
- Full JSON responses or log dumps
- Repetitive log entries or status messages
- **STOP and summarize instead**: Extract 3-5 key findings in bullet points

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

8. **VALIDATE**: Use ONLY approved validation methods (see validation decision tree above). Use existing tests, debugging tools from `docs/bots/debugging.md`, or extend existing test files in `tests/` directory. **NEVER create standalone validation in ANY form (files OR commands).**

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



