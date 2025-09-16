DO NOT PROCEED BEFORE READING @docs/bots/INSTRUCTIONS.md

If you try to read or edit another document before reading these, something BAD will happen.

CRITICAL: After reading INSTRUCTIONS.md, you MUST follow the 9-step workflow for ANY code changes:
1. STOP & understand scope
2. ANALYZE architecture
3. PLAN with GitHub issues
4. TEST first
5. IMPLEMENT minimally
6. DOCUMENT thoroughly
7. VALIDATE with debugging tools
8. COMMIT with documentation AND CREATE PR
9. REFLECT and update docs

NO EXCEPTIONS. If you make code changes without following all 9 steps, STOP and restart the workflow.

⚠️ YOUR WORK IS NOT COMPLETE UNTIL IT'S IN A PR!

MINDSET: You are a software engineer contributing to a long-term project, not a quick-fix assistant - write proper tests, follow the workflow, and commit your work.

DESIGN CHOICES:
- DRY, modular code
- DO NOT MAINTAIN backwards compatibility: one path only.
- **FAIL FAST PHILOSOPHY**: Anything we build either works or it doesn't, nothing in between. This means **NO FALLBACKS**, **NO DEFENSIVE PROGRAMMING**, one golden path, no error recovery, no alternatives. Ever.
- Validate code with FULL end-to-end integration tests with live data; NO COMPROMISES. 
- If you get stuck, STOP and ASK FOR HELP. No workarounds, no loops.

WARNING: You may be interrupted at any time and all your memory and progress will be lost. DOCUMENT your progress and COMMIT at every opportunity!