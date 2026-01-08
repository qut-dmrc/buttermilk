# Buttermilk Project Instructions

@docs/bots/_CORE.md

## Project-Specific Notes

- **NO MOCKS, NO FAKES** - Never use `@patch` or `Mock()` in this project. Tests use real APIs and real data. If you can't test something with real calls, don't write the test.
- so the point of an end2end test using our live config is that you should be able to safely load real data into our real stores. In this example, you shouldn't be mocking anything -- the pipeline endtoend test should query the real zotero api and update the real chromadb library and do everything in between properly.
- no single use tests. that one should be saved for future investigations. make a new test dir called tests/investigations/ and put it in there.
- when debugging, remember not just to prove the fail case -- you also need to prove a success case before you can reliably identify the failure
- if the code review agent raises concerns about your code, you should LISTEN to it and CAREFULLY evaluate the potential options to revise or refactor. YOU ARE NOT SPECIAL; the rules apply to you, and the rules exist for a reason.
- in tests/endtoend: "run these tests with `uv run pytest -m endtoend` DEPENDENCIES: `uv sync --extra dev --extra research --extra azure --upgrade`
- You are working on the 'buttermilk' project. Use the 'bmem' skill to search for and update context in `$ACA_DATA/projects/buttermilk/`
- use '-m ""' to run all tests (alternatively, either/or '-m "slow"' or '-m "endtoend"'?)
- **Debugging and Log Analysis**: Use `ws_debug_cli` for all log analysis. See `docs/bots/DEBUGGING.md` for the authoritative Golden Path.
  - **Analyze logs**: `uv run python -m buttermilk.debug.ws_debug_cli analyze --file <path>`
  - **View logs**: `uv run python -m buttermilk.debug.ws_debug_cli logs -n 50 --file <path>`
  - **Live debugging**: Use `ws_debug_cli start ...` (see Golden Path guide).
  - Note: `/tmp/bm_*` files are **LOG** files, not execution traces. Do not use `trace_analysis` on them.
- our framework goal is success first time, every time, with just-in-time information, that doesn't cause unecessary cost or delay.
