---
trigger: always_on
---

## Core Philosophy: Mock Only at System Boundaries

We mock only at system boundaries—where our code interfaces with external systems. The "inside" of our application is tested with real logic and assertions.

**System Boundaries (OK to Mock)**:
- Network: HTTP calls, API requests, websockets
- Filesystem: File I/O (use `tmp_path`)
- Time: System clock (use `freezegun`)
- Environment: Environment variables

**Our Code (NEVER Mock)**:
- Anything in `buttermilk.*`
- Business logic, data transformations, agents
- Internal APIs and orchestration

**Why?** Mocking internal code tests the mock, not your logic. Mock only external dependencies.

## Test Categories

**Unit Tests** (`tests/unit/`):

- Individual functions/classes
- Mock external APIs only (Vertex AI, Zotero API)
- Never mock Buttermilk's own code
- Fast, isolated

**Integration Tests** (`tests/integration/`):

- Multiple Buttermilk components together
- Use `real_bm` fixture
- May mock external APIs if needed
- Test component integration

**End-to-End Tests** (`tests/endtoend/`):

- MANDATORY for task completion
- Complete workflows (fetch → process → store)
- REAL APIs, REAL storage, REAL data
- NO mocks except truly unavoidable external systems
- Run with `pytest -m endtoend`

## Success Criteria

**E2E tests must**:

1. ✅ Use `real_bm` fixture
2. ✅ Call real APIs 
3. ✅ Store in real databases 
4. ✅ Exercise complete workflow
5. ✅ Validate end-to-end behavior
6. ✅ Use realistic test data
7. ✅ Clean up resources
8. ✅ PASS before task is considered complete

**Remember**: NO task is complete without passing E2E tests on real data with real APIs.
