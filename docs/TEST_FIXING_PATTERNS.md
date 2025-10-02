Your job is to fix our broken tests. First read 'docs/agents/TESTER_QA.md' for general testing info.

## AUTONOMOUS OPERATION

**IMPORTANT**: Execute cycles continuously without asking for permission. Keep fixing tests until:
- Pass rate reaches >90%, OR
- Only complex/decision tests remain (documented in TESTS_NEEDING_DECISIONS.md)

After each cycle, commit your changes and immediately start the next cycle. Only stop when you can no longer make progress.

## EFFICIENT WORKFLOW - CYCLE-BASED APPROACH

### Phase 1: Initial Assessment & Easy Wins (20 min)
1. **Run health dashboard**: `uv run python scripts/test_health_dashboard.py`
2. **Fix all collection errors** across ALL categories:
   - Run: `uv run ruff check tests/ --output-format=concise | grep "F821\|import"`
   - Fix import paths, add skips for removed classes
   - These block test execution - highest priority
3. **Run ruff auto-fixes** across entire test suite:
   ```bash
   uv run ruff check --fix tests/
   ```
4. **Commit**: `fix(tests): collection errors and ruff auto-fixes`
5. **Re-run dashboard** to see new baseline

### Phase 2: Low-Hanging Fruit Across Categories (30-40 min)
Focus on **easy patterns** that appear in multiple files:

**Common Easy Fixes** (check across all categories):
- Field renames (e.g., `name` → `project_name`, `llms_instance` → `llms`)
- Type alias changes (e.g., `StorageConfig` → `BigQueryStorageConfig`)
- Simple assertion updates (expected values changed)
- Missing pytest markers (`@pytest.mark.anyio`)

**Strategy**:
```bash
# Find all instances of a pattern
grep -r "old_pattern" tests/ --include="*.py" | wc -l

# Fix in bulk with sed if straightforward
find tests/ -name "*.py" -exec sed -i 's/old_pattern/new_pattern/g' {} \;

# Or fix 5-10 files manually if complex
```

**Commit after each pattern**: `fix(tests): update [pattern] across test suite`

### Phase 3: Category-Specific Issues (30 min cycles)
Pick ONE category, fix systematically, commit, move to next:

1. **Run ruff on category**: `uv run ruff check tests/[category]/ --output-format=concise`
2. **Run category tests**: `uv run pytest tests/[category]/ -v --tb=no -o addopts=""`
3. **Group failures by error type**:
   - Import errors → Fix imports or skip
   - Assertion errors → Check if expected behavior changed
   - Type errors → Check API changes
4. **Fix 5-10 files** (or all if fewer failures)
5. **Commit**: `fix(tests): [category] - [summary of fixes]`
6. **Move to next category**

**Category Order** (by ease):
1. unit/ (usually simpler, fewer dependencies)
2. integration/ (medium complexity)
3. api/ (depends on setup)
4. groupchat/ (complex interactions)
5. endtoend/ (may need external services)

### Phase 4: Mark Tests Needing Decisions (ongoing)
When you encounter tests where expected behavior is unclear:

1. **Add pytest.mark.skip** with detailed reason:
```python
@pytest.mark.skip(reason="""
NEEDS DECISION: Test expects X but code now does Y.
- Old behavior: [describe]
- New behavior: [describe]
- Question: Should test be updated or code reverted?
See: [link to relevant code/issue if available]
""")
def test_something():
    ...
```

2. **Keep a running list** in `TESTS_NEEDING_DECISIONS.md`:
```markdown
## Tests Requiring Management Decisions

### test_foo.py::test_bar
- **Issue**: Expected behavior changed
- **Old**: Did X
- **New**: Does Y
- **Question**: Which is correct?
- **File**: tests/category/test_foo.py:123

### test_baz.py::test_qux
...
```

3. **Don't spend >5 min** figuring out expected behavior - mark and move on

### Phase 5: Stubborn Tests (focused sessions)
After easy fixes, tackle remaining failures:

1. **Group by error pattern**:
   ```bash
   # Extract error patterns from test output
   uv run pytest tests/ -v --tb=no | grep "FAILED" | awk '{print $NF}' | sort | uniq -c
   ```

2. **Focus on one error pattern** across multiple tests
3. **Deep dive** - read source code, check git history, understand changes
4. **Fix or mark for decision**
5. **Commit**: `fix(tests): resolve [error pattern] issues`

## COMMIT STRATEGY

**Commit frequently** at natural breakpoints:

✅ **Good commit points**:
- After fixing all collection errors
- After bulk pattern replacement (e.g., field renames)
- After fixing a category (if <50 changes)
- After fixing specific error pattern
- After marking batch of tests for decisions
- End of each work session

❌ **Avoid**:
- Massive commits with 100+ files
- Mixing unrelated fixes
- Committing broken tests

**Commit message format**:
```
fix(tests): [scope] - [concise summary]

- Fixed [specific thing 1]
- Updated [specific thing 2]
- Marked [test] for decision (reason)

Pass rate: X% → Y%
```

## DECISION TREE FOR EACH FAILING TEST

```
Test fails
├─ Collection error?
│  ├─ Import issue? → Fix import or skip file
│  └─ Syntax error? → Let ruff fix or fix manually
│
├─ Ruff reports issue?
│  ├─ F821 (undefined)? → Check if class exists → fix or skip
│  ├─ Auto-fixable? → Run ruff --fix
│  └─ Manual fix needed? → Fix according to diagnostics
│
├─ Quick fix obvious (<2 min)?
│  ├─ Field rename? → Update
│  ├─ Import path? → Update
│  └─ Expected value? → Update if clearly correct
│
├─ Needs investigation (>5 min)?
│  ├─ Can determine expected behavior? → Fix
│  └─ Unclear? → Mark for decision, move on
│
└─ Mark for decision if:
   - Expected behavior unclear
   - Requires business logic understanding
   - Multiple valid interpretations
```

## TRACKING PROGRESS

**Update after each cycle**:
```bash
uv run python scripts/test_health_dashboard.py
git diff test_health_report.md  # See improvement
```

**Target metrics**:
- Collection errors: 0
- Pass rate: >90%
- Failures needing decisions: Clearly documented

## EXAMPLE WORK SESSION (1 hour)

```
0:00-0:05  Run dashboard, assess state
0:05-0:15  Fix all collection errors → commit
0:15-0:25  Run ruff auto-fixes → commit
0:25-0:35  Fix field rename pattern across suite → commit
0:35-0:50  Fix unit/ category completely → commit
0:50-0:55  Mark 3 unclear tests for decision
0:55-1:00  Re-run dashboard, update progress
```

**Result**: ~30-50 tests fixed, clear commits, known decision points

## RUFF DIAGNOSTICS GUIDE

### Priority Errors (Fix First)
| Code | Meaning | Action | Time |
|------|---------|--------|------|
| E999 | Syntax error | Auto-fix or manual | 1 min |
| F821 | Undefined name | Check if exists → fix or skip | 2-5 min |
| F401 | Unused import | Auto-fix | 0 min |
| F841 | Unused variable | Auto-fix | 0 min |
| E251 | Spacing | Auto-fix | 0 min |

### Common Patterns → Bulk Fixes

**Field renames**:
```bash
# Find all instances
grep -r "\.old_field" tests/ --include="*.py"

# Replace in bulk
find tests/ -name "*.py" -exec sed -i 's/\.old_field/.new_field/g' {} \;

# Test it worked
uv run pytest tests/ --co -q  # Check collection
```

**Type changes**:
```python
# Before: Union type alias can't be instantiated
config = StorageConfig(type="bigquery", ...)

# After: Use specific type
config = BigQueryStorageConfig(type="bigquery", ...)
```

**Import paths**:
```python
# Before: Class moved
from buttermilk.data.vector import GeminiEmbeddingFunction

# After: New location
from buttermilk.processors.embeddings import GeminiEmbeddingFunction
```

## MOCKING PHILOSOPHY

**Mock only at system boundaries**:
- ✅ Network calls (httpx, requests)
- ✅ Filesystem (temp directories)
- ✅ Time (freezegun)
- ✅ Environment variables
- ❌ Internal buttermilk code
- ❌ Business logic

**When you find internal mocks**:
1. If test is simple → Rewrite without mock
2. If test is complex → Mark for decision (may need refactor)

See [TESTING_PHILOSOPHY.md](TESTING_PHILOSOPHY.md) for details.

## WHEN TO ASK FOR HELP

**Immediate**:
- Test requires external service credentials
- Security/auth related failures
- Database schema migrations needed

**Mark for decision**:
- Expected behavior genuinely unclear
- Business logic questions
- Multiple valid interpretations

**Figure out yourself** (use git, code reading):
- Import paths
- Field renames
- Type signature changes
- Expected test values

## COMPLETION CRITERIA

**Done when**:
- [ ] Collection errors: 0
- [ ] Pass rate: >90%
- [ ] All remaining failures documented in TESTS_NEEDING_DECISIONS.md
- [ ] All commits pushed
- [ ] test_health_report.md shows improvement

**Not done until**:
- You've made multiple cycles through categories
- You've fixed all "easy" patterns
- Only complex/decision tests remain
