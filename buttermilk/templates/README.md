# Template Development Guide

## Creating New Templates

When creating a new Jinja2 template in this directory, you **MUST** add test cases to ensure fail-fast validation works correctly.

### Required Test Updates

**File**: `tests/integration/test_template_required_variables.py`

Add your template to **BOTH** test parameter lists:

#### 1. Missing Variable Detection Test

Add to `TEMPLATE_TEST_CASES`:

```python
TEMPLATE_TEST_CASES = [
    # ... existing cases ...

    # YOUR NEW TEMPLATE
    ("your_template_name", ["required_var1", "required_var2"], {
        # Provide ALL other variables EXCEPT the required ones being tested
        "optional_var": "value",
        "another_var": "value"
    }),
]
```

**Example**:
```python
# analyst template - requires criteria
("analyst", ["criteria"], {
    "record": "test record content"  # Provide record, omit criteria
}),
```

#### 2. Type Mismatch Detection Test

Add to `TYPE_MISMATCH_TEST_CASES` for **each variable that expects a dict/list**:

```python
TYPE_MISMATCH_TEST_CASES = [
    # ... existing cases ...

    # YOUR NEW TEMPLATE - test dict variable with JSON string
    ("your_template", "dict_var_name",
     {"key": "value"},          # Correct: dict
     '{"key": "value"}',        # Wrong: JSON string
     {
         # All other required variables
         "other_var": "value"
     }),
]
```

**Example**:
```python
# score template - expected should be dict, not string
("score", "expected",
 {"reasons": ["reason 1"], "violating": False},  # Correct: dict
 '{"reasons": ["reason 1"], "violating": false}',  # Wrong: JSON string
 {
     "answers": [{"agent_id": "test", "result": "test result"}],
     "criteria": ["test criterion"]
 }),
```

### Why This Matters

**Templates currently have CRITICAL fail-fast violations**:

1. ❌ **Missing variables not detected** - Conditional logic (`{% if %}`) bypasses tracking
2. ❌ **Wrong types silently accepted** - JSON strings processed as dicts without error
3. ❌ **pprint filter bug** - `{{ var | pprint }}` doesn't trigger variable tracking

These tests ensure your template:
- ✅ Detects missing required variables
- ✅ Rejects or clearly errors on wrong data types
- ✅ Doesn't silently render invalid data

### Common Template Bugs to Avoid

#### Bug 1: Conditional checks hide missing variables

```jinja2
{# ❌ BAD - 'reasons' in undefined variable returns False, doesn't track as missing #}
{%- if 'reasons' in expected -%}
  {{ expected.reasons }}
{%- endif -%}

{# ✅ GOOD - Direct access tracks missing variable #}
{{ expected.reasons }}
```

#### Bug 2: Filters bypass variable tracking

```jinja2
{# ❌ BAD - pprint receives Undefined object, renders as "Undefined" string #}
{{ expected | pprint }}

{# ✅ GOOD - Direct access without filter #}
{{ expected }}
```

#### Bug 3: Type checks on strings vs dicts

```jinja2
{# ❌ BAD - 'key' in string checks substring, not dict key #}
{%- if 'key' in var -%}  {# If var='{"key":"val"}', this is TRUE! #}
  {{ var['key'] }}  {# But this FAILS because strings aren't subscriptable #}
{%- endif -%}

{# ✅ GOOD - Explicit type check or fail-fast #}
{%- if var is mapping -%}
  {{ var['key'] }}
{%- else -%}
  ERROR: Expected dict, got {{ var.__class__.__name__ }}
{%- endif -%}
```

### Test Execution

Run template tests before committing:

```bash
# Run all template validation tests
uv run pytest tests/integration/test_template_required_variables.py -v

# Run only your template's tests
uv run pytest tests/integration/test_template_required_variables.py -v -k "your_template"
```

**All tests must pass** before the template is production-ready.

### What Good Tests Look Like

When your template is properly tested:

```bash
# Missing variable test - SHOULD PASS
test_template_detects_missing_required_variable[your_template-...] PASSED

# Type mismatch test - SHOULD PASS
test_template_with_wrong_type_fails_or_warns[your_template-...] PASSED
```

If tests **FAIL**, your template has fail-fast violations that must be fixed.

## Template Directory Structure

```
templates/
├── README.md              # This file
├── prompt/                # Main prompt templates
│   ├── analyst.jinja2
│   ├── judge.jinja2
│   ├── score.jinja2
│   └── ...
├── format/                # Output format templates
├── criteria/              # Criteria templates
└── snippets/              # Reusable template snippets
```

## Template Naming Conventions

- Use lowercase with underscores: `my_template.jinja2`
- Descriptive names that indicate purpose: `score.jinja2`, `synthesise.jinja2`
- Avoid generic names like `template1.jinja2`

## Template Variables

### Required vs Optional Variables

Clearly document which variables are required:

```jinja2
{#
Template: score
Required variables:
  - answers: list[dict] - Answers to score
  - criteria: list[str] - Scoring criteria
  - expected: dict - Expected ground truth with 'reasons' key

Optional variables:
  - instructions: str - Additional instructions
#}
```

### Variable Type Expectations

Document expected types to prevent BigQuery JSON string bugs:

```jinja2
{#
expected: dict, NOT string
  Example: {"reasons": ["r1"], "violating": false}
  NOT: '{"reasons": ["r1"], "violating": false}'

  Common bug: BigQuery returns JSON columns as strings.
  Fix in SQL: JSON_QUERY() for objects, JSON_VALUE() for scalars
#}
```

## Related Documentation

- **Testing Philosophy**: `docs/TESTING_PHILOSOPHY.md`
- **Fail-Fast Principles**: `bot/docs/_CHUNKS/FAIL-FAST.md` (if available)
- **Template Tests**: `tests/integration/test_template_required_variables.py`

## Questions?

If your template has complex requirements or you're unsure how to test it:

1. Look at existing templates as examples
2. Check `test_template_required_variables.py` for patterns
3. Run the tests and read the failure messages - they explain what's wrong

**Remember**: Templates are part of our fail-fast infrastructure. Silent failures corrupt research data.
