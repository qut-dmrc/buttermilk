"""Test AgentConfig schema fields for record and context.

This test suite validates Option B implementation: adding explicit `record`
and `context` fields to AgentConfig that map directly to AgentInput.record
and AgentInput.context instead of going through the `inputs` dict.
"""

import pytest

from buttermilk._core.config import AgentConfig


def test_agentconfig_accepts_toplevel_record_field():
    """Test that AgentConfig accepts a top-level record field (JMESPath string)."""
    config = AgentConfig(
        role="judge",
        description="Test agent with record field",
        record="[FETCH.outputs]||*.record",  # Top-level, goes to .record
        inputs={"criteria": "some.path"},  # Regular inputs
    )

    # Should have the record field accessible
    assert hasattr(config, "record")
    assert config.record == "[FETCH.outputs]||*.record"

    # Regular inputs should still work
    assert config.inputs["criteria"] == "some.path"

    # Record should NOT be in inputs dict
    assert "record" not in config.inputs


def test_agentconfig_accepts_toplevel_context_field():
    """Test that AgentConfig accepts a top-level context field (JMESPath string)."""
    config = AgentConfig(
        role="judge",
        description="Test agent with context field",
        context="[SESSION.context]||defaults.context",  # Top-level, goes to .context
        inputs={"criteria": "some.path"},  # Regular inputs
    )

    # Should have the context field accessible
    assert hasattr(config, "context")
    assert config.context == "[SESSION.context]||defaults.context"

    # Regular inputs should still work
    assert config.inputs["criteria"] == "some.path"

    # Context should NOT be in inputs dict
    assert "context" not in config.inputs


def test_agentconfig_both_record_and_context():
    """Test that AgentConfig can have both record and context fields."""
    config = AgentConfig(
        role="judge",
        description="Test agent with both fields",
        record="[FETCH.outputs]||*.record",
        context="[SESSION.context]||defaults.context",
        inputs={"criteria": "some.path", "threshold": "params.threshold"},
    )

    # Both fields should be accessible
    assert config.record == "[FETCH.outputs]||*.record"
    assert config.context == "[SESSION.context]||defaults.context"

    # Regular inputs should still work
    assert config.inputs["criteria"] == "some.path"
    assert config.inputs["threshold"] == "params.threshold"

    # Neither should be in inputs dict
    assert "record" not in config.inputs
    assert "context" not in config.inputs


def test_agentconfig_record_in_inputs_dict_raises_valueerror():
    """Test that having record in BOTH top-level AND inputs dict raises ValueError.

    This prevents ambiguity - record should only be specified in one place.
    """
    with pytest.raises(
        ValueError,
        match=".*record.*ambiguous.*",  # Should mention record and ambiguity
    ):
        AgentConfig(
            role="judge",
            description="Test agent with ambiguous record",
            record="[FETCH.outputs]",  # Top-level
            inputs={"record": "[OTHER.outputs]"},  # Also in inputs - CONFLICT!
        )


def test_agentconfig_context_in_inputs_dict_raises_valueerror():
    """Test that having context in BOTH top-level AND inputs dict raises ValueError.

    This prevents ambiguity - context should only be specified in one place.
    """
    with pytest.raises(
        ValueError,
        match=".*context.*ambiguous.*",  # Should mention context and ambiguity
    ):
        AgentConfig(
            role="judge",
            description="Test agent with ambiguous context",
            context="[SESSION.context]",  # Top-level
            inputs={"context": "[OTHER.context]"},  # Also in inputs - CONFLICT!
        )


def test_agentconfig_both_fields_conflict_raises_valueerror():
    """Test that conflicts in both record and context are detected.

    Even if both fields have conflicts, the validator should raise an error
    (may detect one or both depending on validation order).
    """
    with pytest.raises(ValueError):
        AgentConfig(
            role="judge",
            description="Test agent with both fields conflicting",
            record="[FETCH.outputs]",  # Top-level
            context="[SESSION.context]",  # Top-level
            inputs={
                "record": "[OTHER.outputs]",  # Conflict!
                "context": "[OTHER.context]",  # Conflict!
            },
        )


def test_agentconfig_record_optional():
    """Test that record field is optional - can create config without it."""
    config = AgentConfig(
        role="judge",
        description="Test agent without record",
        inputs={"criteria": "some.path"},
    )

    # Should have record attribute but it should be None or empty
    assert hasattr(config, "record")
    assert config.record in (None, "")


def test_agentconfig_context_optional():
    """Test that context field is optional - can create config without it."""
    config = AgentConfig(
        role="judge",
        description="Test agent without context",
        inputs={"criteria": "some.path"},
    )

    # Should have context attribute but it should be None or empty
    assert hasattr(config, "context")
    assert config.context in (None, "")


def test_agentconfig_backward_compatibility():
    """Test backward compatibility - existing configs without record/context still work.

    This ensures that configs using the old pattern (record/context in inputs dict)
    continue to work when neither top-level field is specified.
    """
    # Old pattern: record and context in inputs dict, no top-level fields
    config = AgentConfig(
        role="judge",
        description="Test agent with old pattern",
        inputs={
            "record": "[FETCH.outputs]",  # Old way - should work
            "context": "[SESSION.context]",  # Old way - should work
            "criteria": "some.path",
        },
    )

    # Config should be created successfully
    assert config.role == "JUDGE"
    assert config.inputs["record"] == "[FETCH.outputs]"
    assert config.inputs["context"] == "[SESSION.context]"

    # Top-level fields should be None or empty (not set)
    assert config.record in (None, "")
    assert config.context in (None, "")
