"""Tests for OpenTelemetry span attribute capture.

Validates that critical attributes (project name, agent parameters, hashes)
are properly captured in OTEL spans.
"""

from buttermilk.utils.otel import span_with_session


class TestProjectNameCapture:
    """Test project name is captured in all spans."""

<<<<<<< HEAD
    def test_project_name_in_span_attributes(self, real_bm, tracer_provider, get_recorded_spans):
=======
    def test_project_name_in_span_attributes(
        self, real_bm, tracer_provider, get_recorded_spans
    ):
>>>>>>> origin/stable
        """Verify project name appears as span attribute."""
        # Arrange: Set up BM with specific project name
        # Note: span_with_session uses the global bm singleton, so we need to
        # set the project name on the actual global bm instance
        from buttermilk import bm
<<<<<<< HEAD

=======
>>>>>>> origin/stable
        original_project_name = bm.session_info.project_name
        bm.session_info.project_name = "test_project_name"

        try:
            # Act: Create span with session context
            with span_with_session(
                session_id=bm.session_info.session_id,
                name="test.operation",
                attributes={"test.key": "test.value"},
                kind="internal",
            ):
                pass

            # Assert: Project name captured in span
            spans = get_recorded_spans()
            assert len(spans) == 1, "Should create exactly one span"

            span = spans[0]
            # Span exists (ReadableSpan from exporter)
            assert span is not None

            # Verify attributes
<<<<<<< HEAD
            assert "buttermilk.project.name" in span.attributes, "Project name attribute missing"
=======
            assert "buttermilk.project.name" in span.attributes, (
                "Project name attribute missing"
            )
>>>>>>> origin/stable
            assert span.attributes["buttermilk.project.name"] == "test_project_name"
        finally:
            # Restore original project name
            bm.session_info.project_name = original_project_name

<<<<<<< HEAD
    def test_project_name_propagates_to_child_spans(self, real_bm, tracer_provider, get_recorded_spans):
=======
    def test_project_name_propagates_to_child_spans(
        self, real_bm, tracer_provider, get_recorded_spans
    ):
>>>>>>> origin/stable
        """Verify project name propagates to all child spans."""
        real_bm.session_info.project_name = "child_test_project"

        with span_with_session(
            session_id=real_bm.session_info.session_id,
            name="parent.operation",
            kind="internal",
        ):
            # Create child span
            with span_with_session(
                session_id=real_bm.session_info.session_id,
                name="child.operation",
                kind="internal",
            ):
                pass

        spans = get_recorded_spans()
        assert len(spans) == 2, "Should create parent and child spans"

        # Both spans should have project name
        for span in spans:
            assert "buttermilk.project.name" in span.attributes
            assert span.attributes["buttermilk.project.name"] == "child_test_project"

<<<<<<< HEAD
    def test_project_name_fallback_when_bm_unavailable(self, tracer_provider, get_recorded_spans):
=======
    def test_project_name_fallback_when_bm_unavailable(
        self, tracer_provider, get_recorded_spans
    ):
>>>>>>> origin/stable
        """Verify graceful handling when BM not available.

        When span_with_session is called with a session_id but BM is not available,
        the project name should default to "unknown" as a graceful fallback.
        """
        # Act: Create span without BM context (session_id provided but BM not initialized)
        with span_with_session(
            session_id="test-session-123",
            name="test.operation",
            kind="internal",
        ):
            pass

        spans = get_recorded_spans()
        assert len(spans) == 1

        # When session_id is provided, project_name is always set
        # Falls back to "unknown" when BM is not available
        project_name = spans[0].attributes.get("buttermilk.project.name")
<<<<<<< HEAD
        assert project_name == "unknown", f"Expected project name to be 'unknown' when BM not available, got: {project_name}"
=======
        assert project_name == "unknown", (
            f"Expected project name to be 'unknown' when BM not available, got: {project_name}"
        )
>>>>>>> origin/stable


class TestAgentTypeFormatting:
    """Test agent type is formatted as simple name, not full class path."""

    def test_agent_type_simple_name(self, real_bm):
        """Verify agent type is simple lowercase name."""
        from buttermilk._core.agent import get_agent_type_for_trace
        from buttermilk.agents.judge import Judge

        # Create judge agent
        judge = Judge(
            agent_id="test_judge",
            agent_name="Test Judge",
            role="JUDGE",
            description="Test judge",
            parameters={"model": "test-model", "template": "test template"},
        )

        # Act: Get agent type for tracing
        agent_type = get_agent_type_for_trace(judge)

        # Assert: Simple lowercase name
        assert agent_type == "judge"
        assert "<class" not in agent_type
        assert "buttermilk" not in agent_type

    def test_agent_type_various_agents(self, real_bm):
        """Test agent type extraction for different agent classes."""
        from buttermilk._core.agent import get_agent_type_for_trace
        from buttermilk.agents.fetch import FetchAgent
        from buttermilk.agents.judge import Judge

        test_cases = [
            (FetchAgent, "fetchagent"),
            (Judge, "judge"),
        ]

        for agent_class, expected_type in test_cases:
            params = {}
            if agent_class == Judge:
                params = {"model": "test-model", "template": "test template"}

            agent = agent_class(
                agent_id=f"test_{expected_type}",
                agent_name=f"Test {expected_type}",
                role=expected_type.upper(),
                description="Test",
                parameters=params,
            )

            agent_type = get_agent_type_for_trace(agent)
            assert agent_type == expected_type

    def test_agent_trace_info_includes_full_class_path(self, real_bm):
        """Verify full class path preserved as separate attribute."""
        from buttermilk._core.agent import create_agent_trace_info
        from buttermilk.agents.judge import Judge

        judge = Judge(
            agent_id="test_judge",
            agent_name="Test Judge",
            role="JUDGE",
            description="Test",
            parameters={"model": "test-model", "template": "test template"},
        )

        # Act: Create trace info
        trace_info = create_agent_trace_info(judge)

        # Assert: Both simple and full types present
        assert trace_info["agent_type"] == "judge"  # Simple
        assert "agent_class" in trace_info  # Full path
        assert "buttermilk.agents.judge.Judge" in trace_info["agent_class"]


class TestAgentParameterCapture:
    """Test critical agent parameters are captured in traces."""

    def test_template_parameter_captured(self, real_bm):
        """Verify template name is captured in agent trace info."""
        from buttermilk._core.agent import create_agent_trace_info
        from buttermilk.agents.judge import Judge

        judge = Judge(
            agent_id="test_judge",
            agent_name="Test Judge",
            role="JUDGE",
            description="Test",
            parameters={
                "template": "judge_template.jinja2",
                "model": "gemini-2.0-flash",
            },
        )

        # Act
        trace_info = create_agent_trace_info(judge)

        # Assert
        assert "template" in trace_info
        assert trace_info["template"] == "judge_template.jinja2"

    def test_model_parameter_captured(self, real_bm):
        """Verify model identifier is captured."""
        from buttermilk._core.agent import create_agent_trace_info
        from buttermilk.agents.judge import Judge

        judge = Judge(
            agent_id="test_judge",
            agent_name="Test Judge",
            role="JUDGE",
            description="Test",
            parameters={
                "model": "gemini-2.0-flash-thinking-exp",
                "template": "judge.jinja2",
            },
        )

        trace_info = create_agent_trace_info(judge)

        assert "model" in trace_info
        assert trace_info["model"] == "gemini-2.0-flash-thinking-exp"

    def test_template_hash_captured(self, real_bm):
        """Verify template hash is captured when provided."""
        from buttermilk._core.agent import create_agent_trace_info
        from buttermilk.agents.judge import Judge

        judge = Judge(
            agent_id="test_judge",
            agent_name="Test Judge",
            role="JUDGE",
            description="Test",
            parameters={
                "template": "judge_template.jinja2",
                "model": "gemini-2.0-flash",
            },
        )

        # Act: Provide template hash
        trace_info = create_agent_trace_info(judge, template_hash="abc123def456")

        # Assert
        assert "template_hash" in trace_info
        assert trace_info["template_hash"] == "abc123def456"

    def test_template_hash_from_parameters(self, real_bm):
        """Verify template hash from agent parameters if not provided separately."""
        from buttermilk._core.agent import create_agent_trace_info
        from buttermilk.agents.judge import Judge

        judge = Judge(
            agent_id="test_judge",
            agent_name="Test Judge",
            role="JUDGE",
            description="Test",
            parameters={
                "template": "judge_template.jinja2",
                "template_hash": "xyz789",  # Pre-computed in parameters
                "model": "gemini-2.0-flash",
            },
        )

        # Act: Don't provide template_hash argument
        trace_info = create_agent_trace_info(judge)

        # Assert: Falls back to parameters
        assert "template_hash" in trace_info
        assert trace_info["template_hash"] == "xyz789"

    def test_all_parameters_captured_together(self, real_bm):
        """Verify all critical parameters captured in single trace."""
        from buttermilk._core.agent import create_agent_trace_info
        from buttermilk.agents.judge import Judge

        judge = Judge(
            agent_id="test_judge",
            agent_name="Test Judge",
            role="JUDGE",
            description="Test judge agent",
            parameters={
                "template": "judge.jinja2",
                "model": "gemini-2.0-flash",
                "max_tokens": 1000,
                "temperature": 0.7,
            },
        )

        trace_info = create_agent_trace_info(judge, template_hash="hash123")

        # Assert all critical fields present
        assert trace_info["agent_type"] == "judge"
        assert trace_info["template"] == "judge.jinja2"
        assert trace_info["template_hash"] == "hash123"
        assert trace_info["model"] == "gemini-2.0-flash"

        # Assert full parameters preserved
        assert "parameters" in trace_info
        assert trace_info["parameters"]["max_tokens"] == 1000
