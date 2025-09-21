"""Unit tests for query_runner injection in ConfigurationBootstrapper.

Tests that verify ConfigurationBootstrapper properly injects query_runner into BM instances
and that the query_runner can execute SQL queries as expected.
"""

import pandas as pd
import pytest

from buttermilk._core.config_bootstrap import ConfigurationBootstrapper


class TestBootstrapQueryRunnerInjection:
    """Test query_runner injection and functionality in bootstrapped BM instances."""

    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False

    @pytest.mark.anyio
    async def test_bootstrap_creates_bm_with_query_runner(self, real_conf):
        """
        Test that ConfigurationBootstrapper creates a BM instance with proper query_runner injection.

        This test demonstrates:
        - How to use ConfigurationBootstrapper.bootstrap_full_context()
        - How to use ConfigurationBootstrapper.bootstrap_session_context()
        - That the BM instance has a working query_runner property
        """
        # Step 1: Create ConfigurationBootstrapper
        bootstrapper = ConfigurationBootstrapper(config=real_conf)

        # Step 2: Bootstrap full context (ExecutionContext + Infrastructure)
        execution_context = await bootstrapper.bootstrap_full_context()

        # Verify infrastructure was created successfully
        assert execution_context is not None
        assert infrastructure is not None
        assert execution_context.execution_context_id.startswith("exec-")

        # Step 3: Bootstrap session context using existing infrastructure (dashboard pattern)
        session_bm = await bootstrapper.bootstrap_session_context(
            name="test_query_runner_injection",
            job="verify_query_runner_works",
        )

        # Verify BM instance was created successfully
        assert session_bm is not None
        assert session_bm.session_info.name == "test_query_runner_injection"
        assert session_bm.session_info.job == "verify_query_runner_works"

        # CRITICAL: Verify query_runner is properly injected and accessible
        assert hasattr(session_bm, "query_runner"), "BM instance should have query_runner property"
        assert session_bm.query_runner is not None, "query_runner should not be None"

        # Verify query_runner has the expected interface
        assert hasattr(session_bm.query_runner, "run_query"), "query_runner should have run_query method"
        assert hasattr(session_bm.query_runner, "bq_client"), "query_runner should have bq_client attribute"

    @pytest.mark.anyio
    async def test_bootstrap_bm_query_runner_execution(self, real_conf):
        """
        Test that the BM instance's query_runner can execute a simple SQL query.

        This test demonstrates:
        - Query execution through BM.run_query() method
        - Proper delegation to underlying QueryRunner
        - Expected return types for successful queries
        """
        # Step 1: Create ConfigurationBootstrapper and bootstrap contexts
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        execution_contex = await bootstrapper.bootstrap_full_context()

        # Step 2: Create session BM with infrastructure injection
        session_bm = await bootstrapper.bootstrap_session_context(
            name="test_query_execution",
            job="simple_sql_test",
        )

        # Step 3: Execute a simple SQL query that should work universally
        # Using SELECT with literal values (no table dependencies)
        simple_sql = "SELECT 'test' as message, 42 as number, TRUE as flag"

        # CRITICAL: Verify that BM.run_query() method works
        result = session_bm.run_query(simple_sql)

        # Verify result is returned as expected
        assert result is not None, "Query should return a result"
        assert isinstance(result, pd.DataFrame), "Query should return a pandas DataFrame by default"
        assert len(result) == 1, "Simple query should return exactly one row"

        # Verify expected column values
        assert result.iloc[0]["message"] == "test", "Message column should contain expected value"
        assert result.iloc[0]["number"] == 42, "Number column should contain expected value"
        assert result.iloc[0]["flag"] is True, "Flag column should contain expected value"

    @pytest.mark.anyio
    async def test_bootstrap_query_runner_error_handling(self, real_conf):
        """
        Test that query_runner properly handles SQL errors and reports them appropriately.

        This test demonstrates:
        - Error handling for invalid SQL
        - Proper exception propagation
        - Logging of query failures
        """
        # Step 1: Bootstrap BM instance
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        execution_context = await bootstrapper.bootstrap_full_context()
        session_bm = await bootstrapper.bootstrap_session_context(name="test_error_handling", job="invalid_sql_test")

        # Step 2: Execute invalid SQL that should fail
        invalid_sql = "SELECT FROM WHERE INVALID SYNTAX"

        # CRITICAL: Verify that errors are handled appropriately
        # QueryRunner.run_query should return None for failures, not raise exceptions
        result = session_bm.run_query(invalid_sql)

        # Verify that failure is indicated by None return value
        assert result is None, "Invalid SQL should return None to indicate failure"

    @pytest.mark.anyio
    async def test_bootstrap_session_bm_has_run_query_convenience_method(self, real_conf):
        """
        Test that BM instances have the run_query convenience method that delegates properly.

        This test demonstrates:
        - BM.run_query() method exists and works
        - Proper delegation to QueryRunner.run_query()
        - Consistency with dashboard usage pattern (bm.run_query())
        """
        # Step 1: Bootstrap session BM following dashboard pattern
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        execution_context = await bootstrapper.bootstrap_full_context()
        session_bm = await bootstrapper.bootstrap_session_context(name="streamlit_dashboard_pattern", job="tja_template_analysis_pattern")

        # Step 2: Verify BM has run_query method (used by dashboard)
        assert hasattr(session_bm, "run_query"), "BM should have run_query convenience method"
        assert callable(session_bm.run_query), "BM.run_query should be callable"

        # Step 3: Test the convenience method works like dashboard expects
        test_sql = "SELECT 'dashboard_test' as source, 100 as metric_value"

        # This should work exactly like: template_performance_comparison = bm.run_query(sql_perf)
        result = session_bm.run_query(test_sql)

        assert result is not None, "Dashboard-style query should succeed"
        assert isinstance(result, pd.DataFrame), "Should return DataFrame for dashboard consumption"
        assert len(result) == 1, "Test query should return one row"
        assert result.iloc[0]["source"] == "dashboard_test", "Should return expected test data"
        assert result.iloc[0]["metric_value"] == 100, "Should return expected metric value"

    @pytest.mark.anyio
    async def test_bootstrap_infrastructure_sharing_preserves_query_runner(self, real_conf):
        """
        Test that multiple BM instances share infrastructure but each has access to query_runner.

        This test demonstrates:
        - Infrastructure sharing between multiple sessions
        - Each session BM having its own working query_runner access
        - Consistency across multiple bootstrapped sessions
        """
        # Step 1: Bootstrap shared infrastructure once
        bootstrapper = ConfigurationBootstrapper(config=real_conf)
        execution_context = await bootstrapper.bootstrap_full_context()

        # Step 2: Create multiple session BM instances sharing infrastructure
        session_bm_1 = await bootstrapper.bootstrap_session_context(name="shared_infra_test_1", job="session_1")

        session_bm_2 = await bootstrapper.bootstrap_session_context(name="shared_infra_test_2", job="session_2")

        # Step 3: Verify both sessions have working query_runner access
        test_sql = "SELECT 'session_test' as test_type, 1 as session_num"

        # Both should be able to execute queries independently
        result_1 = session_bm_1.run_query(test_sql)
        result_2 = session_bm_2.run_query(test_sql)

        assert result_1 is not None, "Session 1 should be able to execute queries"
        assert result_2 is not None, "Session 2 should be able to execute queries"
        assert isinstance(result_1, pd.DataFrame), "Session 1 should return DataFrame"
        assert isinstance(result_2, pd.DataFrame), "Session 2 should return DataFrame"

        # Verify both have access to the same underlying infrastructure
        assert session_bm_1.query_runner is not None, "Session 1 should have query_runner"
        assert session_bm_2.query_runner is not None, "Session 2 should have query_runner"

        # Both should be able to access the same BigQuery client through shared infrastructure
        assert session_bm_1.query_runner.bq_client is session_bm_2.query_runner.bq_client, (
            "Both sessions should share the same BigQuery client through infrastructure sharing"
        )
