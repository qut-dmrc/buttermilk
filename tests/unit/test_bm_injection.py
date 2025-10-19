"""Unit tests for BM injection system.

Tests the dependency injection mechanism for session-scoped BM instances
without requiring full infrastructure setup.
"""

from unittest.mock import Mock

from buttermilk.runner.flowrunner import FlowRunner


class MockBM:
    """Minimal mock BM for testing injection mechanism."""
    
    def __init__(self, session_id: str):
        self.session_info = Mock()
        self.session_info.session_id = session_id


class TestBMInjectionMechanism:
    """Test the core BM injection functionality."""
    
    def test_flowrunner_bm_injection(self):
        """Test that FlowRunner can store and retrieve session-scoped BM."""
        # Create FlowRunner without BM
        runner = FlowRunner(flows={}, mode="test")
        assert runner.bm is None
        
        # Create mock session BM
        session_bm = MockBM("test-session-123")
        
        # Inject session BM
        runner.set_session_bm(session_bm)
        assert runner.bm is session_bm
        
        # Verify get_effective_bm returns session BM
        effective_bm = runner.get_effective_bm()
        assert effective_bm is session_bm
        assert effective_bm.session_info.session_id == "test-session-123"
    
    def test_flowrunner_fallback_to_global(self, real_bm):
        """Test that FlowRunner falls back to global BM when no session BM is set."""
        runner = FlowRunner(flows={}, mode="test")
        assert runner.bm is None

        # get_effective_bm should return the global BM (real from fixture)
        effective_bm = runner.get_effective_bm()
        assert effective_bm is real_bm  # Should be the real BM from fixture
        # The real BM has an actual session_id, just verify it exists
        assert effective_bm.session_info.session_id
        assert isinstance(effective_bm.session_info.session_id, str)
        assert len(effective_bm.session_info.session_id) > 0
    
    def test_session_isolation_between_runners(self):
        """Test that different FlowRunner instances maintain separate BM sessions."""
        runner1 = FlowRunner(flows={}, mode="test")
        runner2 = FlowRunner(flows={}, mode="test")
        
        bm1 = MockBM("session-1")
        bm2 = MockBM("session-2")
        
        runner1.set_session_bm(bm1)
        runner2.set_session_bm(bm2)
        
        # Verify isolation
        assert runner1.get_effective_bm().session_info.session_id == "session-1"
        assert runner2.get_effective_bm().session_info.session_id == "session-2"
        assert runner1.get_effective_bm() is not runner2.get_effective_bm()


class TestRealBMIntegration:
    """Test BM injection with real BM instances from conftest.py."""
    
    def test_flowrunner_with_real_bm(self, real_bm):
        """Test FlowRunner injection with real BM from test fixtures."""
        # This test uses the real BM from conftest.py
        runner = FlowRunner(flows={}, mode="test")
        
        # Inject the real test BM
        runner.set_session_bm(real_bm)
        
        # Verify injection worked
        effective_bm = runner.get_effective_bm()
        assert effective_bm is real_bm
        assert hasattr(effective_bm, "session_info")
        assert effective_bm.session_info.session_id  # Should have a session ID
    


class TestDocumentedBehavior:
    """Test the documented behavior from our enhanced docstrings."""
    
    def test_session_isolation_example(self):
        """Test the session isolation example from the docstrings."""
        # Create two runners representing different API sessions
        api_session_1 = FlowRunner(flows={}, mode="api")
        api_session_2 = FlowRunner(flows={}, mode="api")
        
        # Each gets its own session-scoped BM (simulating API behavior)
        bm_session_1 = MockBM("api-session-abc123")
        bm_session_2 = MockBM("api-session-def456")
        
        api_session_1.set_session_bm(bm_session_1)
        api_session_2.set_session_bm(bm_session_2)
        
        # Verify each session has isolated observability context
        assert api_session_1.get_effective_bm().session_info.session_id == "api-session-abc123"
        assert api_session_2.get_effective_bm().session_info.session_id == "api-session-def456"
        
        # This solves the original problem: no more shared run_ids between sessions
        assert api_session_1.get_effective_bm() is not api_session_2.get_effective_bm()
