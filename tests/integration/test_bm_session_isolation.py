"""Integration tests for BM session isolation functionality.

These tests verify that the session-scoped BM injection system works correctly
across the FlowRunner -> Orchestrator -> Agent chain, providing proper observability
isolation while maintaining backward compatibility.
"""

import pytest
from unittest.mock import Mock
from buttermilk.runner.flowrunner import FlowRunner
from buttermilk._core.orchestrator import Orchestrator
from buttermilk._core.agent import Agent
from buttermilk._core.types import RunRequest


class MockBM:
    """Mock BM instance for testing session isolation."""
    
    def __init__(self, session_id: str):
        self.session_info = Mock()
        self.session_info.session_id = session_id
        self.session_info.job = "test_job"
        self.session_info.platform = "test_platform"


class MockOrchestrator(Orchestrator):
    """Mock orchestrator for testing BM injection."""
    
    def __init__(self, **kwargs):
        # Minimal initialization for testing
        self.name = kwargs.get("name", "test_orchestrator")
        self.description = kwargs.get("description", "Test orchestrator")
        self.agents = {}
        self.observers = {}
        self.parameters = {}
        self._flow_data = Mock()
        self._bm = None
    
    async def _setup(self, request: RunRequest) -> None:
        pass
    
    async def _run(self, request: RunRequest) -> None:
        pass


class MockAgent(Agent):
    """Mock agent for testing BM injection."""
    
    def __init__(self, **data):
        # Minimal initialization for testing - avoid full Agent init
        self._config = Mock()
        self._config.agent_name = data.get("agent_name", "test_agent")
        self._config.role = data.get("role", "TEST")
        self._config.session_id = data.get("session_id", "")
        
        # Set BM if provided in config
        if "bm" in data:
            self._config.bm = data["bm"]
        else:
            self._config.bm = None
    
    async def _process(self, request):
        # Test method that uses BM
        bm = self.get_effective_bm()
        return {"bm_session_id": bm.session_info.session_id}


@pytest.fixture
def mock_flow_config():
    """Mock flow configuration for testing."""
    return {
        "orchestrator": "tests.integration.test_bm_session_isolation.MockOrchestrator",
        "name": "test_flow",
        "description": "Test flow for BM injection",
        "agents": {},
        "parameters": {}
    }


@pytest.fixture
def session_bm():
    """Session-scoped BM instance for testing."""
    return MockBM("test-session-123")


@pytest.fixture
def global_bm():
    """Global BM instance for testing."""
    return MockBM("global-session")


class TestBMInjectionSystem:
    """Test suite for BM session isolation functionality."""
    
    def test_flowrunner_bm_injection(self, mock_flow_config, session_bm):
        """Test that FlowRunner accepts and stores session-scoped BM."""
        flow_runner = FlowRunner(flows={"test_flow": mock_flow_config}, mode="test")
        
        # Initially should have no BM
        assert flow_runner.bm is None
        
        # Set session BM
        flow_runner.set_session_bm(session_bm)
        assert flow_runner.bm is session_bm
        
        # get_effective_bm should return session BM
        effective_bm = flow_runner.get_effective_bm()
        assert effective_bm is session_bm
        assert effective_bm.session_info.session_id == "test-session-123"
    
    def test_orchestrator_bm_injection(self, session_bm):
        """Test that Orchestrator accepts and stores session-scoped BM."""
        orchestrator = MockOrchestrator(name="test_orch")
        
        # Initially should use global singleton (mocked to raise error)
        with pytest.raises(Exception):  # get_bm() not initialized in test
            orchestrator.get_effective_bm()
        
        # Set session BM
        orchestrator.set_bm(session_bm)
        
        # get_effective_bm should return session BM
        effective_bm = orchestrator.get_effective_bm()
        assert effective_bm is session_bm
        assert effective_bm.session_info.session_id == "test-session-123"
    
    def test_agent_bm_injection(self, session_bm):
        """Test that Agent can access injected BM through config."""
        # Agent with no BM should try to use global singleton
        agent_no_bm = MockAgent(agent_name="test1", role="TEST")
        with pytest.raises(Exception):  # get_bm() not initialized in test
            agent_no_bm.get_effective_bm()
        
        # Agent with injected BM should use it
        agent_with_bm = MockAgent(agent_name="test2", role="TEST", bm=session_bm)
        effective_bm = agent_with_bm.get_effective_bm()
        assert effective_bm is session_bm
        assert effective_bm.session_info.session_id == "test-session-123"
    
    @pytest.mark.asyncio
    async def test_end_to_end_bm_flow(self, mock_flow_config, session_bm):
        """Test complete BM injection flow from FlowRunner to Agent."""
        # This test verifies the complete chain:
        # FlowRunner -> Orchestrator -> Agent all get session-scoped BM
        
        flow_runner = FlowRunner(flows={"test_flow": mock_flow_config}, mode="test")
        flow_runner.set_session_bm(session_bm)
        
        # Create orchestrator (this should get BM injected)
        orchestrator = flow_runner._create_fresh_orchestrator("test_flow")
        
        # Verify orchestrator received session BM
        orch_bm = orchestrator.get_effective_bm()
        assert orch_bm is session_bm
        assert orch_bm.session_info.session_id == "test-session-123"
        
        # Create agent with BM injection (simulate orchestrator creating agent)
        agent = MockAgent(agent_name="test_agent", role="TEST", bm=session_bm)
        
        # Verify agent received session BM
        agent_bm = agent.get_effective_bm()
        assert agent_bm is session_bm
        assert agent_bm.session_info.session_id == "test-session-123"
        
        # Verify all components use the same session-scoped BM instance
        assert flow_runner.get_effective_bm() is orchestrator.get_effective_bm()
        assert orchestrator.get_effective_bm() is agent.get_effective_bm()
    
    def test_session_isolation_between_runners(self, mock_flow_config):
        """Test that different FlowRunner instances have isolated BM sessions."""
        session_bm_1 = MockBM("session-1")
        session_bm_2 = MockBM("session-2")
        
        # Create two FlowRunner instances
        runner_1 = FlowRunner(flows={"test_flow": mock_flow_config}, mode="test")
        runner_2 = FlowRunner(flows={"test_flow": mock_flow_config}, mode="test")
        
        # Set different session BMs
        runner_1.set_session_bm(session_bm_1)
        runner_2.set_session_bm(session_bm_2)
        
        # Verify isolation
        assert runner_1.get_effective_bm().session_info.session_id == "session-1"
        assert runner_2.get_effective_bm().session_info.session_id == "session-2"
        
        # Create orchestrators from each runner
        orch_1 = runner_1._create_fresh_orchestrator("test_flow")
        orch_2 = runner_2._create_fresh_orchestrator("test_flow")
        
        # Verify orchestrators have isolated sessions
        assert orch_1.get_effective_bm().session_info.session_id == "session-1"
        assert orch_2.get_effective_bm().session_info.session_id == "session-2"
        
        # Verify they're different instances
        assert orch_1.get_effective_bm() is not orch_2.get_effective_bm()


class TestBackwardCompatibility:
    """Test suite for backward compatibility with global singleton pattern."""
    
    def test_flowrunner_without_session_bm(self, mock_flow_config):
        """Test that FlowRunner works without session BM (legacy mode)."""
        flow_runner = FlowRunner(flows={"test_flow": mock_flow_config}, mode="test")
        
        # Should have no session BM
        assert flow_runner.bm is None
        
        # get_effective_bm should try to use global singleton (will fail in test)
        with pytest.raises(Exception):  # get_bm() not initialized
            flow_runner.get_effective_bm()
    
    def test_orchestrator_without_session_bm(self):
        """Test that Orchestrator works without session BM (legacy mode)."""
        orchestrator = MockOrchestrator(name="test_orch")
        
        # Should have no session BM
        assert orchestrator._bm is None
        
        # get_effective_bm should try to use global singleton (will fail in test)
        with pytest.raises(Exception):  # get_bm() not initialized
            orchestrator.get_effective_bm()
    
    def test_agent_without_session_bm(self):
        """Test that Agent works without session BM (legacy mode)."""
        agent = MockAgent(agent_name="test", role="TEST")
        
        # Should have no session BM
        assert not hasattr(agent._config, "bm") or agent._config.bm is None
        
        # get_effective_bm should try to use global singleton (will fail in test)
        with pytest.raises(Exception):  # get_bm() not initialized
            agent.get_effective_bm()


if __name__ == "__main__":
    pytest.main([__file__])