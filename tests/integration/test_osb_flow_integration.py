"""
OSB Interactive Flow Integration Tests

Test suite for OSB (Oversight Board) interactive flow functionality,
focusing on flow initialization, agent integration, and vector store access.

Following test-driven development approach:
1. Write failing tests first
2. Implement minimal code to make tests pass
3. Validate integration with existing components
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from hydra import compose, initialize
from omegaconf import OmegaConf

from buttermilk._core.agent import AgentInput, AgentTrace
from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import FlowMessage, UIMessage
from buttermilk._core.orchestrator import Orchestrator
from buttermilk.agents.rag import RagAgent
from buttermilk.orchestrators.groupchat import AutogenOrchestrator
from buttermilk.runner.flowrunner import FlowRunContext, SessionStatus


class TestOSBFlowInitialization:
    """Test OSB flow initialization with vector store access."""

    @pytest.fixture
    def osb_flow_config(self):
        """Load OSB flow configuration from osb.yaml."""
        with initialize(version_base=None, config_path="../../conf"):
            cfg = compose(config_name="osb")
        return cfg

    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store for testing."""
        mock_store = MagicMock()
        mock_store.collection_name = "osb_vector"
        mock_store.ensure_cache_initialized = AsyncMock()
        return mock_store

    @pytest.fixture
    def mock_bm_instance(self, mock_vector_store):
        """Mock Buttermilk instance with storage access."""
        mock_bm = MagicMock()
        mock_bm.get_storage = MagicMock(return_value=mock_vector_store)
        return mock_bm

    @pytest.mark.anyio
    async def test_osb_flow_loads_configuration(self, osb_flow_config):
        """Test: OSB flow configuration loads successfully with all required components."""
        # This test should pass once osb.yaml is properly configured
        assert "osb" in osb_flow_config
        osb_config = osb_flow_config.osb
        
        # Verify core configuration elements
        assert "orchestrator" in osb_config
        assert osb_config.orchestrator == "buttermilk.orchestrators.groupchat.AutogenOrchestrator"
        assert "agents" in osb_config
        assert "observers" in osb_config
        assert "storage" in osb_config
        
        # Verify parameters for interactive flow support
        assert "parameters" in osb_config
        parameters = osb_config.parameters
        assert parameters.enable_multi_agent_synthesis is True
        assert parameters.enable_cross_validation is True

    @pytest.mark.anyio
    async def test_osb_agents_have_vector_store_access(self, osb_flow_config, mock_bm_instance):
        """
        FAILING TEST: OSB agents should initialize with vector store access.
        
        This test currently fails because:
        1. OSB flow configuration lacks WebSocket compatibility settings
        2. Vector store initialization not properly integrated
        3. Agent data configuration not accessible to EnhancedRagAgent
        """
        osb_config = osb_flow_config.osb
        
        # Mock the global BM instance
        with patch("buttermilk._core.dmrc.get_bm", return_value=mock_bm_instance):
            # Extract agent configurations (this will currently fail)
            agents_config = osb_config.agents
            
            # Test that each OSB agent type can access vector store
            expected_agents = ["researcher", "policy_analyst", "fact_checker", "explorer"]
            
            for agent_name in expected_agents:
                # This should work but currently fails due to missing data configuration
                assert agent_name in agents_config, f"Missing {agent_name} in OSB agent configuration"
                agent_cfg = agents_config[agent_name]
                
                # Verify agent has data configuration with vector store
                assert "data" in agent_cfg, f"Agent {agent_name} missing data configuration"
                assert "osb_vector" in agent_cfg.data, f"Agent {agent_name} missing osb_vector data source"
                
                # Test agent initialization with vector store access
                if agent_cfg.get("agent_obj") == "EnhancedRagAgent":
                    # This will fail due to missing vector store integration
                    config_dict = OmegaConf.to_container(agent_cfg, resolve=True)
                    agent = EnhancedRagAgent(**config_dict)
                    
                    # Verify agent can initialize search tools with vector store
                    await agent._initialize_search_tools()
                    assert agent._enhanced_search is not None
                    assert agent._enhanced_search.vectorstore.collection_name == "osb_vector"

    @pytest.mark.anyio
    async def test_osb_flow_supports_websocket_sessions(self, osb_flow_config):
        """
        FAILING TEST: OSB flow should support WebSocket-compatible session management.
        
        This test currently fails because:
        1. OSB configuration lacks session management parameters
        2. No WebSocket compatibility settings
        3. Missing session isolation configuration
        """
        osb_config = osb_flow_config.osb
        
        # Test for WebSocket session support parameters
        assert "session_management" in osb_config.parameters, "Missing session_management configuration"
        session_params = osb_config.parameters.session_management
        
        # Required for WebSocket compatibility
        assert "enable_websocket_sessions" in session_params
        assert session_params.enable_websocket_sessions is True
        assert "session_timeout" in session_params
        assert session_params.session_timeout >= 3600  # At least 1 hour
        assert "max_concurrent_sessions" in session_params
        assert session_params.max_concurrent_sessions >= 10  # Support multiple users
        
        # Required for session isolation
        assert "enable_session_isolation" in session_params
        assert session_params.enable_session_isolation is True

    @pytest.mark.anyio
    async def test_osb_orchestrator_initialization(self, osb_flow_config, mock_bm_instance):
        """
        FAILING TEST: OSB orchestrator should initialize with proper agent registration.
        
        This test currently fails because:
        1. Agent configurations not properly loaded
        2. Vector store integration not working
        3. Orchestrator initialization incomplete
        """
        osb_config = osb_flow_config.osb
        
        with patch("buttermilk._core.dmrc.get_bm", return_value=mock_bm_instance):
            # Test orchestrator initialization
            orchestrator_class_path = osb_config.orchestrator
            module_path, class_name = orchestrator_class_path.rsplit(".", 1)
            
            # This should work but may fail due to configuration issues
            import importlib
            module = importlib.import_module(module_path)
            orchestrator_cls = getattr(module, class_name)
            
            # Create orchestrator with OSB configuration
            # This will fail due to incomplete agent configuration
            config_dict = OmegaConf.to_container(osb_config, resolve=True)
            orchestrator = orchestrator_cls(**config_dict)
            
            # Test that orchestrator has all expected OSB agents
            expected_agents = ["researcher", "policy_analyst", "fact_checker", "explorer"]
            for agent_name in expected_agents:
                assert agent_name in orchestrator.agents, f"Orchestrator missing {agent_name} agent"

    @pytest.mark.anyio
    async def test_osb_session_context_creation(self):
        """
        FAILING TEST: OSB should support session context creation for WebSocket sessions.
        
        This test currently fails because:
        1. No OSB-specific session context implementation
        2. Missing session isolation features
        3. No WebSocket integration hooks
        """
        # Test session context creation for OSB flows
        session_id = "test-osb-session-12345"
        
        # This should create an OSB-specific session context
        # Currently fails because OSBFlowContext doesn't exist
        from buttermilk.runner.flowrunner import FlowRunContext
        
        session_context = FlowRunContext(
            session_id=session_id,
            flow_name="osb",
            status=SessionStatus.INITIALIZING
        )
        
        # Test OSB-specific session features
        assert session_context.flow_name == "osb"
        assert session_context.session_id == session_id
        
        # Test session isolation for OSB queries
        base_topic = "osb_query"
        isolated_topic = session_context.get_isolated_topic(base_topic)
        assert isolated_topic == f"{session_id}:{base_topic}"
        
        # Test that session can track OSB-specific resources
        # This should work but may need enhancements for OSB
        mock_websocket = MagicMock()
        session_context.add_websocket(mock_websocket)
        assert mock_websocket in session_context.resources.websockets


class TestOSBWebSocketIntegration:
    """Test OSB WebSocket query processing integration."""

    @pytest.mark.anyio
    async def test_osb_websocket_message_routing(self):
        """
        FAILING TEST: OSB WebSocket should route messages to appropriate agents.
        
        This test currently fails because:
        1. No OSB-specific WebSocket message handlers
        2. Missing message routing logic for OSB queries
        3. No integration with existing WebSocket infrastructure
        """
        # Test OSB-specific WebSocket message types
        osb_query_message = {
            "type": "osb_query",
            "query": "What are the policy implications of this content?",
            "session_id": "test-session",
            "context": {
                "case_number": "OSB-2025-001",
                "priority": "high"
            }
        }
        
        # This should route to OSB flow but currently fails
        # Missing: OSB-specific message handler in WebSocket API
        from buttermilk.api.flow import handle_websocket_message  # This function needs to exist
        
        # Mock WebSocket connection
        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock()
        
        # This call should work but currently fails due to missing OSB handler
        with pytest.raises(NotImplementedError, match="OSB WebSocket handler not implemented"):
            await handle_websocket_message(mock_websocket, osb_query_message)

    @pytest.mark.anyio
    async def test_osb_query_processing_flow(self):
        """
        FAILING TEST: OSB query should flow through all agents and return synthesized response.
        
        This test currently fails because:
        1. No end-to-end OSB query processing pipeline
        2. Missing agent coordination logic
        3. No response synthesis for multiple OSB agents
        """
        # Mock OSB query
        query = "Analyze this content for policy violations and provide recommendations"
        session_id = "test-osb-session"
        
        # Expected flow: Query → Researcher → Policy Analyst → Fact Checker → Explorer → Synthesis
        # This entire pipeline needs to be implemented
        
        expected_agents_order = ["researcher", "policy_analyst", "fact_checker", "explorer"]
        expected_response_structure = {
            "session_id": session_id,
            "query": query,
            "agent_responses": {},
            "synthesis": "",
            "confidence_score": 0.0,
            "recommendations": [],
            "case_metadata": {}
        }
        
        # This test documents the expected behavior but currently fails
        # Implementation needed in Phase 1
        with pytest.raises(NotImplementedError, match="OSB query processing pipeline not implemented"):
            result = await process_osb_query(query, session_id)
            
            # Validate response structure
            assert result["session_id"] == session_id
            assert result["query"] == query
            assert len(result["agent_responses"]) == len(expected_agents_order)
            assert all(agent in result["agent_responses"] for agent in expected_agents_order)


# Mock functions that need to be implemented in Phase 1
async def handle_websocket_message(websocket, message):
    """OSB WebSocket message handler - TO BE IMPLEMENTED."""
    if message.get("type") == "osb_query":
        raise NotImplementedError("OSB WebSocket handler not implemented")
    raise ValueError(f"Unknown message type: {message.get('type')}")


async def process_osb_query(query: str, session_id: str):
    """OSB query processing pipeline - TO BE IMPLEMENTED."""
    raise NotImplementedError("OSB query processing pipeline not implemented")