"""Integration tests for osb, trans, and tox flows with the new tool definition system."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Any
import os

from buttermilk._core import AgentInput, StepRequest
from buttermilk._core.agent import Agent, ManagerMessage
from buttermilk._core.contract import AgentOutput, AgentTrace, ConductorRequest
from buttermilk._core.constants import END, MANAGER
from buttermilk._core.tool_definition import AgentToolDefinition, UnifiedRequest
from buttermilk._core.mcp_decorators import tool, MCPRoute
from buttermilk.agents.flowcontrol.structured_llmhost import StructuredLLMHostAgent
from buttermilk.agents.flowcontrol.host import HostAgent
from buttermilk.orchestrators.groupchat import AutogenOrchestrator

# Use anyio for async tests
pytestmark = pytest.mark.anyio


@pytest.fixture
def mock_bm():
    """Mock the BM singleton."""
    # Create mock BM instance
    mock_bm = Mock()
    mock_bm.llms = Mock()
    mock_bm.llms.get_autogen_chat_client = Mock(return_value=Mock())
    mock_bm.databases = {}
    mock_bm.storage = {}
    
    # Mock weave
    mock_bm.weave = Mock()
    mock_bm.weave.init_trace = Mock()
    
    # Use MagicMock to prevent AttributeError
    with patch('buttermilk.buttermilk', mock_bm):
        yield mock_bm


class TestOSBFlowIntegration:
    """Test OSB flow with structured tool definitions."""
    
    async def test_osb_flow_with_structured_host(self, mock_bm):
        """Test OSB flow with StructuredLLMHostAgent replacing the sequencer."""
        # Create mock agents with tool definitions
        class MockResearcherAgent(Agent):
            """Mock researcher agent for OSB."""
            
            async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
                return AgentOutput(
                    agent_id=self.agent_id,
                    outputs={"research": "OSB research results"}
                )
            
            @tool
            @MCPRoute("/search_osb")
            async def search_osb_database(self, query: str, case_type: str = "all") -> dict[str, Any]:
                """Search OSB database for cases."""
                return {
                    "query": query,
                    "case_type": case_type,
                    "results": [
                        {"case_id": "OSB-001", "title": "Sample Case 1"},
                        {"case_id": "OSB-002", "title": "Sample Case 2"}
                    ]
                }
        
        class MockPolicyAnalystAgent(Agent):
            """Mock policy analyst for OSB."""
            
            async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
                return AgentOutput(
                    agent_id=self.agent_id,
                    outputs={"analysis": "Policy analysis complete"}
                )
            
            @tool
            def analyze_policy(self, case_id: str, policy_area: str) -> dict[str, Any]:
                """Analyze policy implications of a case."""
                return {
                    "case_id": case_id,
                    "policy_area": policy_area,
                    "recommendations": ["Recommendation 1", "Recommendation 2"]
                }
        
        # Create agents
        researcher = MockResearcherAgent(
            agent_name="researcher",
            model_name="test",
            role="RESEARCHER"
        )
        
        policy_analyst = MockPolicyAnalystAgent(
            agent_name="policy_analyst",
            model_name="test",
            role="POLICY_ANALYST"
        )
        
        # Create structured host to replace sequencer
        structured_host = StructuredLLMHostAgent(
            agent_name="host",
            model_name="test-model",
            role="HOST",
            parameters={
                "model": "test-model",
                "template": "host_structured_tools",
                "human_in_loop": False
            }
        )
        
        # Mock the orchestrator
        mock_orchestrator = AsyncMock(spec=AutogenOrchestrator)
        
        # Set up participants
        agents = {
            "RESEARCHER": researcher,
            "POLICY_ANALYST": policy_analyst
        }
        observers = {
            "HOST": structured_host
        }
        
        # Initialize host with participants
        structured_host._participants = agents
        structured_host.tools = []
        structured_host.callback_to_groupchat = AsyncMock()
        
        await structured_host._initialize(callback_to_groupchat=structured_host.callback_to_groupchat)
        
        # Verify tools were registered
        tool_names = [t.name for t in structured_host._tools_list]
        assert "researcher.search_osb_database" in tool_names
        assert "policy_analyst.analyze_policy" in tool_names
        
        # Test tool invocation through structured host
        search_tool = next(t for t in structured_host._tools_list if t.name == "researcher.search_osb_database")
        await search_tool._func(inputs={"query": "discrimination cases", "case_type": "employment"})
        
        # Verify the right StepRequest was sent
        structured_host.callback_to_groupchat.assert_called()
        step_request = structured_host.callback_to_groupchat.call_args[0][0]
        assert isinstance(step_request, StepRequest)
        assert step_request.role == "RESEARCHER"
        assert step_request.inputs["tool"] == "search_osb_database"
        assert step_request.inputs["tool_inputs"]["query"] == "discrimination cases"


class TestTransFlowIntegration:
    """Test trans flow with structured tool definitions."""
    
    async def test_trans_flow_with_structured_tools(self, mock_bm):
        """Test trans flow with agents using tool definitions."""
        # Create mock judge agent with tools
        class MockJudgeAgent(Agent):
            """Mock judge agent for trans flow."""
            
            async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
                return AgentOutput(
                    agent_id=self.agent_id,
                    outputs={"judgment": "Content assessed"}
                )
            
            @tool
            @MCPRoute("/assess_content")
            def assess_journalism_quality(
                self, 
                content: str, 
                criteria: list[str]
            ) -> dict[str, Any]:
                """Assess journalism quality against criteria."""
                return {
                    "content_snippet": content[:100],
                    "criteria_applied": criteria,
                    "scores": {
                        criterion: 0.8 + (i * 0.05) 
                        for i, criterion in enumerate(criteria)
                    },
                    "overall_quality": "high"
                }
        
        class MockSynthAgent(Agent):
            """Mock synthesis agent."""
            
            async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
                return AgentOutput(
                    agent_id=self.agent_id,
                    outputs={"synthesis": "Synthesized results"}
                )
            
            @tool
            def synthesize_assessments(
                self,
                assessments: list[dict[str, Any]]
            ) -> dict[str, Any]:
                """Synthesize multiple assessments."""
                return {
                    "assessment_count": len(assessments),
                    "summary": "Multiple perspectives synthesized",
                    "consensus_areas": ["Area 1", "Area 2"],
                    "divergence_areas": ["Area 3"]
                }
        
        # Create agents
        judge = MockJudgeAgent(
            agent_name="judge",
            model_name="test",
            role="JUDGE"
        )
        
        synth = MockSynthAgent(
            agent_name="synth",
            model_name="test",
            role="SYNTH"
        )
        
        # Test UnifiedRequest handling
        assess_request = UnifiedRequest(
            target="judge.assess_journalism_quality",
            inputs={
                "content": "Article about trans issues in sports...",
                "criteria": ["accuracy", "fairness", "context"]
            }
        )
        
        result = await judge.handle_unified_request(assess_request)
        assert "scores" in result
        assert result["overall_quality"] == "high"
        assert len(result["criteria_applied"]) == 3
        
        # Test synthesis
        synth_request = UnifiedRequest(
            target="synth.synthesize_assessments",
            inputs={
                "assessments": [result, result]  # Use same result twice for test
            }
        )
        
        synth_result = await synth.handle_unified_request(synth_request)
        assert synth_result["assessment_count"] == 2
        assert "consensus_areas" in synth_result


class TestToxFlowIntegration:
    """Test tox flow with structured tool definitions."""
    
    async def test_tox_flow_tool_coordination(self, mock_bm):
        """Test tox flow with tool-based agent coordination."""
        # Create scorer agent with tools
        class MockScorerAgent(Agent):
            """Mock scorer agent for tox flow."""
            
            async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
                return AgentOutput(
                    agent_id=self.agent_id,
                    outputs={"scores": "Toxicity scored"}
                )
            
            @tool
            @MCPRoute("/score_toxicity", permissions=["analyze"])
            async def score_content_toxicity(
                self,
                content: str,
                criteria_type: str = "standard"
            ) -> dict[str, Any]:
                """Score content for toxicity."""
                await asyncio.sleep(0.1)  # Simulate async work
                
                return {
                    "content_hash": hash(content),
                    "criteria_type": criteria_type,
                    "toxicity_scores": {
                        "hate_speech": 0.1,
                        "harassment": 0.2,
                        "explicit_content": 0.05
                    },
                    "overall_toxicity": 0.15,
                    "confidence": 0.95
                }
            
            @tool
            def get_scoring_thresholds(self) -> dict[str, float]:
                """Get current toxicity scoring thresholds."""
                return {
                    "hate_speech": 0.7,
                    "harassment": 0.6,
                    "explicit_content": 0.8,
                    "overall": 0.65
                }
        
        # Create spy observer with monitoring tools
        class MockSpyAgent(Agent):
            """Mock spy observer for monitoring."""
            
            async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
                return AgentOutput(
                    agent_id=self.agent_id,
                    outputs={"observation": "Monitoring active"}
                )
            
            @tool
            def track_assessment_progress(
                self,
                total_items: int,
                completed_items: int
            ) -> dict[str, Any]:
                """Track progress of toxicity assessments."""
                return {
                    "total": total_items,
                    "completed": completed_items,
                    "progress_percentage": (completed_items / total_items * 100) if total_items > 0 else 0,
                    "estimated_time_remaining": (total_items - completed_items) * 2  # 2 seconds per item
                }
        
        # Create agents
        scorer = MockScorerAgent(
            agent_name="scorer",
            model_name="test",
            role="SCORER"
        )
        
        spy = MockSpyAgent(
            agent_name="spy",
            model_name="test",
            role="SPY"
        )
        
        # Test concurrent tool execution
        test_contents = [
            "This is a normal comment",
            "This might contain mild negativity",
            "Another test content"
        ]
        
        # Create concurrent scoring requests
        tasks = []
        for content in test_contents:
            req = UnifiedRequest(
                target="scorer.score_content_toxicity",
                inputs={
                    "content": content,
                    "criteria_type": "standard"
                }
            )
            task = scorer.handle_unified_request(req)
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        # Verify all completed
        assert len(results) == 3
        for result in results:
            assert "toxicity_scores" in result
            assert "overall_toxicity" in result
            assert result["confidence"] > 0.9
        
        # Test progress tracking
        progress_req = UnifiedRequest(
            target="spy.track_assessment_progress",
            inputs={
                "total_items": len(test_contents),
                "completed_items": len(results)
            }
        )
        
        progress = await spy.handle_unified_request(progress_req)
        assert progress["progress_percentage"] == 100.0
        assert progress["estimated_time_remaining"] == 0


class TestFlowMigration:
    """Test migration of existing flows to use structured tool definitions."""
    
    async def test_host_agent_compatibility(self):
        """Test that HostAgent can work alongside StructuredLLMHostAgent."""
        # Create a basic HostAgent (existing)
        basic_host = HostAgent(
            agent_name="basic_host",
            model_name="test",
            role="HOST"
        )
        
        # Create a StructuredLLMHostAgent (new)
        structured_host = StructuredLLMHostAgent(
            agent_name="structured_host",
            model_name="test-model",
            role="HOST",
            parameters={"model": "test-model", "template": "host_structured_tools"}
        )
        
        # Both should have compatible interfaces
        assert hasattr(basic_host, "_sequence")
        assert hasattr(structured_host, "_sequence")
        assert hasattr(basic_host, "_listen")
        assert hasattr(structured_host, "_listen")
        
        # Both should handle ConductorRequest
        conductor_req = ConductorRequest(
            participants={"AGENT1": "Test agent"}
        )
        
        # Basic host should work as before
        basic_host.callback_to_groupchat = AsyncMock()
        basic_host._participants = {"AGENT1": Mock()}
        
        # Structured host should also handle it
        structured_host.callback_to_groupchat = AsyncMock()
        structured_host._participants = {"AGENT1": Mock()}
        structured_host.tools = []
    
    async def test_gradual_migration_path(self):
        """Test that flows can gradually migrate agents to use tool definitions."""
        # Create an agent without tools (existing style)
        class LegacyAgent(Agent):
            async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
                return AgentOutput(
                    agent_id=self.agent_id,
                    outputs={"result": "Legacy processing"}
                )
        
        # Create an agent with tools (new style)
        class ModernAgent(Agent):
            async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
                return AgentOutput(
                    agent_id=self.agent_id,
                    outputs={"result": "Modern processing"}
                )
            
            @tool
            def analyze(self, data: str) -> dict[str, Any]:
                """Modern tool-based analysis."""
                return {"analyzed": data, "method": "tool"}
        
        legacy = LegacyAgent(agent_name="legacy", model_name="test", role="LEGACY")
        modern = ModernAgent(agent_name="modern", model_name="test", role="MODERN")
        
        # Both should work with get_tool_definitions
        legacy_tools = legacy.get_tool_definitions()
        modern_tools = modern.get_tool_definitions()
        
        assert len(legacy_tools) == 0  # No tools
        assert len(modern_tools) == 1  # Has analyze tool
        assert modern_tools[0].name == "analyze"
        
        # Structured host should handle both
        host = StructuredLLMHostAgent(
            agent_name="host",
            model_name="test-model",
            role="HOST",
            parameters={"model": "test-model"}
        )
        host._participants = {
            "LEGACY": legacy,
            "MODERN": modern
        }
        host.tools = []
        host.callback_to_groupchat = AsyncMock()
        
        await host._initialize(callback_to_groupchat=host.callback_to_groupchat)
        
        # Should have created default tool for legacy and actual tool for modern
        tool_names = [t.name for t in host._tools_list]
        assert "legacy.call_legacy" in tool_names  # Default tool created
        assert "modern.analyze" in tool_names  # Actual tool registered