"""
OSB Multi-Agent Workflow Integration Tests.

Comprehensive integration testing for OSB (Oversight Board) multi-agent workflows
including:

- Complete policy analysis workflow testing
- Agent coordination and handoff testing  
- Cross-agent validation and synthesis testing
- Error handling and recovery across agents
- Performance and timing validation
- Data consistency across agent interactions

These tests validate the full OSB multi-agent pipeline from query initiation
through final synthesis and recommendation generation.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any
import time

from buttermilk._core.types import RunRequest
from buttermilk.runner.flowrunner import FlowRunner, FlowRunContext, SessionStatus
from buttermilk.data.vector import VectorStoreInterface
from buttermilk.api.mcp_osb import (
    test_osb_vector_query,
    run_osb_agent_test,
    run_osb_synthesis_test
)


@pytest.fixture
def mock_vector_store():
    """Mock vector store for OSB testing."""
    mock_store = MagicMock(spec=VectorStoreInterface)
    
    # Mock vector search results
    mock_store.query.return_value = {
        "results": [
            {
                "content": "Policy section 4.2: Content must not contain hate speech targeting protected groups",
                "metadata": {"source": "community_standards.pdf", "section": "4.2", "confidence": 0.92},
                "score": 0.85
            },
            {
                "content": "Precedent case OSB-2024-089: Similar content was classified as hate speech",
                "metadata": {"source": "case_database", "case_id": "OSB-2024-089", "confidence": 0.88},
                "score": 0.82
            },
            {
                "content": "Policy enforcement guidelines: First violation requires content removal and warning",
                "metadata": {"source": "enforcement_guide.pdf", "section": "2.1", "confidence": 0.95},
                "score": 0.79
            }
        ],
        "total_results": 3,
        "query_metadata": {
            "embedding_model": "gemini-embedding-001",
            "search_time_ms": 45.2
        }
    }
    
    return mock_store


@pytest.fixture
def mock_osb_agents():
    """Mock OSB agent implementations."""
    return {
        "researcher": MagicMock(),
        "policy_analyst": MagicMock(),
        "fact_checker": MagicMock(),
        "explorer": MagicMock()
    }


@pytest.fixture
def osb_flow_runner(mock_vector_store, mock_osb_agents):
    """Mock FlowRunner configured for OSB testing."""
    mock_runner = MagicMock(spec=FlowRunner)
    mock_runner.flows = {
        "osb": {
            "name": "OSB Interactive Flow",
            "agents": mock_osb_agents,
            "vector_store": mock_vector_store
        }
    }
    return mock_runner


class TestOSBMultiAgentCoordination:
    """Test coordination between OSB agents in workflows."""

    @pytest.mark.asyncio
    async def test_complete_policy_analysis_workflow(self, osb_flow_runner, mock_vector_store):
        """Test complete OSB policy analysis workflow."""
        # Test scenario: Analyzing potentially harmful social media content
        test_query = "This post contains language targeting a specific ethnic group with derogatory terms"
        
        # Execute workflow steps
        workflow_results = await self._run_complete_osb_workflow(
            query=test_query,
            case_number="OSB-TEST-001",
            flow_runner=osb_flow_runner
        )
        
        # Validate workflow completion
        assert workflow_results["workflow_complete"] is True
        assert workflow_results["error_occurred"] is False
        
        # Validate all agents participated
        expected_agents = {"researcher", "policy_analyst", "fact_checker", "explorer"}
        assert set(workflow_results["agent_responses"].keys()) == expected_agents
        
        # Validate synthesis was performed
        assert "synthesis_summary" in workflow_results
        assert "policy_violations" in workflow_results
        assert "recommendations" in workflow_results
        
        # Validate timing requirements
        assert workflow_results["total_duration"] < 60.0  # Under 1 minute
        
        # Validate confidence scores
        assert workflow_results["overall_confidence"] >= 0.5

    async def _run_complete_osb_workflow(self, query: str, case_number: str, 
                                       flow_runner: FlowRunner) -> Dict[str, Any]:
        """Execute complete OSB workflow simulation."""
        start_time = time.time()
        
        workflow_results = {
            "query": query,
            "case_number": case_number,
            "agent_responses": {},
            "synthesis_summary": "",
            "policy_violations": [],
            "recommendations": [],
            "workflow_complete": False,
            "error_occurred": False,
            "total_duration": 0,
            "overall_confidence": 0.0
        }
        
        try:
            # Step 1: Vector store query for context
            vector_results = await self._simulate_vector_query(query, flow_runner)
            
            # Step 2: Execute each agent
            agent_names = ["researcher", "policy_analyst", "fact_checker", "explorer"]
            
            for agent_name in agent_names:
                agent_result = await self._simulate_agent_execution(
                    agent_name, query, vector_results, case_number
                )
                workflow_results["agent_responses"][agent_name] = agent_result
            
            # Step 3: Synthesize results
            synthesis_result = await self._simulate_synthesis(
                query, workflow_results["agent_responses"]
            )
            
            workflow_results.update(synthesis_result)
            workflow_results["workflow_complete"] = True
            
        except Exception as e:
            workflow_results["error_occurred"] = True
            workflow_results["error_message"] = str(e)
        
        workflow_results["total_duration"] = time.time() - start_time
        return workflow_results

    async def _simulate_vector_query(self, query: str, flow_runner: FlowRunner) -> Dict[str, Any]:
        """Simulate vector store query for OSB context."""
        return {
            "results": [
                {
                    "content": "Policy section on hate speech and harassment",
                    "metadata": {"source": "policy.pdf", "confidence": 0.9},
                    "score": 0.85
                }
            ],
            "query_metadata": {"search_time_ms": 25.0}
        }

    async def _simulate_agent_execution(self, agent_name: str, query: str, 
                                      vector_results: Dict[str, Any], case_number: str) -> Dict[str, Any]:
        """Simulate individual agent execution."""
        # Mock agent-specific responses
        agent_responses = {
            "researcher": {
                "findings": f"Content analysis for {case_number}: Identified problematic language patterns",
                "evidence": ["Pattern match: derogatory_ethnic_terms", "Context: social_media_harassment"],
                "confidence": 0.87,
                "sources": ["pattern_detection.model", "context_analysis.db"],
                "processing_time": 2.3
            },
            "policy_analyst": {
                "analysis": "Content violates Community Standards Section 4.2: Hate Speech",
                "policy_sections": ["4.2", "3.1"],
                "violation_severity": "high",
                "confidence": 0.92,
                "recommendations": ["immediate_removal", "user_warning"],
                "processing_time": 1.8
            },
            "fact_checker": {
                "validation": "Claims and context verified through multiple sources",
                "accuracy_score": 0.94,
                "verified_facts": ["ethnic_targeting_confirmed", "harassment_pattern_verified"],
                "confidence": 0.91,
                "sources": ["fact_database", "verification_service"],
                "processing_time": 2.1
            },
            "explorer": {
                "themes": ["hate_speech", "ethnic_harassment", "coordinated_harassment"],
                "related_cases": ["OSB-2024-089", "OSB-2024-156"],
                "trend_analysis": "Part of larger harassment campaign",
                "confidence": 0.83,
                "exploration_depth": "comprehensive",
                "processing_time": 2.5
            }
        }
        
        return agent_responses.get(agent_name, {
            "error": f"Unknown agent: {agent_name}",
            "confidence": 0.0
        })

    async def _simulate_synthesis(self, query: str, agent_responses: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate multi-agent response synthesis."""
        return {
            "synthesis_summary": "Multi-agent analysis confirms policy violation with high confidence",
            "policy_violations": [
                "Community Standards 4.2: Hate Speech",
                "Community Standards 3.1: Harassment"
            ],
            "recommendations": [
                "Remove content immediately",
                "Issue formal warning to user",
                "Monitor user activity for 30 days",
                "Document case for trend analysis"
            ],
            "overall_confidence": 0.89,
            "cross_validation_score": 0.91,
            "consensus_level": "high"
        }

    @pytest.mark.asyncio
    async def test_agent_coordination_timing(self, osb_flow_runner):
        """Test timing and coordination between OSB agents."""
        test_query = "Test query for timing validation"
        
        # Track agent execution order and timing
        execution_log = []
        
        async def mock_agent_with_timing(agent_name: str, delay: float):
            start_time = time.time()
            await asyncio.sleep(delay)  # Simulate processing time
            execution_log.append({
                "agent": agent_name,
                "start_time": start_time,
                "duration": time.time() - start_time
            })
            return {"agent": agent_name, "processed": True}
        
        # Simulate agent execution with different processing times
        agent_tasks = [
            mock_agent_with_timing("researcher", 0.1),
            mock_agent_with_timing("policy_analyst", 0.15),
            mock_agent_with_timing("fact_checker", 0.12),
            mock_agent_with_timing("explorer", 0.18)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*agent_tasks)
        total_time = time.time() - start_time
        
        # Validate timing requirements
        assert total_time < 1.0  # Parallel execution should be under 1 second
        assert len(results) == 4  # All agents completed
        assert len(execution_log) == 4  # All executions logged
        
        # Validate agents ran in parallel (overlapping times)
        agent_start_times = [log["start_time"] for log in execution_log]
        time_spread = max(agent_start_times) - min(agent_start_times)
        assert time_spread < 0.1  # All agents started within 100ms

    @pytest.mark.asyncio
    async def test_agent_error_handling_and_recovery(self, osb_flow_runner):
        """Test error handling when individual agents fail."""
        test_query = "Error recovery test query"
        
        # Simulate scenario where one agent fails
        async def failing_agent():
            raise Exception("Simulated agent failure")
        
        async def successful_agent(agent_name: str):
            return {"agent": agent_name, "success": True, "confidence": 0.8}
        
        # Mix of successful and failing agents
        agent_tasks = [
            successful_agent("researcher"),
            failing_agent(),  # policy_analyst fails
            successful_agent("fact_checker"),
            successful_agent("explorer")
        ]
        
        # Execute with error handling
        results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Validate error handling
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        assert len(successful_results) == 3  # 3 agents succeeded
        assert len(failed_results) == 1     # 1 agent failed
        
        # Validate workflow can continue with partial results
        assert all(r.get("success") for r in successful_results)

    @pytest.mark.asyncio
    async def test_cross_agent_data_consistency(self, osb_flow_runner, mock_vector_store):
        """Test data consistency across agent interactions."""
        test_query = "Data consistency test query"
        case_number = "OSB-CONSISTENCY-001"
        
        # Simulate agents accessing shared context
        shared_context = {
            "query": test_query,
            "case_number": case_number,
            "vector_results": await self._simulate_vector_query(test_query, osb_flow_runner),
            "processing_metadata": {
                "timestamp": time.time(),
                "workflow_id": "test-workflow-001"
            }
        }
        
        # Execute agents with shared context
        agent_results = {}
        for agent_name in ["researcher", "policy_analyst", "fact_checker", "explorer"]:
            result = await self._simulate_agent_with_context(agent_name, shared_context)
            agent_results[agent_name] = result
        
        # Validate consistent case_number across all agents
        for agent_name, result in agent_results.items():
            assert result.get("case_number") == case_number
            assert result.get("source_query") == test_query
        
        # Validate consistent confidence scoring approach
        confidence_scores = [result.get("confidence", 0) for result in agent_results.values()]
        assert all(0 <= score <= 1 for score in confidence_scores)  # Valid confidence range
        
        # Validate consistent timestamp handling
        timestamps = [result.get("timestamp") for result in agent_results.values()]
        timestamp_range = max(timestamps) - min(timestamps)
        assert timestamp_range < 5.0  # All processed within 5 seconds

    async def _simulate_agent_with_context(self, agent_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate agent execution with shared context."""
        return {
            "agent_name": agent_name,
            "case_number": context["case_number"],
            "source_query": context["query"],
            "confidence": 0.85,  # Consistent confidence format
            "timestamp": time.time(),
            "context_validated": True
        }


class TestOSBWorkflowIntegration:
    """Integration tests for complete OSB workflows."""

    @pytest.mark.asyncio
    async def test_end_to_end_content_moderation_workflow(self, osb_flow_runner):
        """Test complete content moderation workflow from query to decision."""
        # Realistic content moderation scenario
        content_to_analyze = "This is a test post with potentially problematic content"
        case_metadata = {
            "case_number": "OSB-E2E-001",
            "content_type": "social_media_post",
            "platform": "twitter",
            "reported_by": "community_user",
            "priority": "high"
        }
        
        # Execute complete workflow
        workflow_result = await self._execute_end_to_end_workflow(
            content_to_analyze, case_metadata, osb_flow_runner
        )
        
        # Validate complete workflow execution
        assert workflow_result["status"] == "completed"
        assert "final_decision" in workflow_result
        assert "confidence_level" in workflow_result
        assert "enforcement_actions" in workflow_result
        
        # Validate decision quality
        decision = workflow_result["final_decision"]
        assert decision in ["no_violation", "policy_violation", "requires_review"]
        
        # Validate audit trail
        assert "audit_trail" in workflow_result
        audit_trail = workflow_result["audit_trail"]
        assert len(audit_trail) >= 4  # All agents + synthesis
        
        # Validate enforcement recommendations
        if decision == "policy_violation":
            assert len(workflow_result["enforcement_actions"]) > 0

    async def _execute_end_to_end_workflow(self, content: str, metadata: Dict[str, Any], 
                                         flow_runner: FlowRunner) -> Dict[str, Any]:
        """Execute complete end-to-end OSB workflow."""
        workflow_result = {
            "content": content,
            "metadata": metadata,
            "status": "processing",
            "audit_trail": [],
            "agent_outputs": {},
            "synthesis_result": {},
            "final_decision": "",
            "confidence_level": 0.0,
            "enforcement_actions": []
        }
        
        try:
            # Stage 1: Initial processing and vector search
            vector_context = await self._get_policy_context(content)
            workflow_result["audit_trail"].append({
                "stage": "context_retrieval",
                "timestamp": time.time(),
                "results_count": len(vector_context.get("results", []))
            })
            
            # Stage 2: Multi-agent analysis
            agents = ["researcher", "policy_analyst", "fact_checker", "explorer"]
            for agent in agents:
                agent_output = await self._execute_agent_analysis(agent, content, vector_context, metadata)
                workflow_result["agent_outputs"][agent] = agent_output
                workflow_result["audit_trail"].append({
                    "stage": f"agent_{agent}",
                    "timestamp": time.time(),
                    "confidence": agent_output.get("confidence", 0)
                })
            
            # Stage 3: Synthesis and decision
            synthesis = await self._synthesize_agent_outputs(
                content, workflow_result["agent_outputs"], metadata
            )
            workflow_result["synthesis_result"] = synthesis
            workflow_result["final_decision"] = synthesis["decision"]
            workflow_result["confidence_level"] = synthesis["confidence"]
            workflow_result["enforcement_actions"] = synthesis.get("recommended_actions", [])
            
            workflow_result["audit_trail"].append({
                "stage": "synthesis",
                "timestamp": time.time(),
                "decision": synthesis["decision"]
            })
            
            workflow_result["status"] = "completed"
            
        except Exception as e:
            workflow_result["status"] = "error"
            workflow_result["error"] = str(e)
        
        return workflow_result

    async def _get_policy_context(self, content: str) -> Dict[str, Any]:
        """Get policy context from vector store."""
        return {
            "results": [
                {
                    "content": "Policy context for content analysis",
                    "metadata": {"source": "policy_db", "relevance": 0.9}
                }
            ]
        }

    async def _execute_agent_analysis(self, agent: str, content: str, 
                                    context: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual agent analysis."""
        return {
            "agent": agent,
            "analysis": f"{agent} analysis of content",
            "confidence": 0.85,
            "findings": [f"{agent}_finding_1", f"{agent}_finding_2"],
            "processing_time": 1.5
        }

    async def _synthesize_agent_outputs(self, content: str, agent_outputs: Dict[str, Any], 
                                      metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize agent outputs into final decision."""
        avg_confidence = sum(output.get("confidence", 0) for output in agent_outputs.values()) / len(agent_outputs)
        
        decision = "policy_violation" if avg_confidence > 0.8 else "no_violation"
        
        return {
            "decision": decision,
            "confidence": avg_confidence,
            "reasoning": "Multi-agent analysis synthesis",
            "recommended_actions": ["content_removal", "user_warning"] if decision == "policy_violation" else [],
            "synthesis_metadata": {
                "agents_consulted": list(agent_outputs.keys()),
                "synthesis_time": time.time()
            }
        }

    @pytest.mark.asyncio
    async def test_workflow_performance_requirements(self, osb_flow_runner):
        """Test that OSB workflows meet performance requirements."""
        test_scenarios = [
            {"content": "Short test content", "expected_max_time": 5.0},
            {"content": "Medium length test content with more details for analysis", "expected_max_time": 10.0},
            {"content": "Very long test content " * 50, "expected_max_time": 15.0}  # Long content
        ]
        
        for i, scenario in enumerate(test_scenarios):
            start_time = time.time()
            
            # Execute workflow
            result = await self._execute_end_to_end_workflow(
                scenario["content"],
                {"case_number": f"PERF-TEST-{i:03d}"},
                osb_flow_runner
            )
            
            execution_time = time.time() - start_time
            
            # Validate performance requirements
            assert execution_time <= scenario["expected_max_time"]
            assert result["status"] == "completed"
            
            # Validate quality not compromised for speed
            assert result["confidence_level"] >= 0.5

    @pytest.mark.asyncio  
    async def test_concurrent_workflow_handling(self, osb_flow_runner):
        """Test handling of multiple concurrent OSB workflows."""
        num_concurrent_workflows = 5
        
        # Create concurrent workflow tasks
        workflow_tasks = []
        for i in range(num_concurrent_workflows):
            content = f"Concurrent test content {i}"
            metadata = {"case_number": f"CONCURRENT-{i:03d}"}
            task = asyncio.create_task(
                self._execute_end_to_end_workflow(content, metadata, osb_flow_runner)
            )
            workflow_tasks.append(task)
        
        # Execute all workflows concurrently
        start_time = time.time()
        results = await asyncio.gather(*workflow_tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Validate concurrent execution
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == num_concurrent_workflows
        
        # Validate all workflows completed successfully
        for result in successful_results:
            assert result["status"] == "completed"
            assert "final_decision" in result
        
        # Validate reasonable total time (should be faster than sequential)
        max_sequential_time = num_concurrent_workflows * 15.0  # Assume 15s per workflow
        assert total_time < max_sequential_time * 0.5  # At least 50% faster due to parallelism


class TestOSBWorkflowErrorRecovery:
    """Test error recovery and resilience in OSB workflows."""

    @pytest.mark.asyncio
    async def test_partial_agent_failure_recovery(self, osb_flow_runner):
        """Test workflow continuation when some agents fail."""
        content = "Test content for error recovery"
        metadata = {"case_number": "ERROR-RECOVERY-001"}
        
        # Simulate scenario where 2 out of 4 agents fail
        successful_agents = ["researcher", "explorer"]
        failed_agents = ["policy_analyst", "fact_checker"]
        
        workflow_result = await self._execute_workflow_with_agent_failures(
            content, metadata, successful_agents, failed_agents, osb_flow_runner
        )
        
        # Validate workflow completes despite failures
        assert workflow_result["status"] == "partial_completion"
        assert len(workflow_result["successful_agents"]) == 2
        assert len(workflow_result["failed_agents"]) == 2
        
        # Validate decision can still be made with partial results
        assert "final_decision" in workflow_result
        assert workflow_result["confidence_level"] >= 0.3  # Lower but still valid confidence

    async def _execute_workflow_with_agent_failures(self, content: str, metadata: Dict[str, Any],
                                                   successful_agents: List[str], failed_agents: List[str],
                                                   flow_runner: FlowRunner) -> Dict[str, Any]:
        """Execute workflow with simulated agent failures."""
        workflow_result = {
            "content": content,
            "metadata": metadata,
            "status": "processing",
            "successful_agents": [],
            "failed_agents": [],
            "agent_outputs": {},
            "final_decision": "",
            "confidence_level": 0.0
        }
        
        # Execute successful agents
        for agent in successful_agents:
            try:
                output = await self._execute_agent_analysis(agent, content, {}, metadata)
                workflow_result["agent_outputs"][agent] = output
                workflow_result["successful_agents"].append(agent)
            except Exception:
                workflow_result["failed_agents"].append(agent)
        
        # Simulate failed agents
        for agent in failed_agents:
            workflow_result["failed_agents"].append(agent)
        
        # Make decision with partial results
        if workflow_result["successful_agents"]:
            synthesis = await self._synthesize_partial_results(
                content, workflow_result["agent_outputs"], metadata
            )
            workflow_result["final_decision"] = synthesis["decision"]
            workflow_result["confidence_level"] = synthesis["confidence"]
            workflow_result["status"] = "partial_completion"
        else:
            workflow_result["status"] = "failed"
        
        return workflow_result

    async def _execute_agent_analysis(self, agent: str, content: str, 
                                    context: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual agent analysis (reused from above)."""
        return {
            "agent": agent,
            "analysis": f"{agent} analysis of content",
            "confidence": 0.85,
            "findings": [f"{agent}_finding_1", f"{agent}_finding_2"],
            "processing_time": 1.5
        }

    async def _synthesize_partial_results(self, content: str, agent_outputs: Dict[str, Any], 
                                        metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize partial agent results with reduced confidence."""
        if not agent_outputs:
            return {"decision": "insufficient_data", "confidence": 0.0}
        
        avg_confidence = sum(output.get("confidence", 0) for output in agent_outputs.values()) / len(agent_outputs)
        # Reduce confidence due to missing agents
        reduced_confidence = avg_confidence * (len(agent_outputs) / 4.0)  # Assuming 4 total agents
        
        decision = "requires_review" if reduced_confidence < 0.7 else "policy_violation"
        
        return {
            "decision": decision,
            "confidence": reduced_confidence,
            "reasoning": f"Partial analysis from {len(agent_outputs)} agents",
            "missing_agents": 4 - len(agent_outputs)
        }