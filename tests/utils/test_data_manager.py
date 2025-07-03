"""
Test Data Management for Flow-Agnostic Testing.

This module provides comprehensive test data management for any flow
configuration, including:

- Mock response generation for any agent type
- Test scenario data for different flow configurations
- Performance baseline data management
- Test fixture creation and cleanup
- Realistic mock data for vector stores and agents

Designed to support testing of any YAML-configured flow without
hardcoded dependencies on specific agent types or flow structures.
"""

import json
import time
import uuid
from typing import Dict, Any, List, Optional, Generator, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import random


@dataclass
class AgentResponse:
    """Standard agent response structure for testing."""
    agent_name: str
    confidence: float
    response_data: Dict[str, Any]
    sources: List[str]
    processing_time: float
    metadata: Dict[str, Any]


@dataclass
class VectorSearchResult:
    """Standard vector search result for testing."""
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str


@dataclass
class WorkflowResult:
    """Complete workflow result for testing."""
    flow_name: str
    query: str
    agent_responses: Dict[str, AgentResponse]
    synthesis_result: Optional[Dict[str, Any]]
    total_time: float
    success: bool
    error_message: Optional[str]


class TestDataGenerator:
    """Generates realistic test data for any flow configuration."""
    
    def __init__(self, seed: int = 42):
        """Initialize with optional seed for reproducible test data."""
        random.seed(seed)
        self.agent_response_templates = self._load_agent_response_templates()
        self.vector_content_templates = self._load_vector_content_templates()
        self.query_templates = self._load_query_templates()
    
    def generate_agent_response(self, agent_name: str, query: str, 
                               flow_context: str = "") -> AgentResponse:
        """Generate realistic agent response for any agent type."""
        # Base confidence varies by query complexity
        base_confidence = 0.7 + (len(query) % 30) / 100  # 0.7-0.99 range
        
        # Simulate processing time based on agent type and query
        processing_time = self._estimate_processing_time(agent_name, query)
        
        # Generate response data based on agent patterns
        response_data = self._generate_response_data(agent_name, query, flow_context)
        
        # Generate realistic sources
        sources = self._generate_sources(agent_name, flow_context)
        
        # Create metadata
        metadata = {
            "agent_type": agent_name,
            "query_hash": hash(query) % 10000,
            "flow_context": flow_context,
            "generation_time": time.time(),
            "confidence_factors": self._get_confidence_factors(agent_name, query)
        }
        
        return AgentResponse(
            agent_name=agent_name,
            confidence=min(base_confidence, 1.0),
            response_data=response_data,
            sources=sources,
            processing_time=processing_time,
            metadata=metadata
        )
    
    def generate_vector_search_results(self, query: str, flow_context: str = "",
                                     max_results: int = 5) -> List[VectorSearchResult]:
        """Generate realistic vector search results for any flow."""
        results = []
        
        for i in range(random.randint(1, max_results)):
            # Generate content relevant to query and flow
            content = self._generate_vector_content(query, flow_context, i)
            
            # Generate metadata
            metadata = {
                "source": f"{flow_context}_doc_{i+1}.pdf" if flow_context else f"doc_{i+1}.pdf",
                "section": f"{random.randint(1,10)}.{random.randint(1,5)}",
                "last_updated": "2025-01-15T10:30:00Z",
                "content_type": "policy" if "policy" in query.lower() else "general",
                "relevance_score": 0.9 - (i * 0.1)
            }
            
            # Score decreases with result ranking
            score = 0.95 - (i * 0.1) + (random.random() * 0.05)
            
            # Source varies by flow context
            source = f"{flow_context}_{random.choice(['policy', 'guideline', 'case', 'document'])}_{i+1}"
            
            results.append(VectorSearchResult(
                content=content,
                metadata=metadata,
                score=max(score, 0.1),
                source=source
            ))
        
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def generate_workflow_result(self, flow_name: str, query: str, 
                                agent_names: List[str]) -> WorkflowResult:
        """Generate complete workflow result for testing."""
        start_time = time.time()
        
        # Generate responses for each agent
        agent_responses = {}
        for agent_name in agent_names:
            agent_responses[agent_name] = self.generate_agent_response(
                agent_name, query, flow_name
            )
        
        # Generate synthesis result if multiple agents
        synthesis_result = None
        if len(agent_names) > 1:
            synthesis_result = self._generate_synthesis_result(
                query, agent_responses, flow_name
            )
        
        # Calculate total time
        total_time = time.time() - start_time + sum(
            resp.processing_time for resp in agent_responses.values()
        )
        
        return WorkflowResult(
            flow_name=flow_name,
            query=query,
            agent_responses=agent_responses,
            synthesis_result=synthesis_result,
            total_time=total_time,
            success=True,
            error_message=None
        )
    
    def _load_agent_response_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load response templates for different agent types."""
        return {
            "researcher": {
                "response_keys": ["findings", "evidence", "analysis"],
                "typical_confidence": 0.85,
                "processing_time_range": (1.0, 3.0),
                "source_types": ["academic", "research", "database"]
            },
            "policy_analyst": {
                "response_keys": ["policy_analysis", "violations", "recommendations"],
                "typical_confidence": 0.90,
                "processing_time_range": (1.5, 2.5),
                "source_types": ["policy", "guideline", "regulation"]
            },
            "fact_checker": {
                "response_keys": ["verification", "accuracy_score", "sources"],
                "typical_confidence": 0.88,
                "processing_time_range": (1.2, 2.8),
                "source_types": ["verification", "database", "reference"]
            },
            "explorer": {
                "response_keys": ["themes", "patterns", "connections"],
                "typical_confidence": 0.82,
                "processing_time_range": (2.0, 4.0),
                "source_types": ["exploration", "pattern", "network"]
            },
            "classifier": {
                "response_keys": ["classification", "confidence", "category"],
                "typical_confidence": 0.93,
                "processing_time_range": (0.5, 1.5),
                "source_types": ["model", "training_data", "classification"]
            },
            "reviewer": {
                "response_keys": ["review", "decision", "reasoning"],
                "typical_confidence": 0.87,
                "processing_time_range": (1.0, 2.0),
                "source_types": ["review", "criteria", "precedent"]
            },
            "analyst": {
                "response_keys": ["analysis", "insights", "trends"],
                "typical_confidence": 0.86,
                "processing_time_range": (2.0, 3.5),
                "source_types": ["analytical", "statistical", "trend"]
            },
            "synthesizer": {
                "response_keys": ["synthesis", "summary", "conclusions"],
                "typical_confidence": 0.84,
                "processing_time_range": (1.5, 3.0),
                "source_types": ["synthesis", "compilation", "summary"]
            }
        }
    
    def _load_vector_content_templates(self) -> Dict[str, List[str]]:
        """Load content templates for vector search results."""
        return {
            "policy": [
                "Section {section}: Content must comply with community standards regarding {topic}",
                "Policy {number}: Users are prohibited from {action} that may {consequence}",
                "Guideline {ref}: Content should not contain {prohibited_content} directed at {target}",
                "Standard {code}: Acceptable content includes {allowed_content} but excludes {excluded_content}"
            ],
            "precedent": [
                "Case {case_id}: Similar content was {action} based on {reasoning}",
                "Precedent {ref}: Previous decisions indicate {decision} for {content_type}",
                "Historical case: Content with {characteristics} resulted in {outcome}",
                "Reference decision: {authority} determined {finding} for comparable content"
            ],
            "research": [
                "Study findings: Research indicates {finding} related to {topic}",
                "Analysis shows: Data suggests {correlation} between {factor1} and {factor2}",
                "Research conclusion: Evidence supports {hypothesis} regarding {subject}",
                "Academic source: {institution} research demonstrates {result} in {domain}"
            ],
            "general": [
                "Information regarding {topic}: {description} with implications for {area}",
                "Documentation shows: {entity} has {characteristic} affecting {outcome}",
                "Content analysis: {subject} exhibits {pattern} suggesting {interpretation}",
                "Reference material: {source} indicates {finding} about {topic}"
            ]
        }
    
    def _load_query_templates(self) -> Dict[str, List[str]]:
        """Load query templates for different flow types."""
        return {
            "osb": [
                "Analyze this {content_type} for policy violations: '{content}'",
                "Review this {platform} content for hate speech: '{content}'",
                "Moderate this user submission: '{content}'",
                "Evaluate potential harassment in: '{content}'",
                "Assess community standards compliance: '{content}'"
            ],
            "content_moderation": [
                "Classify this content: '{content}'",
                "Review this {content_type}: '{content}'",
                "Moderate this message: '{content}'",
                "Evaluate this post: '{content}'",
                "Check this submission: '{content}'"
            ],
            "research": [
                "Research topic: '{topic}'",
                "Analyze trends in: '{domain}'",
                "Investigate: '{subject}'",
                "Study the impact of: '{factor}'",
                "Examine the relationship between: '{concept1}' and '{concept2}'"
            ]
        }
    
    def _estimate_processing_time(self, agent_name: str, query: str) -> float:
        """Estimate realistic processing time for agent and query."""
        template = self.agent_response_templates.get(agent_name, {})
        time_range = template.get("processing_time_range", (1.0, 3.0))
        
        # Base time from template
        base_time = random.uniform(*time_range)
        
        # Adjust for query complexity
        complexity_factor = min(len(query) / 100, 2.0)  # Max 2x multiplier
        
        return base_time * (1 + complexity_factor * 0.5)
    
    def _generate_response_data(self, agent_name: str, query: str, 
                               flow_context: str) -> Dict[str, Any]:
        """Generate realistic response data for agent."""
        template = self.agent_response_templates.get(agent_name, {})
        response_keys = template.get("response_keys", ["analysis", "findings"])
        
        response_data = {}
        
        for key in response_keys:
            if key in ["findings", "analysis"]:
                response_data[key] = f"{agent_name} analysis of query regarding {flow_context}"
            elif key in ["confidence", "accuracy_score"]:
                response_data[key] = random.uniform(0.7, 0.95)
            elif key in ["violations", "recommendations"]:
                response_data[key] = [f"{agent_name}_recommendation_{i}" for i in range(1, 4)]
            elif key in ["themes", "patterns"]:
                response_data[key] = [f"theme_{i}" for i in range(1, random.randint(2, 5))]
            else:
                response_data[key] = f"{agent_name} {key} result for {flow_context}"
        
        return response_data
    
    def _generate_sources(self, agent_name: str, flow_context: str) -> List[str]:
        """Generate realistic sources for agent response."""
        template = self.agent_response_templates.get(agent_name, {})
        source_types = template.get("source_types", ["general"])
        
        sources = []
        for i, source_type in enumerate(source_types[:3]):  # Max 3 sources
            if flow_context:
                source = f"{flow_context}_{source_type}_{i+1}"
            else:
                source = f"{source_type}_source_{i+1}"
            sources.append(source)
        
        return sources
    
    def _get_confidence_factors(self, agent_name: str, query: str) -> Dict[str, float]:
        """Generate confidence factors for response."""
        return {
            "query_clarity": min(len(query.split()) / 10, 1.0),
            "agent_expertise": random.uniform(0.8, 1.0),
            "source_quality": random.uniform(0.7, 0.95),
            "processing_completeness": random.uniform(0.85, 1.0)
        }
    
    def _generate_vector_content(self, query: str, flow_context: str, 
                                result_index: int) -> str:
        """Generate realistic vector search content."""
        # Determine content type based on query and flow
        if "policy" in query.lower() or flow_context == "osb":
            content_type = "policy"
        elif "research" in query.lower() or flow_context == "research":
            content_type = "research"
        elif "case" in query.lower():
            content_type = "precedent"
        else:
            content_type = "general"
        
        templates = self.vector_content_templates.get(content_type, self.vector_content_templates["general"])
        template = random.choice(templates)
        
        # Fill template with relevant content
        placeholders = {
            "section": f"{random.randint(1,10)}.{random.randint(1,5)}",
            "topic": query.split()[:3],  # First few words of query
            "number": f"{random.randint(1,20)}",
            "action": random.choice(["removed", "flagged", "reviewed", "approved"]),
            "case_id": f"CASE-{random.randint(1000,9999)}",
            "finding": "significant correlation",
            "content_type": random.choice(["social media post", "comment", "video", "image"]),
            "institution": random.choice(["Stanford", "MIT", "Oxford", "Cambridge"]),
            "platform": random.choice(["Twitter", "Facebook", "Instagram", "TikTok"])
        }
        
        # Simple template filling (in production, would use proper templating)
        content = template
        for key, value in placeholders.items():
            if f"{{{key}}}" in content:
                content = content.replace(f"{{{key}}}", str(value))
        
        return content
    
    def _generate_synthesis_result(self, query: str, agent_responses: Dict[str, AgentResponse],
                                  flow_context: str) -> Dict[str, Any]:
        """Generate realistic synthesis result from agent responses."""
        # Calculate average confidence
        avg_confidence = sum(resp.confidence for resp in agent_responses.values()) / len(agent_responses)
        
        # Generate synthesis based on flow context
        if flow_context == "osb":
            synthesis = {
                "summary": "Multi-agent analysis indicates policy violation with high confidence",
                "policy_violations": ["Community Standards 4.2", "Harassment Policy 3.1"],
                "recommendations": ["Remove content", "Issue warning", "Monitor user"],
                "confidence": avg_confidence,
                "consensus_level": "high" if avg_confidence > 0.8 else "medium"
            }
        elif flow_context == "content_moderation":
            synthesis = {
                "summary": "Content classification completed with recommendations",
                "classification": "violation" if avg_confidence > 0.8 else "borderline",
                "action": "remove" if avg_confidence > 0.85 else "review",
                "confidence": avg_confidence
            }
        elif flow_context == "research":
            synthesis = {
                "summary": "Research analysis completed with comprehensive findings",
                "key_insights": ["insight_1", "insight_2", "insight_3"],
                "conclusions": "Evidence supports research hypothesis",
                "confidence": avg_confidence,
                "research_quality": "high" if avg_confidence > 0.8 else "medium"
            }
        else:
            synthesis = {
                "summary": f"Multi-agent analysis completed for {flow_context}",
                "overall_result": "analysis_complete",
                "confidence": avg_confidence
            }
        
        # Add cross-validation results
        synthesis["cross_validation"] = {
            "agent_agreement": random.uniform(0.7, 0.95),
            "consistency_score": random.uniform(0.8, 1.0),
            "validation_passes": len(agent_responses)
        }
        
        return synthesis


class TestDataManager:
    """Manages test data lifecycle and persistence."""
    
    def __init__(self, data_dir: Path = None):
        """Initialize with optional data directory."""
        self.data_dir = data_dir or Path(__file__).parent / "test_data"
        self.data_dir.mkdir(exist_ok=True)
        self.generator = TestDataGenerator()
        self.active_datasets = {}
    
    def create_test_dataset(self, dataset_name: str, flow_configs: Dict[str, Any]) -> str:
        """Create complete test dataset for flow configurations."""
        dataset_id = f"{dataset_name}_{int(time.time())}"
        dataset = {
            "id": dataset_id,
            "name": dataset_name,
            "created_at": time.time(),
            "flow_configs": flow_configs,
            "test_scenarios": {},
            "mock_data": {}
        }
        
        # Generate test scenarios for each flow
        for flow_name, flow_config in flow_configs.items():
            dataset["test_scenarios"][flow_name] = self._create_flow_scenarios(
                flow_name, flow_config
            )
            dataset["mock_data"][flow_name] = self._create_flow_mock_data(
                flow_name, flow_config
            )
        
        # Persist dataset
        dataset_file = self.data_dir / f"{dataset_id}.json"
        with open(dataset_file, 'w') as f:
            json.dump(dataset, f, indent=2, default=str)
        
        self.active_datasets[dataset_id] = dataset
        return dataset_id
    
    def get_test_data(self, dataset_id: str, flow_name: str, 
                     data_type: str = "scenarios") -> Any:
        """Retrieve test data from dataset."""
        if dataset_id not in self.active_datasets:
            self._load_dataset(dataset_id)
        
        dataset = self.active_datasets[dataset_id]
        
        if data_type == "scenarios":
            return dataset["test_scenarios"].get(flow_name, [])
        elif data_type == "mock_data":
            return dataset["mock_data"].get(flow_name, {})
        else:
            return dataset.get(data_type)
    
    def cleanup_dataset(self, dataset_id: str):
        """Clean up test dataset."""
        if dataset_id in self.active_datasets:
            del self.active_datasets[dataset_id]
        
        dataset_file = self.data_dir / f"{dataset_id}.json"
        if dataset_file.exists():
            dataset_file.unlink()
    
    def _load_dataset(self, dataset_id: str):
        """Load dataset from disk."""
        dataset_file = self.data_dir / f"{dataset_id}.json"
        if dataset_file.exists():
            with open(dataset_file, 'r') as f:
                dataset = json.load(f)
            self.active_datasets[dataset_id] = dataset
    
    def _create_flow_scenarios(self, flow_name: str, flow_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create test scenarios for specific flow."""
        scenarios = []
        agents = list(flow_config.get("agents", {}).keys())
        
        # Generate different types of scenarios
        scenario_types = ["basic", "complex", "edge_case", "error", "performance"]
        
        for scenario_type in scenario_types:
            scenario = {
                "type": scenario_type,
                "flow": flow_name,
                "agents": agents,
                "queries": self._generate_scenario_queries(flow_name, scenario_type),
                "expected_responses": len(agents),
                "expected_time": self._estimate_scenario_time(scenario_type, len(agents)),
                "success_criteria": self._define_success_criteria(scenario_type)
            }
            scenarios.append(scenario)
        
        return scenarios
    
    def _create_flow_mock_data(self, flow_name: str, flow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create mock data for specific flow."""
        agents = list(flow_config.get("agents", {}).keys())
        
        mock_data = {
            "vector_results": [],
            "agent_responses": {},
            "synthesis_examples": []
        }
        
        # Generate vector search results
        sample_queries = self.generator.query_templates.get(flow_name, ["sample query"])
        for query in sample_queries[:3]:
            results = self.generator.generate_vector_search_results(query, flow_name)
            mock_data["vector_results"].extend([asdict(result) for result in results])
        
        # Generate agent responses
        for agent in agents:
            responses = []
            for query in sample_queries[:2]:
                response = self.generator.generate_agent_response(agent, query, flow_name)
                responses.append(asdict(response))
            mock_data["agent_responses"][agent] = responses
        
        # Generate synthesis examples
        for query in sample_queries[:2]:
            workflow_result = self.generator.generate_workflow_result(flow_name, query, agents)
            if workflow_result.synthesis_result:
                mock_data["synthesis_examples"].append(workflow_result.synthesis_result)
        
        return mock_data
    
    def _generate_scenario_queries(self, flow_name: str, scenario_type: str) -> List[str]:
        """Generate queries for scenario type."""
        templates = self.generator.query_templates.get(flow_name, ["Test query for {flow}"])
        
        if scenario_type == "basic":
            return [template.format(content="basic test content", topic="basic topic", flow=flow_name) 
                   for template in templates[:2]]
        elif scenario_type == "complex":
            return [template.format(content="complex multi-faceted content requiring deep analysis", 
                                  topic="complex interdisciplinary topic", flow=flow_name)
                   for template in templates[:2]]
        elif scenario_type == "edge_case":
            return ["", "x" * 1000, "Special chars: !@#$%^&*()", "Non-ASCII: 你好世界"]
        elif scenario_type == "error":
            return ["Query with invalid flow", "Malformed request", "Missing parameters"]
        else:  # performance
            return [f"Performance test query {i}" for i in range(5)]
    
    def _estimate_scenario_time(self, scenario_type: str, num_agents: int) -> float:
        """Estimate expected scenario completion time."""
        base_time = num_agents * 2.0  # 2 seconds per agent
        
        multipliers = {
            "basic": 1.0,
            "complex": 2.0,
            "edge_case": 1.5,
            "error": 0.5,
            "performance": 1.2
        }
        
        return base_time * multipliers.get(scenario_type, 1.0)
    
    def _define_success_criteria(self, scenario_type: str) -> Dict[str, Any]:
        """Define success criteria for scenario type."""
        criteria = {
            "basic": {
                "min_confidence": 0.7,
                "required_responses": "all",
                "max_errors": 0,
                "response_quality": "good"
            },
            "complex": {
                "min_confidence": 0.6,
                "required_responses": "all",
                "max_errors": 0,
                "response_quality": "detailed"
            },
            "edge_case": {
                "min_confidence": 0.3,
                "required_responses": "any",
                "max_errors": 2,
                "response_quality": "graceful"
            },
            "error": {
                "min_confidence": 0.0,
                "required_responses": "none",
                "max_errors": "unlimited",
                "response_quality": "error_handled"
            },
            "performance": {
                "min_confidence": 0.5,
                "required_responses": "most",
                "max_errors": 1,
                "response_quality": "fast"
            }
        }
        
        return criteria.get(scenario_type, criteria["basic"])


# Convenience functions for test setup

def create_mock_agent_response(agent_name: str, query: str = "test query") -> Dict[str, Any]:
    """Quick mock agent response creation."""
    generator = TestDataGenerator()
    response = generator.generate_agent_response(agent_name, query)
    return asdict(response)


def create_mock_vector_results(query: str = "test query", flow: str = "test", 
                              count: int = 3) -> List[Dict[str, Any]]:
    """Quick mock vector results creation."""
    generator = TestDataGenerator()
    results = generator.generate_vector_search_results(query, flow, count)
    return [asdict(result) for result in results]


def create_test_flow_config(flow_name: str, agents: List[str]) -> Dict[str, Any]:
    """Create test flow configuration."""
    return {
        "name": f"{flow_name.title()} Test Flow",
        "agents": {agent: {"type": f"{agent}_agent"} for agent in agents},
        "description": f"Test configuration for {flow_name} flow"
    }