"""Test the tool-based architecture of EnhancedRagAgent."""

import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from buttermilk._core.contract import AgentInput, AgentOutput, ToolOutput
from buttermilk._core.tool_definition import UnifiedRequest
from buttermilk.agents.rag.enhanced_rag_agent import EnhancedRagAgent


class TestEnhancedRAGToolBased:
    """Test the new tool-based architecture of EnhancedRagAgent."""
    
    @pytest.fixture
    def mock_rag_agent(self):
        """Create a properly mocked enhanced RAG agent."""
        # Patch the parent class initialization
        with patch('buttermilk.agents.rag.rag_agent.RagAgent._load_tools'), \
             patch('buttermilk.agents.rag.rag_agent.RagAgent.ensure_chromadb_ready', new_callable=AsyncMock):
            
            agent = EnhancedRagAgent(
                agent_name="test_rag",
                model_name="test-model",
                role="RAG",
                parameters={"model": "test-model"},
                data={
                    "chromadb": {
                        "type": "chromadb",
                        "collection_name": "test_collection"
                    }
                }
            )
            
            # Mock internal components
            agent._chromadb = Mock()
            agent._vectorstore = Mock()
            agent._tools_list = []
            
            # Mock the parent's _query_db method
            async def mock_query_db(query):
                return ToolOutput(
                    name="test_rag",
                    call_id="",
                    content="Test results",
                    results=[],
                    args={"query": query},
                    messages=[]
                )
            
            agent._query_db = mock_query_db
            
            return agent
    
    def test_extract_query_variations(self, mock_rag_agent):
        """Test extracting query from various input formats."""
        test_cases = [
            ({"query": "test query"}, "test query"),
            ({"question": "test question"}, "test question"),
            ({"search": "test search"}, "test search"),
            ({"text": "test text"}, "test text"),
            ({"content": "test content"}, "test content"),
            ({"custom": "custom value"}, "custom value"),
            ({}, ""),
        ]
        
        for inputs, expected in test_cases:
            message = AgentInput(inputs=inputs)
            query = mock_rag_agent._extract_query(message)
            assert query == expected, f"Failed for inputs: {inputs}"
    
    @pytest.mark.asyncio
    async def test_semantic_search_tool(self, mock_rag_agent):
        """Test the semantic_search tool method."""
        result = await mock_rag_agent.semantic_search(
            query="test semantic search",
            n_results=5
        )
        
        # Parse and verify result
        result_data = json.loads(result)
        assert result_data["query"] == "test semantic search"
        assert "results" in result_data
        assert "total_found" in result_data
    
    @pytest.mark.asyncio
    async def test_field_search_tool(self, mock_rag_agent):
        """Test the field_search tool method."""
        result = await mock_rag_agent.field_search(
            query="test field search",
            content_type="title",
            n_results=3
        )
        
        # Parse and verify result
        result_data = json.loads(result)
        assert result_data["query"] == "test field search"
        assert "results" in result_data
    
    @pytest.mark.asyncio
    async def test_analyze_query_tool(self, mock_rag_agent):
        """Test the analyze_query tool method."""
        result = await mock_rag_agent.analyze_query("What are the legal implications?")
        
        # Parse and verify result
        analysis = json.loads(result)
        assert analysis["query"] == "What are the legal implications?"
        assert "intent" in analysis
        assert "query_type" in analysis
        assert "suggested_strategies" in analysis
    
    @pytest.mark.asyncio
    async def test_create_search_plan_tool(self, mock_rag_agent):
        """Test the create_search_plan tool method."""
        result = await mock_rag_agent.create_search_plan("Find technical documentation")
        
        # Parse and verify result
        plan = json.loads(result)
        assert plan["query"] == "Find technical documentation"
        assert "primary_strategy" in plan
        assert "secondary_strategies" in plan
        assert "metadata_filters" in plan
    
    @pytest.mark.asyncio
    async def test_hybrid_search_tool(self, mock_rag_agent):
        """Test the hybrid_search tool method."""
        # Mock the field_search method to return test data
        async def mock_field_search(query, content_type, n_results, metadata_filters=None):
            return json.dumps({
                "query": query,
                "results": [{
                    "chunk_id": f"{content_type}_chunk_1",
                    "document_id": f"doc_{content_type}",
                    "content": f"Test content from {content_type}"
                }],
                "total_found": 1
            })
        
        mock_rag_agent.field_search = mock_field_search
        
        result = await mock_rag_agent.hybrid_search(
            query="test hybrid search",
            n_results=9
        )
        
        # Parse and verify result
        result_data = json.loads(result)
        assert result_data["query"] == "test hybrid search"
        assert result_data["search_type"] == "hybrid"
        assert len(result_data["results"]) > 0
    
    @pytest.mark.asyncio
    async def test_execute_search_plan_tool(self, mock_rag_agent):
        """Test the execute_search_plan tool method."""
        # Create a test plan
        plan = {
            "query": "test execution",
            "primary_strategy": "semantic",
            "secondary_strategies": ["title"],
            "metadata_filters": {},
            "max_results_per_strategy": 5,
            "confidence_threshold": 0.5
        }
        
        # Mock semantic_search to return test data
        async def mock_semantic_search(query, n_results, content_type=None, metadata_filters=None):
            return json.dumps({
                "query": query,
                "results": [{
                    "chunk_id": "semantic_1",
                    "document_id": "doc_1",
                    "content": "Semantic search result"
                }],
                "total_found": 1
            })
        
        mock_rag_agent.semantic_search = mock_semantic_search
        
        result = await mock_rag_agent.execute_search_plan(json.dumps(plan))
        
        # Parse and verify result
        result_data = json.loads(result)
        assert result_data["query"] == "test execution"
        assert "semantic" in result_data["strategies_used"]
        assert len(result_data["results"]) > 0
    
    def test_tool_definitions(self, mock_rag_agent):
        """Test that all expected tools are defined."""
        tools = mock_rag_agent.get_tool_definitions()
        tool_names = {tool.name for tool in tools}
        
        expected_tools = {
            "semantic_search",
            "field_search", 
            "hybrid_search",
            "analyze_query",
            "create_search_plan",
            "execute_search_plan"
        }
        
        # Check parent tools are also included
        assert "search_knowledge_base" in tool_names  # From parent RagAgent
        
        # Check our new tools
        for tool_name in expected_tools:
            assert tool_name in tool_names, f"Missing tool: {tool_name}"
    
    @pytest.mark.asyncio
    async def test_process_with_planning_enabled(self, mock_rag_agent):
        """Test _process method with query planning enabled."""
        mock_rag_agent.enable_query_planning = True
        
        # Mock the create_search_plan method
        async def mock_create_plan(query):
            return json.dumps({
                "query": query,
                "intent": "Test search",
                "primary_strategy": "semantic",
                "secondary_strategies": [],
                "metadata_filters": {}
            })
        
        mock_rag_agent.create_search_plan = mock_create_plan
        
        # Mock semantic_search
        async def mock_search(query, n_results, content_type=None):
            return json.dumps({
                "query": query,
                "results": [{
                    "document_id": "test_doc",
                    "content": "Test content"
                }],
                "total_found": 1
            })
        
        mock_rag_agent.semantic_search = mock_search
        
        # Process a message
        message = AgentInput(inputs={"query": "test query"})
        result = await mock_rag_agent._process(message=message)
        
        assert isinstance(result, AgentOutput)
        assert "Found 1 relevant results" in result.outputs
        assert result.metadata["query"] == "test query"
    
    @pytest.mark.asyncio  
    async def test_unified_request_handling(self, mock_rag_agent):
        """Test handling of UnifiedRequest for tool methods."""
        request = UnifiedRequest(
            target="test_rag.semantic_search",
            inputs={
                "query": "unified request test",
                "n_results": 3
            },
            context={},
            metadata={}
        )
        
        result = await mock_rag_agent.handle_unified_request(request)
        
        # Should return JSON string
        result_data = json.loads(result)
        assert result_data["query"] == "unified request test"