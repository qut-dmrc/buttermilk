"""Test fix for enhanced RAG agent query extraction with structured tools."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from buttermilk._core.contract import AgentInput, AgentOutput
from buttermilk._core.tool_definition import UnifiedRequest
from buttermilk.agents.rag.enhanced_rag_agent import EnhancedRagAgent


class TestEnhancedRAGQueryFix:
    """Test that enhanced RAG agent properly extracts query from different input formats."""
    
    @pytest.fixture
    def mock_rag_agent(self):
        """Create a mock enhanced RAG agent."""
        # Mock the data configuration for ChromaDB
        with patch('buttermilk.agents.rag.rag_agent.ChromaDBEmbeddings'):
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
            
            # Mock the ChromaDB components
            agent._chromadb = Mock()
            agent._vectorstore = Mock()
            agent.ensure_chromadb_ready = AsyncMock()
            
            # Mock the query_db method from parent
            from buttermilk._core.contract import ToolOutput
            agent._query_db = AsyncMock(return_value=ToolOutput(
                name="test_rag",
                call_id="",
                content="Test results",
                results=[],
                args={"query": "test"},
                messages=[]
            ))
            
            return agent
    
    def test_extract_query_from_inputs_query(self, mock_rag_agent):
        """Test extracting query from inputs.query field."""
        message = AgentInput(inputs={"query": "test query"})
        query = mock_rag_agent._extract_query(message)
        assert query == "test query"
    
    def test_extract_query_from_inputs_question(self, mock_rag_agent):
        """Test extracting query from inputs.question field."""
        message = AgentInput(inputs={"question": "test question"})
        query = mock_rag_agent._extract_query(message)
        assert query == "test question"
    
    def test_extract_query_from_inputs_search(self, mock_rag_agent):
        """Test extracting query from inputs.search field."""
        message = AgentInput(inputs={"search": "test search"})
        query = mock_rag_agent._extract_query(message)
        assert query == "test search"
    
    def test_extract_query_from_inputs_text(self, mock_rag_agent):
        """Test extracting query from inputs.text field."""
        message = AgentInput(inputs={"text": "test text"})
        query = mock_rag_agent._extract_query(message)
        assert query == "test text"
    
    def test_extract_query_from_inputs_content(self, mock_rag_agent):
        """Test extracting query from inputs.content field."""
        message = AgentInput(inputs={"content": "test content"})
        query = mock_rag_agent._extract_query(message)
        assert query == "test content"
    
    def test_extract_query_fallback_to_first_string(self, mock_rag_agent):
        """Test fallback to first string value in inputs."""
        message = AgentInput(inputs={"custom_field": "custom query", "number": 123})
        query = mock_rag_agent._extract_query(message)
        assert query == "custom query"
    
    def test_extract_query_empty_inputs(self, mock_rag_agent):
        """Test empty string returned when no query found."""
        message = AgentInput(inputs={})
        query = mock_rag_agent._extract_query(message)
        assert query == ""
    
    @pytest.mark.asyncio
    async def test_semantic_search_tool_method(self, mock_rag_agent):
        """Test the @tool decorated semantic_search method."""
        # Call the semantic search tool
        result = await mock_rag_agent.semantic_search(query="test query", n_results=5)
        
        # Parse result
        import json
        result_data = json.loads(result)
        
        # Verify result structure
        assert "query" in result_data
        assert result_data["query"] == "test query"
        assert "results" in result_data
        assert "total_found" in result_data
        
        # Verify the parent's _query_db was called
        mock_rag_agent._query_db.assert_called_once_with("test query")
    
    @pytest.mark.asyncio
    async def test_unified_request_handling(self, mock_rag_agent):
        """Test that UnifiedRequest is properly handled."""
        # Create a UnifiedRequest for semantic_search
        request = UnifiedRequest(
            target="test_rag.semantic_search",
            inputs={"query": "unified test query", "n_results": 5},
            context={},
            metadata={}
        )
        
        # Handle the unified request
        result = await mock_rag_agent.handle_unified_request(request)
        
        # Parse result
        import json
        result_data = json.loads(result)
        
        # Should return JSON with search results
        assert "query" in result_data
        assert result_data["query"] == "unified test query"
    
    def test_tool_definitions_exist(self, mock_rag_agent):
        """Test that the search tools are properly defined."""
        tools = mock_rag_agent.get_tool_definitions()
        
        # Should have multiple search tools
        assert len(tools) >= 5  # At least 5 search-related tools
        
        # Check for expected tools
        tool_names = {tool.name for tool in tools}
        expected_tools = {
            "semantic_search",
            "field_search",
            "hybrid_search",
            "analyze_query",
            "create_search_plan",
            "execute_search_plan"
        }
        
        # Find semantic_search tool specifically
        semantic_tool = None
        for tool in tools:
            if tool.name == "semantic_search":
                semantic_tool = tool
                break
        
        assert semantic_tool is not None
        assert "query" in semantic_tool.input_schema["properties"]
        assert "n_results" in semantic_tool.input_schema["properties"]
        
        # Verify all expected tools are present
        for expected in expected_tools:
            assert expected in tool_names, f"Missing tool: {expected}"
    
    @pytest.mark.asyncio
    async def test_analyze_query_tool(self, mock_rag_agent):
        """Test the analyze_query tool method."""
        result = await mock_rag_agent.analyze_query("What are the legal implications?")
        
        import json
        analysis = json.loads(result)
        
        # Verify result structure
        assert "query" in analysis
        assert "intent" in analysis
        assert "query_type" in analysis
        assert "key_concepts" in analysis
        assert "suggested_strategies" in analysis
        assert analysis["query"] == "What are the legal implications?"
    
    @pytest.mark.asyncio
    async def test_create_search_plan_tool(self, mock_rag_agent):
        """Test the create_search_plan tool method."""
        result = await mock_rag_agent.create_search_plan("Find technical documentation")
        
        import json
        plan = json.loads(result)
        
        # Verify plan structure
        assert "query" in plan
        assert "intent" in plan
        assert "primary_strategy" in plan
        assert "secondary_strategies" in plan
        assert "metadata_filters" in plan
        assert plan["query"] == "Find technical documentation"
    
    @pytest.mark.asyncio
    async def test_hybrid_search_tool(self, mock_rag_agent):
        """Test the hybrid_search tool method."""
        # Mock field_search to return different results
        async def mock_field_search(query, content_type, n_results, metadata_filters=None):
            import json
            return json.dumps({
                "query": query,
                "results": [{
                    "chunk_id": f"{content_type}_1",
                    "document_id": f"doc_{content_type}_1",
                    "content": f"Test {content_type} content"
                }],
                "total_found": 1
            })
        
        mock_rag_agent.field_search = mock_field_search
        
        result = await mock_rag_agent.hybrid_search("test hybrid query", n_results=10)
        
        import json
        result_data = json.loads(result)
        
        # Verify result structure
        assert "query" in result_data
        assert "results" in result_data
        assert "total_found" in result_data
        assert "search_type" in result_data
        assert result_data["search_type"] == "hybrid"
        assert result_data["query"] == "test hybrid query"