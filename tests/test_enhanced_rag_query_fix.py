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
        with patch('buttermilk.agents.rag.enhanced_rag_agent.EnhancedVectorSearch'):
            agent = EnhancedRagAgent(
                agent_name="test_rag",
                model_name="test-model",
                role="RAG",
                parameters={"model": "test-model"}
            )
            
            # Mock the search components
            agent._enhanced_search = Mock()
            agent._enhanced_search.create_search_plan = AsyncMock()
            agent._enhanced_search.execute_search_plan = AsyncMock()
            agent._generate_response = AsyncMock(return_value="Test response")
            
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
    async def test_search_tool_method(self, mock_rag_agent):
        """Test the new @tool decorated search method."""
        # Mock the search plan and results
        from buttermilk.agents.rag.search_planning import SearchPlan, SearchResults, SearchStrategy
        
        mock_plan = SearchPlan(
            query="test query",
            intent="Test search",
            primary_strategy=SearchStrategy.SEMANTIC,
            max_results_per_strategy=10,
            confidence_threshold=0.7
        )
        
        mock_results = SearchResults(
            query="test query",
            plan=mock_plan,
            results=[],
            total_found=0,
            strategies_used=[SearchStrategy.SEMANTIC],
            confidence_score=0.0,
            key_themes=[]
        )
        
        mock_rag_agent._enhanced_search.create_search_plan.return_value = mock_plan
        mock_rag_agent._enhanced_search.execute_search_plan.return_value = mock_results
        
        # Call the tool method
        result = await mock_rag_agent.search(query="test query", max_results=5)
        
        # Verify result
        assert result == "Test response"
        
        # Verify the search was called with correct query
        mock_rag_agent._enhanced_search.create_search_plan.assert_called_once()
        call_args = mock_rag_agent._enhanced_search.create_search_plan.call_args[0]
        assert call_args[0] == "test query"
    
    @pytest.mark.asyncio
    async def test_unified_request_handling(self, mock_rag_agent):
        """Test that UnifiedRequest is properly handled."""
        # Create a UnifiedRequest
        request = UnifiedRequest(
            target="test_rag.search",
            inputs={"query": "unified test query"},
            context={},
            metadata={}
        )
        
        # Mock the search components
        from buttermilk.agents.rag.search_planning import SearchPlan, SearchResults, SearchStrategy
        
        mock_plan = SearchPlan(
            query="unified test query",
            intent="Test search",
            primary_strategy=SearchStrategy.SEMANTIC,
            max_results_per_strategy=10,
            confidence_threshold=0.7
        )
        
        mock_results = SearchResults(
            query="unified test query",
            plan=mock_plan,
            results=[],
            total_found=0,
            strategies_used=[SearchStrategy.SEMANTIC],
            confidence_score=0.0,
            key_themes=[]
        )
        
        mock_rag_agent._enhanced_search.create_search_plan.return_value = mock_plan
        mock_rag_agent._enhanced_search.execute_search_plan.return_value = mock_results
        
        # Handle the unified request
        result = await mock_rag_agent.handle_unified_request(request)
        
        # Should return the string output
        assert result == "Test response"
    
    def test_tool_definition_exists(self, mock_rag_agent):
        """Test that the search tool is properly defined."""
        tools = mock_rag_agent.get_tool_definitions()
        
        # Should have at least the search tool
        assert len(tools) >= 1
        
        # Find the search tool
        search_tool = None
        for tool in tools:
            if tool.name == "search":
                search_tool = tool
                break
        
        assert search_tool is not None
        assert search_tool.description == "Search the knowledge base with an intelligent RAG system"
        assert "query" in search_tool.input_schema["properties"]
        assert "max_results" in search_tool.input_schema["properties"]