"""Integration tests for TMDB tool with autogen agents.

These tests verify that the TMDBTool works correctly when integrated with
the autogen agent system, including tool announcements, function calling,
and observability integration.
"""

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool

from buttermilk.tools.catalog_test import TMDBTool, Observation


async def run_function_tool(function_tool: FunctionTool, **kwargs) -> Any:
    """Helper to run FunctionTool with proper argument model."""
    args_class = function_tool.args_type()
    args_model = args_class(**kwargs)
    return await function_tool.run(args_model, CancellationToken())


# Pytest markers for conditional test execution
pytestmark = pytest.mark.integration


@pytest.fixture
def tmdb_tool_for_agent() -> TMDBTool:
    """Create TMDBTool for agent integration testing."""
    return TMDBTool(api_key="fake-api-key-for-agent-testing")


class TestTMDBAgentIntegration:
    """Test TMDB tool integration with autogen agents."""

    def test_tool_as_function_tool_creation(self, tmdb_tool_for_agent: TMDBTool) -> None:
        """Test that TMDBTool can be converted to FunctionTool for agent use."""
        function_tool = tmdb_tool_for_agent.as_tool()
        
        # Verify it's a proper FunctionTool
        assert isinstance(function_tool, FunctionTool)
        assert function_tool.name == "tmdb_search"
        assert "movie availability" in function_tool.description.lower()
        assert "TMDB" in function_tool.description or "Movie Database" in function_tool.description
        
        # Verify it has the run method for execution
        assert hasattr(function_tool, "run")
        assert callable(function_tool.run)

    def test_tool_function_signature_compatibility(self, tmdb_tool_for_agent: TMDBTool) -> None:
        """Test that the tool function signature is compatible with autogen."""
        function_tool = tmdb_tool_for_agent.as_tool()
        
        # Check the tool schema for expected parameters
        schema = function_tool.schema
        
        # Schema should have parameters with properties
        parameters = schema.get("parameters", {})
        properties = parameters.get("properties", {})
        assert "title" in properties
        assert "year" in properties
        assert "region" in properties
        
        # Title should be required
        required = parameters.get("required", [])
        assert "title" in required

    @pytest.mark.anyio
    async def test_tool_execution_through_function_tool(self, tmdb_tool_for_agent: TMDBTool) -> None:
        """Test executing the tool through FunctionTool interface."""
        function_tool = tmdb_tool_for_agent.as_tool()
        
        # Mock the underlying TMDB calls to avoid real API calls
        with patch.object(tmdb_tool_for_agent, "tmdb") as mock_tmdb:
            # Mock movie object
            mock_movie = AsyncMock()
            mock_movie.id = 550
            mock_movie.title = "Fight Club"
            
            # Mock search response
            mock_search = AsyncMock()
            mock_search.movies = AsyncMock(return_value=[mock_movie])
            mock_tmdb.search.return_value = mock_search
            
            # Mock availability response
            mock_availability = {
                "results": {
                    "US": {
                        "flatrate": [{"provider_id": 8, "provider_name": "Netflix", "provider_type": "flatrate"}]
                    }
                }
            }
            mock_movies_resource = AsyncMock()
            mock_movies_resource.watch_providers = AsyncMock(return_value=mock_availability)
            mock_tmdb.movies.return_value = mock_movies_resource
            
            # Execute through FunctionTool using helper
            results = await run_function_tool(function_tool, title="Fight Club", year=1999, region="US")
            
            # Verify results
            assert isinstance(results, list)
            assert len(results) > 0
            
            result = results[0]
            assert isinstance(result, Observation)
            assert result.available is True
            assert result.provider_name == "Netflix"
            assert result.source == "TMDB"

    def test_tool_metadata_for_agent_discovery(self, tmdb_tool_for_agent: TMDBTool) -> None:
        """Test that tool metadata is suitable for agent discovery and routing."""
        function_tool = tmdb_tool_for_agent.as_tool()
        
        # Tool should have descriptive name for routing
        assert function_tool.name.lower() in {"tmdb_search", "movie_search", "tmdb_tool"}
        
        # Description should be informative for LLM agents
        description = function_tool.description.lower()
        keywords = ["movie", "availability", "streaming", "search", "tmdb", "database"]
        
        # Should contain relevant keywords
        found_keywords = [kw for kw in keywords if kw in description]
        min_keywords = 3
        assert len(found_keywords) >= min_keywords, f"Description should contain movie-related keywords. Found: {found_keywords}"

    @pytest.mark.anyio
    async def test_tool_error_handling_for_agents(self, tmdb_tool_for_agent: TMDBTool) -> None:
        """Test that tool errors are handled gracefully when called by agents."""
        function_tool = tmdb_tool_for_agent.as_tool()
        
        # Mock API failure
        with patch.object(tmdb_tool_for_agent, "tmdb") as mock_tmdb:
            mock_search = AsyncMock()
            mock_search.movies = AsyncMock(side_effect=Exception("API Error"))
            mock_tmdb.search.return_value = mock_search
            
            # Tool should handle errors gracefully
            results = await run_function_tool(function_tool, title="Test Movie", region="US")
            
            assert isinstance(results, list)
            assert len(results) == 1
            
            result = results[0]
            assert isinstance(result, Observation)
            assert result.available is False
            assert len(result.error) > 0
            assert "API Error" in str(result.error)

    def test_tool_parameter_validation_for_agents(self, tmdb_tool_for_agent: TMDBTool) -> None:
        """Test that tool validates parameters appropriately for agent use."""
        function_tool = tmdb_tool_for_agent.as_tool()
        
        # Check the tool schema for parameter validation
        schema = function_tool.schema
        parameters = schema.get("parameters", {})
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        
        # Title should be required string
        assert "title" in required
        assert properties["title"]["type"] == "string"
        
        # Year should be optional integer (may have anyOf for nullable)
        year_prop = properties.get("year", {})
        assert "year" not in required  # Should be optional
        assert year_prop.get("type") == "integer" or "anyOf" in year_prop
        
        # Region should be string with default
        assert properties["region"]["type"] == "string"
        assert "default" in properties["region"]

    @pytest.mark.anyio
    async def test_tool_observability_integration(self, tmdb_tool_for_agent: TMDBTool) -> None:
        """Test that tool execution is properly observable when called by agents."""
        function_tool = tmdb_tool_for_agent.as_tool()
        
        # Mock the underlying TMDB calls
        with patch.object(tmdb_tool_for_agent, "tmdb") as mock_tmdb:
            mock_movie = AsyncMock()
            mock_movie.id = 550
            mock_movie.title = "Test Movie"
            
            mock_search = AsyncMock()
            mock_search.movies = AsyncMock(return_value=[mock_movie])
            mock_tmdb.search.return_value = mock_search
            
            mock_availability = {"results": {"US": {}}}
            mock_movies_resource = AsyncMock()
            mock_movies_resource.watch_providers = AsyncMock(return_value=mock_availability)
            mock_tmdb.movies.return_value = mock_movies_resource
            
            # Execute tool
            results = await run_function_tool(function_tool, title="Test Movie", region="US")
            
            # Verify observability data
            assert isinstance(results, list)
            result = results[0]
            assert isinstance(result, Observation)
            
            # Should have tracing metadata
            assert result.call_id is not None
            assert result.test_date is not None
            assert result.record_id is not None
            
            # Metadata should include search parameters for tracing
            assert "search_title" in result.metadata
            assert result.metadata["search_title"] == "Test Movie"

    def test_tool_configuration_for_production_agents(self) -> None:
        """Test tool configuration patterns suitable for production agent deployment."""
        # Test with environment variable
        with patch.dict(os.environ, {"TMDB_API_KEY": "production-api-key"}):
            tool = TMDBTool()
            assert tool.api_key == "production-api-key"
            
            function_tool = tool.as_tool()
            assert isinstance(function_tool, FunctionTool)
        
        # Test with explicit configuration
        tool = TMDBTool(
            api_key="explicit-api-key",
            language="en-US",
            region="US"
        )
        assert tool.language == "en-US"
        assert tool.region == "US"

    @pytest.mark.anyio
    async def test_tool_concurrent_execution_safety(self, tmdb_tool_for_agent: TMDBTool) -> None:
        """Test that tool can be safely used by multiple agents concurrently."""
        function_tool = tmdb_tool_for_agent.as_tool()
        
        # Mock the underlying calls
        with patch.object(tmdb_tool_for_agent, "tmdb") as mock_tmdb:
            mock_movie = AsyncMock()
            mock_movie.id = 550
            mock_movie.title = "Concurrent Movie"
            
            mock_search = AsyncMock()
            mock_search.movies = AsyncMock(return_value=[mock_movie])
            mock_tmdb.search.return_value = mock_search
            
            mock_availability = {"results": {"US": {}}}
            mock_movies_resource = AsyncMock()
            mock_movies_resource.watch_providers = AsyncMock(return_value=mock_availability)
            mock_tmdb.movies.return_value = mock_movies_resource
            
            # Execute multiple concurrent calls
            import asyncio  # noqa: F401
            tasks = [
                run_function_tool(function_tool, title=f"Movie {i}", region="US")
                for i in range(3)
            ]
            
            results_list = await asyncio.gather(*tasks)
            
            # All calls should succeed
            expected_results = 3
            assert len(results_list) == expected_results
            for results in results_list:
                assert isinstance(results, list)
                assert len(results) == 1
                assert isinstance(results[0], Observation)

    def test_tool_integration_with_strict_mode(self, tmdb_tool_for_agent: TMDBTool) -> None:
        """Test that tool works with autogen's strict mode."""
        function_tool = tmdb_tool_for_agent.as_tool()
        
        # Check if strict mode is supported (may be a private attribute)
        # Note: strict mode requirements are mainly about schema completeness
        
        # Tool should have well-defined schema for strict mode
        schema = function_tool.schema
        
        # Schema should have all required fields for strict mode
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema
        
        # Parameters should have proper structure
        parameters = schema.get("parameters", {})
        assert "type" in parameters
        assert "properties" in parameters
        assert "required" in parameters
        
        # All properties should have proper type definitions
        properties = parameters.get("properties", {})
        for param_name, param_schema in properties.items():
            # Check if type is directly present or in anyOf structure
            has_type = "type" in param_schema or "anyOf" in param_schema
            assert has_type, f"Parameter {param_name} missing type definition in schema"


if __name__ == "__main__":
    # Allow running this file directly for quick testing
    pytest.main([__file__, "-v"])
