"""
Tests for score pages API endpoints
"""

import pytest
from unittest.mock import Mock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, HTTPException

from buttermilk.api.services.data_service import DataService


class TestDataService:
    """Test the DataService methods"""

    @pytest.fixture
    def mock_flow_runner(self):
        """Mock flow runner with test data"""
        mock_runner = Mock()
        mock_runner.flows = {
            "test_flow": Mock()
        }
        
        # Mock data loader
        mock_record = Mock()
        mock_record.record_id = "test_record_1"
        mock_record.title = "Test Record"
        mock_record.content = "Test content for toxicity analysis"
        
        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([mock_record]))
        
        # Mock the data loader creation
        with pytest.MonkeyPatch().context() as m:
            m.setattr("buttermilk.api.services.data_service.create_data_loader", lambda x: mock_loader)
            yield mock_runner

    @pytest.fixture
    def mock_bm_instance(self):
        """Mock BM instance with BigQuery client"""
        mock_bm = Mock()
        mock_bm.bq = Mock()
        mock_bm.bq.project = "test-project"
        return mock_bm

    @pytest.mark.anyio
    async def test_get_record_by_id_found(self, mock_flow_runner):
        """Test getting a record that exists"""
        with pytest.MonkeyPatch().context() as m:
            # Mock the data loader
            mock_record = Mock()
            mock_record.record_id = "test_record_1"
            mock_record.title = "Test Record"
            mock_record.content = "Test content"
            
            mock_loader = [mock_record]
            m.setattr("buttermilk.api.services.data_service.create_data_loader", lambda x: mock_loader)
            
            result = await DataService.get_record_by_id("test_record_1", "test_flow", mock_flow_runner)
            
            assert result is not None
            assert result["id"] == "test_record_1"
            assert result["name"] == "Test Record"
            assert result["content"] == "Test content"
            assert result["metadata"]["dataset"] == "test_flow"

    @pytest.mark.anyio
    async def test_get_record_by_id_not_found(self, mock_flow_runner):
        """Test getting a record that doesn't exist"""
        with pytest.MonkeyPatch().context() as m:
            mock_loader = []  # Empty loader
            m.setattr("buttermilk.api.services.data_service.create_data_loader", lambda x: mock_loader)
            
            result = await DataService.get_record_by_id("nonexistent", "test_flow", mock_flow_runner)
            
            assert result is None

    @pytest.mark.anyio
    async def test_get_records_for_flow_without_scores(self, mock_flow_runner):
        """Test getting records list without scores"""
        with pytest.MonkeyPatch().context() as m:
            mock_record = Mock()
            mock_record.record_id = "test_record_1"
            mock_record.title = "Test Record"
            
            mock_loader = [mock_record]
            m.setattr("buttermilk.api.services.data_service.create_data_loader", lambda x: mock_loader)
            
            result = await DataService.get_records_for_flow("test_flow", mock_flow_runner, include_scores=False)
            
            assert len(result) == 1
            assert result[0]["record_id"] == "test_record_1"
            assert result[0]["name"] == "Test Record"
            assert "summary_scores" not in result[0]

    @pytest.mark.anyio
    async def test_get_records_for_flow_with_scores(self, mock_flow_runner):
        """Test getting records list with scores"""
        with pytest.MonkeyPatch().context() as m:
            mock_record = Mock()
            mock_record.record_id = "test_record_1"
            mock_record.title = "Test Record"
            
            mock_loader = [mock_record]
            m.setattr("buttermilk.api.services.data_service.create_data_loader", lambda x: mock_loader)
            
            result = await DataService.get_records_for_flow("test_flow", mock_flow_runner, include_scores=True)
            
            assert len(result) == 1
            assert result[0]["record_id"] == "test_record_1"
            assert "summary_scores" in result[0]
            assert result[0]["summary_scores"]["total_evaluations"] == 8

    @pytest.mark.anyio
    async def test_get_scores_for_record_no_data(self, mock_bm_instance):
        """Test getting scores when no data exists"""
        with pytest.MonkeyPatch().context() as m:
            # Mock empty DataFrame
            import pandas as pd
            mock_df = pd.DataFrame()
            
            mock_query_runner = Mock()
            mock_query_runner.run_query = Mock(return_value=mock_df)
            
            m.setattr("buttermilk.api.services.data_service.QueryRunner", lambda bq_client: mock_query_runner)
            
            result = await DataService.get_scores_for_record("test_record", "test_flow", mock_bm_instance)
            
            assert result["record_id"] == "test_record"
            assert result["off_shelf_results"] == {}
            assert result["custom_results"] == {}
            assert result["summary"]["total_evaluations"] == 0

    @pytest.mark.anyio
    async def test_get_scores_for_record_with_data(self, mock_bm_instance):
        """Test getting scores with actual data"""
        with pytest.MonkeyPatch().context() as m:
            # Mock DataFrame with test data
            import pandas as pd
            
            test_data = {
                "judge": ["GPT-4", "Claude-3"],
                "judge_model": ["gpt-4-0613", "claude-3-sonnet"],
                "violating": [True, False],
                "confidence": ["high", "medium"],
                "correctness": [0.85, 0.42],
                "judge_template": ["toxicity_v1", "toxicity_v1"],
                "judge_criteria": ["guidelines", "guidelines"],
                "scorer": [None, None]
            }
            mock_df = pd.DataFrame(test_data)
            
            mock_query_runner = Mock()
            mock_query_runner.run_query = Mock(return_value=mock_df)
            
            m.setattr("buttermilk.api.services.data_service.QueryRunner", lambda bq_client: mock_query_runner)
            
            result = await DataService.get_scores_for_record("test_record", "test_flow", mock_bm_instance)
            
            assert result["record_id"] == "test_record"
            assert len(result["off_shelf_results"]) == 2
            assert "gpt-4-0613" in result["off_shelf_results"]
            assert "claude-3-sonnet" in result["off_shelf_results"]
            assert result["summary"]["total_evaluations"] == 2

    @pytest.mark.anyio
    async def test_get_responses_for_record(self, mock_bm_instance):
        """Test getting detailed responses for a record"""
        with pytest.MonkeyPatch().context() as m:
            # Mock DataFrame with response data
            import pandas as pd
            from datetime import datetime
            
            test_data = {
                "judge": ["GPT-4"],
                "judge_model": ["gpt-4-0613"],
                "judge_role": ["JUDGE"],
                "conclusion": ["This content violates guidelines"],
                "violating": [True],
                "confidence": ["high"],
                "reasons": [["Contains hate speech", "Targets specific group"]],
                "judge_criteria": ["community_guidelines"],
                "judge_template": ["toxicity_judge_v1"],
                "timestamp": [datetime.now()]
            }
            mock_df = pd.DataFrame(test_data)
            
            mock_query_runner = Mock()
            mock_query_runner.run_query = Mock(return_value=mock_df)
            
            m.setattr("buttermilk.api.services.data_service.QueryRunner", lambda bq_client: mock_query_runner)
            
            result = await DataService.get_responses_for_record("test_record", "test_flow", mock_bm_instance)
            
            assert result["record_id"] == "test_record"
            assert len(result["responses"]) == 1
            response = result["responses"][0]
            assert response["agent"] == "GPT-4-gpt-4-0613"
            assert response["type"] == "judge"
            assert response["content"] == "This content violates guidelines"
            assert response["prediction"] is True
            assert "reasoning" in response


class TestScoreEndpointsIntegration:
    """Integration tests for the API endpoints"""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all the FastAPI dependencies"""
        mock_flows = Mock()
        mock_flows.flows = {"test_flow": Mock()}
        
        mock_bm = Mock()
        mock_bm.bq = Mock()
        mock_bm.bq.project = "test-project"
        
        return {
            "flows": mock_flows,
            "bm_instance": mock_bm
        }

    def test_imports_work(self):
        """Test that all imports work correctly"""
        from buttermilk.api.routes import flow_data_router
        from buttermilk.api.services.data_service import DataService
        
        # Basic smoke test
        assert flow_data_router is not None
        assert DataService is not None

    @pytest.mark.anyio
    async def test_data_service_error_handling(self):
        """Test that DataService handles errors gracefully"""
        # Test with invalid flow runner
        result = await DataService.get_record_by_id("test", "flow", None)
        assert result is None
        
        # Test with broken BM instance for scores
        broken_bm = Mock()
        broken_bm.bq = None
        
        result = await DataService.get_scores_for_record("test", "flow", broken_bm)
        # Should return empty results structure instead of crashing
        assert "record_id" in result
        assert result["off_shelf_results"] == {}


@pytest.mark.integration
class TestScoreAPIEndpoints:
    """Integration tests that require real FastAPI setup"""

    def test_endpoint_registration(self):
        """Test that endpoints are properly registered"""
        from buttermilk.api.routes import flow_data_router
        
        # Check that our new endpoints are registered
        routes = [route.path for route in flow_data_router.routes]
        
        assert "/api/records/{record_id}" in routes
        assert "/api/records/{record_id}/scores" in routes
        assert "/api/records/{record_id}/responses" in routes
        assert "/api/records" in routes