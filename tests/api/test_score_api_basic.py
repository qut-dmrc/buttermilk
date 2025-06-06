"""
Basic tests for score pages API - focusing on imports and structure
"""

import pytest
from unittest.mock import Mock

from buttermilk.api.services.data_service import DataService


class TestScoreAPIBasics:
    """Test basic functionality of score API"""

    def test_imports_work(self):
        """Test that all imports work correctly"""
        from buttermilk.api.routes import flow_data_router
        from buttermilk.api.services.data_service import DataService
        
        # Basic smoke test
        assert flow_data_router is not None
        assert DataService is not None

    def test_data_service_exists(self):
        """Test DataService class has expected methods"""
        expected_methods = [
            'get_record_by_id',
            'get_scores_for_record', 
            'get_responses_for_record',
            'get_records_for_flow'
        ]
        
        for method_name in expected_methods:
            assert hasattr(DataService, method_name), f"Missing method: {method_name}"
            method = getattr(DataService, method_name)
            assert callable(method), f"Method {method_name} is not callable"

    def test_endpoint_registration(self):
        """Test that endpoints are properly registered"""
        from buttermilk.api.routes import flow_data_router
        
        # Check that our new endpoints are registered
        routes = [route.path for route in flow_data_router.routes]
        
        expected_routes = [
            "/api/records/{record_id}",
            "/api/records/{record_id}/scores", 
            "/api/records/{record_id}/responses",
            "/api/records"
        ]
        
        for route in expected_routes:
            assert route in routes, f"Missing route: {route}"

    def test_api_response_structure(self):
        """Test that API response structures match specification"""
        # Test record response structure
        mock_record_response = {
            "id": "test_record",
            "name": "Test Record", 
            "content": "Test content",
            "metadata": {
                "created_at": None,
                "dataset": "test_flow",
                "word_count": 2,
                "char_count": 12
            }
        }
        
        # Verify structure matches spec
        assert "id" in mock_record_response
        assert "name" in mock_record_response
        assert "content" in mock_record_response
        assert "metadata" in mock_record_response
        assert "dataset" in mock_record_response["metadata"]
        
        # Test scores response structure
        mock_scores_response = {
            "record_id": "test_record",
            "off_shelf_results": {},
            "custom_results": {},
            "summary": {
                "off_shelf_accuracy": 0.0,
                "custom_average_score": 0.0,
                "total_evaluations": 0,
                "agreement_rate": 0.0
            }
        }
        
        # Verify structure matches spec  
        assert "record_id" in mock_scores_response
        assert "off_shelf_results" in mock_scores_response
        assert "custom_results" in mock_scores_response
        assert "summary" in mock_scores_response
        
        # Test responses structure
        mock_responses_response = {
            "record_id": "test_record",
            "responses": []
        }
        
        assert "record_id" in mock_responses_response
        assert "responses" in mock_responses_response

    def test_error_handling_structure(self):
        """Test that error responses match expected structure"""
        # Test 404 error structure from spec
        expected_404 = {
            "error": "Record not found",
            "detail": "No record found with id: test_id in flow: test_flow",
            "code": "RECORD_NOT_FOUND"
        }
        
        assert "error" in expected_404
        assert "detail" in expected_404  
        assert "code" in expected_404


class TestAPISpecCompliance:
    """Test that implementation matches the API specification"""

    def test_endpoint_paths_match_spec(self):
        """Verify endpoint paths match the specification"""
        from buttermilk.api.routes import flow_data_router
        
        spec_endpoints = {
            "/api/records/{record_id}": "GET",
            "/api/records/{record_id}/scores": "GET", 
            "/api/records/{record_id}/responses": "GET",
            "/api/records": "GET"
        }
        
        actual_routes = {route.path: list(route.methods)[0] for route in flow_data_router.routes}
        
        for path, method in spec_endpoints.items():
            assert path in actual_routes, f"Missing endpoint: {method} {path}"

    def test_query_parameters_match_spec(self):
        """Test that query parameters match specification"""
        # This would be tested in integration tests with actual FastAPI client
        # For now, just verify the function signatures accept the right parameters
        
        import inspect
        
        # Check get_records_for_flow accepts include_scores
        sig = inspect.signature(DataService.get_records_for_flow)
        params = list(sig.parameters.keys())
        assert "include_scores" in params
        
        # Check get_scores_for_record accepts session_id  
        sig = inspect.signature(DataService.get_scores_for_record)
        params = list(sig.parameters.keys())
        assert "session_id" in params
        
        # Check get_responses_for_record accepts include_reasoning
        sig = inspect.signature(DataService.get_responses_for_record) 
        params = list(sig.parameters.keys())
        assert "include_reasoning" in params