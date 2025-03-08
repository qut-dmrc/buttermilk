import pytest

from buttermilk.runner.varmap import FlowVariableRouter


class TestFlowVariableRouter:
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing variable routing with multiple outputs per step"""
        # Create a router with pre-populated data
        router = FlowVariableRouter()

        # Add judge step data (multiple outputs)
        router.step_outputs["judge"] = [
            {
                "answer": "This is the judge's answer",
                "scores": [
                    {"name": "quality", "score": 0.9, "explanation": "High quality"},
                    {
                        "name": "relevance",
                        "score": 0.7,
                        "explanation": "Somewhat relevant",
                    },
                    {
                        "name": "correctness",
                        "score": 0.85,
                        "explanation": "Mostly correct",
                    },
                ],
                "metadata": {
                    "version": "1.0",
                    "timestamp": "2025-03-08T12:00:00Z",
                },
            },
            # Second judge output with different data
            {
                "answer": "Alternative judge answer",
                "scores": [
                    {"name": "accuracy", "score": 0.95, "explanation": "Very accurate"},
                ],
                "metadata": {
                    "version": "1.1",
                    "timestamp": "2025-03-08T12:05:00Z",
                },
            },
        ]

        # Add draft step data
        router.step_outputs["draft"] = [
            {
                "answer": "This is the draft answer",
                "feedback": "Needs improvement in clarity",
                "revision_count": 2,
                "sections": ["intro", "body", "conclusion"],
            },
        ]

        # Add context step data
        router.step_outputs["context"] = [
            {
                "source": "user query",
                "timestamp": "2025-03-08T11:55:00Z",
                "query_id": "q123456",
                "relevant_docs": ["doc1", "doc2"],
            },
        ]

        # Add a simple scalar value wrapped in a list
        router.step_outputs["job_id"] = ["job-987654"]

        return router

    def test_simple_path_resolution(self, sample_data):
        """Test resolving simple paths with multiple outputs per step"""
        mappings = {
            "judge_answer": "judge.answer",
            "draft_feedback": "draft.feedback",
            "job": "job_id",
        }

        result = sample_data._resolve_mappings(mappings)

        # Should return a list of all judges' answers
        assert isinstance(result["judge_answer"], list)
        assert len(result["judge_answer"]) == 2
        assert "This is the judge's answer" in result["judge_answer"]
        assert "Alternative judge answer" in result["judge_answer"]

        # Should return a list with one draft feedback
        assert result["draft_feedback"] == ["Needs improvement in clarity"]

        # For scalar values in lists, returns the whole list
        assert result["job"] == ["job-987654"]

    def test_whole_step_resolution(self, sample_data):
        """Test resolving an entire step output with multiple outputs per step"""
        mappings = {
            "full_context": "context",
            "full_judge": "judge",
        }

        result = sample_data._resolve_mappings(mappings)

        # Should return the lists of outputs
        assert isinstance(result["full_judge"], list)
        assert len(result["full_judge"]) == 2
        assert result["full_judge"][0]["answer"] == "This is the judge's answer"
        assert result["full_judge"][1]["answer"] == "Alternative judge answer"

        assert isinstance(result["full_context"], list)
        assert result["full_context"][0]["source"] == "user query"

    def test_nested_path_resolution(self, sample_data):
        """Test resolving nested paths"""
        mappings = {
            "judge_metadata_version": "judge.metadata.version",
            "first_score": "judge.scores[0].name",
        }

        result = sample_data._resolve_mappings(mappings)

        # Should return values from all judge outputs
        assert isinstance(result["judge_metadata_version"], list)
        assert result["judge_metadata_version"] == ["1.0", "1.1"]

        # Should return first score name from each judge
        assert isinstance(result["first_score"], list)
        assert result["first_score"] == ["quality", "accuracy"]

    def test_nested_mappings(self, sample_data):
        """Test resolving nested mapping structures"""
        mappings = {
            "analysis": {
                "score_summary": "judge.scores",
                "feedback": "draft.feedback",
                "metadata": {
                    "timestamp": "context.timestamp",
                    "version": "judge.metadata.version",
                },
            },
        }

        result = sample_data._resolve_mappings(mappings)

        # Should include scores from both judges
        assert isinstance(result["analysis"]["score_summary"], list)
        assert (
            len(result["analysis"]["score_summary"]) == 4
        )  # 3 from first judge + 1 from second

        # Should have feedback as a list
        assert result["analysis"]["feedback"] == ["Needs improvement in clarity"]

        # Should have timestamp as a list
        assert result["analysis"]["metadata"]["timestamp"] == ["2025-03-08T11:55:00Z"]

        # Should have both versions
        assert result["analysis"]["metadata"]["version"] == ["1.0", "1.1"]

    def test_list_aggregation(self, sample_data):
        """Test aggregating values in lists"""
        mappings = {
            "combined_answers": [
                "judge.answer",
                "draft.answer",
            ],
        }

        result = sample_data._resolve_mappings(mappings)

        # First element should be a list of all judge answers
        assert isinstance(result["combined_answers"][0], list)
        assert len(result["combined_answers"][0]) == 2
        assert "This is the judge's answer" in result["combined_answers"][0]
        assert "Alternative judge answer" in result["combined_answers"][0]

        # Second element should be a list with one draft answer
        assert result["combined_answers"][1] == ["This is the draft answer"]

    def test_complex_list_aggregation(self, sample_data):
        """Test aggregating complex values including nested mappings in lists"""
        mappings = {
            "complex_data": [
                "judge.answer",
                {"version": "judge.metadata.version"},
                "draft.sections",
            ],
        }

        result = sample_data._resolve_mappings(mappings)

        assert len(result["complex_data"]) == 3
        # First element should be a list of judge answers
        assert isinstance(result["complex_data"][0], list)
        assert len(result["complex_data"][0]) == 2
        assert "This is the judge's answer" in result["complex_data"][0]

        # Second element should have versions from both judges
        assert result["complex_data"][1] == {"version": ["1.0", "1.1"]}

        # Third element should be the draft sections
        assert result["complex_data"][2] == ["intro", "body", "conclusion"]

    def test_jmespath_expressions(self, sample_data):
        """Test using JMESPath expressions for complex queries"""
        mappings = {
            "high_scores": "judge.scores[?score > `0.8`]",
            "score_names": "judge.scores[].name",
            "first_section": "draft.sections[0]",
        }

        result = sample_data._resolve_mappings(mappings)

        # Should have high scores from both judges
        assert isinstance(result["high_scores"], list)
        assert len(result["high_scores"]) == 3  # 2 from first judge + 1 from second

        # Names should be collected from all scores
        assert isinstance(result["score_names"], list)
        assert len(result["score_names"]) == 4  # 3 from first judge + 1 from second
        assert "quality" in result["score_names"]
        assert "accuracy" in result["score_names"]

        # First section should be a list with a single value
        assert result["first_section"] == ["intro"]  # Not 'intro'

    def test_nonexistent_path(self, sample_data):
        """Test handling of nonexistent paths"""
        mappings = {
            "missing_step": "nonexistent.field",
            "missing_field": "judge.nonexistent",
            "valid_field": "judge.answer",
        }

        result = sample_data._resolve_mappings(mappings)

        assert result["missing_step"] is None
        assert result["missing_field"] is None
        assert isinstance(result["valid_field"], list)
        assert len(result["valid_field"]) == 2

    def test_multiple_outputs_all_matches(self, sample_data):
        """Test that all matching outputs are returned when there are multiple outputs"""
        # Add another step with multiple outputs where both have field1 but only second has field2
        router = sample_data
        router.step_outputs["special"] = [
            {"field1": "value1"},
            {"field1": "value1-alt", "field2": "value2"},
        ]

        mappings = {
            "field2_value": "special.field2",
            "field1_value": "special.field1",
        }

        result = router._resolve_mappings(mappings)

        # Should return the field from the second output since the first doesn't have it
        assert result["field2_value"] == ["value2"]

        # Should return field1 from both outputs
        assert isinstance(result["field1_value"], list)
        assert len(result["field1_value"]) == 2
        assert "value1" in result["field1_value"]
        assert "value1-alt" in result["field1_value"]

    def test_collect_all_matching_fields(self, sample_data):
        """Test collecting all matching fields from multiple outputs"""
        # Create a router with data that has multiple agents with similar field structures
        router = sample_data

        # Ensure judge step has outputs with scores
        assert len(router.step_outputs["judge"]) == 2
        assert "scores" in router.step_outputs["judge"][0]
        assert "scores" in router.step_outputs["judge"][1]

        mappings = {
            # This should collect all score values from all judge outputs
            "all_scores": "judge.scores[].score",
            # This should collect all quality scores where they exist
            "quality_scores": "judge.scores[?name=='quality'].score",
        }

        result = router._resolve_mappings(mappings)

        # Should have scores from both judge outputs flattened into a single list
        assert isinstance(result["all_scores"], list)
        assert len(result["all_scores"]) == 4  # 3 from first output + 1 from second
        assert 0.9 in result["all_scores"]
        assert 0.7 in result["all_scores"]
        assert 0.85 in result["all_scores"]
        assert 0.95 in result["all_scores"]

        # Should only have quality scores where that name exists
        assert result["quality_scores"] == [
            0.9,
        ]  # Only the first judge has a "quality" score

    def test_score_agg(self, sample_data):
        mappings = {"draft": "judge.scores[].score"}
        result = sample_data._resolve_mappings(mappings)
        assert set(result["draft"]) == {0.9, 0.7, 0.85, 0.95}
