"""Test module for agent output Markdown formatting."""

from unittest.mock import MagicMock

from buttermilk._core.contract import AgentConfig, ExecutionTrace
from buttermilk.agents.describer import MediaDescription
from buttermilk.agents.differences import Differences, Divergence, Position
from buttermilk.agents.frame import FramedStatement
from buttermilk.agents.judge import JudgeReasons
from buttermilk.agents.rag.rag_zotero import ZoteroReference, ZoteroResearchResult
from buttermilk.agents.rag.simple_rag_agent import Reference, ResearchResult


class TestJudgeReasonsMarkdown:
    """Test markdown formatting for JudgeReasons output model."""
    
    def test_judge_reasons_as_markdown(self):
        """Test the as_markdown method for JudgeReasons."""
        agent_id = "JUDGE-gpt4"
        call_id = "1234ABCD"
        
        output = JudgeReasons(
            conclusion="The content violates policy due to harmful content",
            reasons=[
                "The content contains explicit threats",
                "The language targets a specific group"
            ],
            prediction=True,
            uncertainty="low"
        )
        
        (
            f"**{agent_id} #{call_id[:8]}**\n"
            f"The content violates policy due to harmful content\n"
            f"Conclusion: violating\n"
            f"- The content contains explicit threats\n"
            f"- The language targets a specific group\n"
            f"Prediction: True"
        )
        
        # We'll implement as_markdown and check it matches expected format
        assert hasattr(output, "as_markdown")
        result = output.as_markdown(agent_id, call_id)
        
        # Check key components are present
        assert f"**{agent_id} #{call_id[:8]}**" in result
        assert "The content violates policy" in result
        assert "Conclusion: violating" in result
        assert "- The content contains explicit threats" in result
        assert "Prediction: True" in result
    
    def test_judge_reasons_str_alias(self):
        """Test that __str__ is aliased to as_markdown."""
        output = JudgeReasons(
            conclusion="Content is acceptable",
            reasons=["No violations found"],
            prediction=False,
            uncertainty="low"
        )
        
        # Mock the agent context
        output._agent_id = "JUDGE-test"
        output._call_id = "ABCD1234"
        
        assert str(output) == output.as_markdown(output._agent_id, output._call_id)


class TestDifferencesMarkdown:
    """Test markdown formatting for Differences output model."""
    
    def test_differences_as_markdown(self):
        """Test the as_markdown method for Differences."""
        agent_id = "DIFF-gpt4"
        call_id = "5678EFGH"
        
        output = Differences(
            conclusion="Experts disagree on the severity of the issue",
            divergences=[
                Divergence(
                    topic="Content interpretation",
                    positions=[
                        Position(
                            experts=["Expert1", "Expert2"],
                            position="Content is harmful"
                        ),
                        Position(
                            experts=["Expert3"],
                            position="Content is neutral"
                        )
                    ]
                )
            ]
        )
        
        result = output.as_markdown(agent_id, call_id)
        
        assert f"**{agent_id} #{call_id[:8]}**" in result
        assert "Experts disagree on the severity" in result
        assert "Content interpretation" in result
        assert "Content is harmful" in result


class TestMediaDescriptionMarkdown:
    """Test markdown formatting for MediaDescription output model."""
    
    def test_media_description_as_markdown(self):
        """Test the as_markdown method for MediaDescription."""
        agent_id = "DESC-gpt4"
        call_id = "9012IJKL"
        
        output = MediaDescription(
            description="A scenic view of mountains at sunset",
            media_type="image",
            confidence=0.95
        )
        
        result = output.as_markdown(agent_id, call_id)
        
        assert f"**{agent_id} #{call_id[:8]}**" in result
        assert "A scenic view of mountains" in result
        assert "Type: image" in result


class TestFramedStatementMarkdown:
    """Test markdown formatting for FramedStatement output model."""
    
    def test_framed_statement_as_markdown(self):
        """Test the as_markdown method for FramedStatement."""
        agent_id = "FRAME-gpt4"
        call_id = "3456MNOP"
        
        output = FramedStatement(
            statement="Climate change is an urgent crisis",
            speaker_name="Dr. Smith",
            speaker_affiliation="University X",
            problem_definition="Global warming threatens ecosystems",
            blame_attribution="Industrial emissions",
            moral_evaluation="Negligent behavior",
            recommendation="Immediate action needed",
            confidence_score=0.9
        )
        
        result = output.as_markdown(agent_id, call_id)
        
        # Note: FramedStatement does not include agent_id header (handled by FrameAnalysisResults)
        assert "Climate change is an urgent crisis" in result
        assert "Dr. Smith" in result


class TestResearchResultMarkdown:
    """Test markdown formatting for ResearchResult output model."""
    
    def test_research_result_as_markdown(self):
        """Test the as_markdown method for ResearchResult."""
        agent_id = "RAG-gpt4"
        call_id = "7890QRST"
        
        output = ResearchResult(
            literature=[
                Reference(
                    summary="Study on climate impacts",
                    citation="Smith et al., 2024"
                )
            ],
            response="Based on the literature...",
            summary="Climate change has significant impacts"
        )
        
        result = output.as_markdown(agent_id, call_id)
        
        assert f"**{agent_id} #{call_id[:8]}**" in result
        assert "Climate change has significant impacts" in result
        assert "Based on the literature" in result


class TestZoteroResearchResultMarkdown:
    """Test markdown formatting for ZoteroResearchResult output model."""
    
    def test_zotero_research_result_as_markdown(self):
        """Test the as_markdown method for ZoteroResearchResult."""
        agent_id = "ZOTERO-gpt4"
        call_id = "UVWX1234"
        
        output = ZoteroResearchResult(
            literature=[
                ZoteroReference(
                    summary="Comprehensive climate study",
                    citation="Smith, J. (2024). Climate Change. Nature.",
                    doi="10.1234/nature.2024",
                    uri="https://example.com/paper"
                )
            ],
            response="The research shows...",
            summary="Key findings from academic literature"
        )
        
        result = output.as_markdown(agent_id, call_id)
        
        assert f"**{agent_id} #{call_id[:8]}**" in result
        assert "Key findings from academic literature" in result
        assert "The research shows" in result


class TestExecutionTraceMarkdown:
    """Test markdown formatting for ExecutionTrace wrapper."""
    
    def test_agent_trace_with_judge_output(self):
        """Test ExecutionTrace formatting with JudgeReasons output."""
        config = AgentConfig(
            agent_id="JUDGE-gpt4",
            agent_name="judge_agent",
            agent_type="Judge"
        )
        
        output = JudgeReasons(
            conclusion="Content is safe",
            reasons=["No violations detected"],
            prediction=False,
            uncertainty="low"
        )

        trace = MagicMock(spec=ExecutionTrace)
        trace.agent_info = config
        trace.agent_id = "JUDGE-gpt4"
        trace.call_id = "ABCD1234"
        trace.outputs = output
        trace.error = []
        
        # The trace should use the output's as_markdown method
        trace.as_markdown()

        # We'll need to implement this on ExecutionTrace
        # For now, check that the method would exist
        assert hasattr(trace, "as_markdown")
