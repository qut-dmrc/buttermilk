"""Test the simplified RagAgent with structured outputs."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from buttermilk.agents.rag.simple_rag_agent import RagAgent, ResearchResult, Reference
from buttermilk.agents.rag.rag_zotero import RagZotero, ZoteroResearchResult, ZoteroReference


class TestSimpleRagAgent:
    """Test the simplified RagAgent implementation."""
    
    def test_rag_agent_forces_structured_output(self):
        """Test that RagAgent sets the correct output model."""
        agent = RagAgent(
            agent_name="test_rag",
            role="RESEARCHER",
            parameters={}
        )
        
        assert agent._output_model == ResearchResult
        assert agent.parameters.get('template') == 'research_synthesis_with_citations'
    
    def test_rag_agent_custom_template(self):
        """Test that RagAgent respects custom template parameter."""
        agent = RagAgent(
            agent_name="test_rag", 
            role="RESEARCHER",
            parameters={'template': 'custom_template'}
        )
        
        assert agent.parameters.get('template') == 'custom_template'
    
    def test_research_result_structure(self):
        """Test ResearchResult model structure."""
        result = ResearchResult(
            literature=[
                Reference(
                    summary="Test finding",
                    source="Test Document (ID: test123)"
                )
            ],
            response="Test synthesis"
        )
        
        assert len(result.literature) == 1
        assert result.literature[0].summary == "Test finding"
        assert result.response == "Test synthesis"


class TestRagZotero:
    """Test the RagZotero implementation."""
    
    def test_rag_zotero_forces_zotero_output(self):
        """Test that RagZotero uses ZoteroResearchResult."""
        agent = RagZotero(
            agent_name="test_zotero",
            role="ZOTERO_RESEARCHER", 
            parameters={}
        )
        
        assert agent._output_model == ZoteroResearchResult
        assert agent.parameters.get('template') == 'research_synthesis_with_citations'
    
    def test_zotero_result_structure(self):
        """Test ZoteroResearchResult with academic citations."""
        result = ZoteroResearchResult(
            literature=[
                ZoteroReference(
                    summary="Social media impacts mental health",
                    source="Smith et al., 2023",
                    citation="Smith, J., Brown, A., & Davis, C. (2023). The impact of social media. Journal of Psychology, 45(3), 234-251.",
                    doi="10.1037/jap.2023.045"
                )
            ],
            response="Research shows social media impacts..."
        )
        
        assert len(result.literature) == 1
        assert result.literature[0].citation == "Smith, J., Brown, A., & Davis, C. (2023). The impact of social media. Journal of Psychology, 45(3), 234-251."
        assert result.literature[0].doi == "10.1037/jap.2023.045"