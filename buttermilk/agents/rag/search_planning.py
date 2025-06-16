"""
Enhanced RAG search planning with LLM-driven query analysis.

This module provides intelligent search planning capabilities that use LLMs to:
1. Analyze user queries to understand intent
2. Plan optimal search strategies across multiple content fields
3. Execute targeted searches with metadata filtering
4. Synthesize and rank results intelligently
"""

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class SearchStrategy(str, Enum):
    """Available search strategies for the enhanced RAG system."""
    
    SEMANTIC = "semantic"           # Vector similarity across all content
    TITLE = "title"                # Search only document titles
    SUMMARY = "summary"            # Search only document summaries  
    CONTENT = "content"            # Search only main document content
    METADATA = "metadata"          # Search using metadata filters
    HYBRID = "hybrid"              # Combine multiple search types
    MULTI_FIELD = "multi_field"    # Search across specific field combinations


class ContentType(str, Enum):
    """Content types for targeted search."""
    
    TITLE = "title"
    SUMMARY = "summary"
    CONTENT = "content"
    METADATA = "metadata"
    ALL = "all"


class SearchPlan(BaseModel):
    """
    Represents an intelligent search plan created by LLM analysis.
    
    This plan guides how to search across different content fields and metadata
    to find the most relevant information for a user query.
    """
    
    query: str = Field(description="Original user query")
    intent: str = Field(description="LLM-analyzed query intent")
    primary_strategy: SearchStrategy = Field(description="Main search approach to use")
    secondary_strategies: List[SearchStrategy] = Field(
        default_factory=list,
        description="Additional search strategies to try"
    )
    content_types: List[ContentType] = Field(
        default_factory=lambda: [ContentType.ALL],
        description="Types of content to prioritize"
    )
    metadata_filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata filters to apply (e.g., case_number, date)"
    )
    search_terms: List[str] = Field(
        default_factory=list,
        description="Refined search terms extracted from query"
    )
    expected_result_types: List[str] = Field(
        default_factory=list,
        description="Expected types of results (e.g., 'policy', 'case_study')"
    )
    max_results_per_strategy: int = Field(
        default=5,
        description="Maximum results to return per search strategy"
    )
    confidence_threshold: float = Field(
        default=0.5,
        description="Minimum similarity threshold for results"
    )


class SearchResult(BaseModel):
    """Individual search result with metadata and scoring."""
    
    document_id: str = Field(description="Document identifier")
    chunk_id: str = Field(description="Chunk identifier")
    content: str = Field(description="Content text")
    content_type: str = Field(description="Type of content (title, summary, content)")
    similarity_score: float = Field(description="Vector similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    search_strategy: SearchStrategy = Field(description="Strategy that found this result")
    rank: Optional[int] = Field(default=None, description="LLM-assigned rank")
    relevance_explanation: Optional[str] = Field(
        default=None,
        description="LLM explanation of why this result is relevant"
    )


class SearchResults(BaseModel):
    """Collection of search results with synthesis metadata."""
    
    query: str = Field(description="Original query")
    plan: SearchPlan = Field(description="Search plan that was executed")
    results: List[SearchResult] = Field(description="All search results")
    total_found: int = Field(description="Total number of results found")
    strategies_used: List[SearchStrategy] = Field(description="Strategies that were executed")
    synthesis_summary: Optional[str] = Field(
        default=None,
        description="LLM-generated summary of findings"
    )
    key_themes: List[str] = Field(
        default_factory=list,
        description="Main themes identified across results"
    )
    confidence_score: float = Field(
        default=0.0,
        description="Overall confidence in result quality"
    )


class QueryAnalysis(BaseModel):
    """LLM analysis of user query for search planning."""
    
    query: str = Field(description="Original user query")
    intent: str = Field(description="Identified user intent")
    query_type: str = Field(description="Type of query (factual, analytical, exploratory)")
    key_concepts: List[str] = Field(description="Main concepts in the query")
    suggested_strategies: List[SearchStrategy] = Field(description="Recommended search strategies")
    metadata_hints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Suggested metadata filters"
    )
    reformulated_queries: List[str] = Field(
        default_factory=list,
        description="Alternative query formulations"
    )
    expected_answer_type: str = Field(description="Expected type of answer")


# Query planning prompts
QUERY_ANALYSIS_PROMPT = """
Analyze this user query for intelligent search planning:

Query: "{query}"

Available search capabilities:
- semantic: Vector similarity search across all content
- title: Search document titles only
- summary: Search document summaries only  
- content: Search main document content only
- metadata: Search using metadata filters (case_number, date, category, etc.)
- hybrid: Combine multiple search approaches
- multi_field: Search across specific field combinations

Available content types: title, summary, content, metadata
Available metadata fields: title, case_number, url, summary, category, date

Analyze the query and return a JSON object with:
{{
    "intent": "Brief description of what the user wants to know",
    "query_type": "factual|analytical|exploratory|comparative",
    "key_concepts": ["concept1", "concept2"],
    "suggested_strategies": ["strategy1", "strategy2"],
    "metadata_hints": {{"field": "value"}},
    "reformulated_queries": ["alternative query 1", "alternative query 2"],
    "expected_answer_type": "policy|case_study|statistics|overview|recommendation"
}}

Focus on understanding the user's intent and recommending the most effective search approach.
"""

RESULT_SYNTHESIS_PROMPT = """
Synthesize and rank these search results for the user query:

Query: "{query}"
Search strategies used: {strategies}

Results:
{results}

Provide a JSON response with:
{{
    "synthesis_summary": "Brief summary of what was found",
    "key_themes": ["theme1", "theme2", "theme3"],
    "ranked_results": [
        {{
            "result_index": 0,
            "rank": 1,
            "relevance_explanation": "Why this result is most relevant"
        }}
    ],
    "confidence_score": 0.85,
    "missing_information": "What important information might be missing"
}}

Rank results by relevance to the query, not just similarity scores.
"""


def create_search_plan_from_analysis(query: str, analysis: QueryAnalysis) -> SearchPlan:
    """Create a SearchPlan from QueryAnalysis results."""
    
    return SearchPlan(
        query=query,
        intent=analysis.intent,
        primary_strategy=analysis.suggested_strategies[0] if analysis.suggested_strategies else SearchStrategy.SEMANTIC,
        secondary_strategies=analysis.suggested_strategies[1:3],  # Take up to 2 additional strategies
        content_types=[ContentType.ALL],  # Can be refined based on analysis
        metadata_filters=analysis.metadata_hints,
        search_terms=analysis.key_concepts,
        expected_result_types=[analysis.expected_answer_type],
        max_results_per_strategy=5,
        confidence_threshold=0.5
    )