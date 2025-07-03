"""
Enhanced search capabilities for intelligent RAG systems.

This module provides advanced search tools that can:
1. Execute multi-field searches across different content types
2. Apply intelligent metadata filtering
3. Combine results from multiple search strategies
4. Use LLM reasoning for result synthesis and ranking
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union

from buttermilk._core.log import logger
from buttermilk.agents.rag.search_planning import (
    ContentType,
    SearchPlan,
    SearchResult,
    SearchResults,
    SearchStrategy,
    QueryAnalysis,
    QUERY_ANALYSIS_PROMPT,
    RESULT_SYNTHESIS_PROMPT,
    create_search_plan_from_analysis,
)


class EnhancedVectorSearch:
    """
    Enhanced vector search capabilities with LLM-driven planning and synthesis.
    
    This class wraps a ChromaDB vectorstore to provide intelligent search
    capabilities that go beyond simple vector similarity.
    """
    
    def __init__(self, vectorstore, llm_client=None):
        """
        Initialize enhanced search with vectorstore and LLM client.
        
        Args:
            vectorstore: ChromaDBEmbeddings instance with multi-field capabilities
            llm_client: LLM client for query planning and result synthesis
        """
        self.vectorstore = vectorstore
        self.llm_client = llm_client
        self.collection = vectorstore.collection
        
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Use LLM to analyze query and understand search intent.
        
        Args:
            query: User's search query
            
        Returns:
            QueryAnalysis with intent, strategies, and metadata hints
        """
        if not self.llm_client:
            # Fallback to basic analysis if no LLM available
            return QueryAnalysis(
                query=query,
                intent="Basic search query",
                query_type="factual",
                key_concepts=[query],
                suggested_strategies=[SearchStrategy.SEMANTIC],
                expected_answer_type="information"
            )
        
        try:
            prompt = QUERY_ANALYSIS_PROMPT.format(query=query)
            response = await self.llm_client.generate_async(prompt)
            
            # Parse LLM response
            analysis_data = json.loads(response)
            analysis_data["query"] = query
            
            return QueryAnalysis(**analysis_data)
            
        except Exception as e:
            logger.warning(f"Query analysis failed, using fallback: {e}")
            return QueryAnalysis(
                query=query,
                intent="Search query analysis failed",
                query_type="factual", 
                key_concepts=[query],
                suggested_strategies=[SearchStrategy.SEMANTIC],
                expected_answer_type="information"
            )
    
    async def create_search_plan(self, query: str) -> SearchPlan:
        """
        Create an intelligent search plan based on query analysis.
        
        Args:
            query: User's search query
            
        Returns:
            SearchPlan with strategies and targeting information
        """
        analysis = await self.analyze_query(query)
        return create_search_plan_from_analysis(query, analysis)
    
    async def semantic_search(
        self, 
        query: str, 
        n_results: int = 5,
        content_type_filter: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform semantic vector search across all content.
        
        Args:
            query: Search query
            n_results: Maximum number of results
            content_type_filter: Filter by content type (title, summary, content)
            metadata_filters: Additional metadata filters
            
        Returns:
            List of SearchResult objects
        """
        where_clause = {}
        
        if content_type_filter:
            where_clause["content_type"] = content_type_filter
            
        if metadata_filters:
            where_clause.update(metadata_filters)
        
        # Perform ChromaDB query
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause if where_clause else None,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to SearchResult objects
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, (doc_id, doc, metadata, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0], 
                results["metadatas"][0],
                results["distances"][0]
            )):
                search_results.append(SearchResult(
                    document_id=metadata.get("document_id", doc_id),
                    chunk_id=doc_id,
                    content=doc,
                    content_type=metadata.get("content_type", "unknown"),
                    similarity_score=1 - distance,  # Convert distance to similarity
                    metadata=metadata,
                    search_strategy=SearchStrategy.SEMANTIC
                ))
        
        return search_results
    
    async def field_search(
        self,
        query: str,
        content_type: str,
        n_results: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search within a specific content field (title, summary, content).
        
        Args:
            query: Search query
            content_type: Type of content to search (title, summary, content)
            n_results: Maximum number of results
            metadata_filters: Additional metadata filters
            
        Returns:
            List of SearchResult objects from the specified field
        """
        return await self.semantic_search(
            query=query,
            n_results=n_results,
            content_type_filter=content_type,
            metadata_filters=metadata_filters
        )
    
    async def metadata_search(
        self,
        query: str,
        metadata_filters: Dict[str, Any],
        n_results: int = 5
    ) -> List[SearchResult]:
        """
        Search using metadata filters with semantic ranking.
        
        Args:
            query: Search query for semantic ranking
            metadata_filters: Metadata filters to apply
            n_results: Maximum number of results
            
        Returns:
            List of SearchResult objects matching metadata filters
        """
        return await self.semantic_search(
            query=query,
            n_results=n_results,
            metadata_filters=metadata_filters
        )
    
    async def hybrid_search(
        self,
        query: str,
        n_results: int = 10,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search across multiple content types and combine results.
        
        Args:
            query: Search query
            n_results: Total maximum number of results
            metadata_filters: Additional metadata filters
            
        Returns:
            Combined and deduplicated list of SearchResult objects
        """
        # Search across different content types
        tasks = [
            self.field_search(query, "title", n_results // 3, metadata_filters),
            self.field_search(query, "summary", n_results // 3, metadata_filters),
            self.field_search(query, "content", n_results // 3, metadata_filters),
        ]
        
        results_lists = await asyncio.gather(*tasks)
        
        # Combine and deduplicate results
        all_results = []
        seen_chunk_ids = set()
        
        for results_list in results_lists:
            for result in results_list:
                if result.chunk_id not in seen_chunk_ids:
                    result.search_strategy = SearchStrategy.HYBRID
                    all_results.append(result)
                    seen_chunk_ids.add(result.chunk_id)
        
        # Sort by similarity score and limit results
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return all_results[:n_results]
    
    async def execute_search_plan(self, plan: SearchPlan) -> SearchResults:
        """
        Execute a complete search plan with multiple strategies.
        
        Args:
            plan: SearchPlan to execute
            
        Returns:
            SearchResults with all results and synthesis
        """
        all_results = []
        strategies_used = []
        
        # Execute primary strategy
        primary_results = await self._execute_strategy(
            plan.primary_strategy,
            plan.query,
            plan.max_results_per_strategy,
            plan.metadata_filters
        )
        all_results.extend(primary_results)
        strategies_used.append(plan.primary_strategy)
        
        # Execute secondary strategies
        for strategy in plan.secondary_strategies:
            try:
                secondary_results = await self._execute_strategy(
                    strategy,
                    plan.query,
                    plan.max_results_per_strategy // 2,  # Fewer results for secondary
                    plan.metadata_filters
                )
                all_results.extend(secondary_results)
                strategies_used.append(strategy)
            except Exception as e:
                logger.warning(f"Secondary strategy {strategy} failed: {e}")
        
        # Deduplicate results
        unique_results = self._deduplicate_results(all_results)
        
        # Filter by confidence threshold
        filtered_results = [
            r for r in unique_results 
            if r.similarity_score >= plan.confidence_threshold
        ]
        
        # Create SearchResults object
        search_results = SearchResults(
            query=plan.query,
            plan=plan,
            results=filtered_results,
            total_found=len(filtered_results),
            strategies_used=strategies_used
        )
        
        # Synthesize results with LLM if available
        if self.llm_client and filtered_results:
            await self._synthesize_results(search_results)
        
        return search_results
    
    async def _execute_strategy(
        self,
        strategy: SearchStrategy,
        query: str,
        n_results: int,
        metadata_filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Execute a single search strategy."""
        
        if strategy == SearchStrategy.SEMANTIC:
            return await self.semantic_search(query, n_results, metadata_filters=metadata_filters)
        elif strategy == SearchStrategy.TITLE:
            return await self.field_search(query, "title", n_results, metadata_filters)
        elif strategy == SearchStrategy.SUMMARY:
            return await self.field_search(query, "summary", n_results, metadata_filters)
        elif strategy == SearchStrategy.CONTENT:
            return await self.field_search(query, "content", n_results, metadata_filters)
        elif strategy == SearchStrategy.METADATA:
            return await self.metadata_search(query, metadata_filters or {}, n_results)
        elif strategy == SearchStrategy.HYBRID:
            return await self.hybrid_search(query, n_results, metadata_filters)
        else:
            # Fallback to semantic search
            return await self.semantic_search(query, n_results, metadata_filters=metadata_filters)
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on chunk_id, keeping highest scoring."""
        seen = {}
        for result in results:
            if result.chunk_id not in seen or result.similarity_score > seen[result.chunk_id].similarity_score:
                seen[result.chunk_id] = result
        
        return list(seen.values())
    
    async def _synthesize_results(self, search_results: SearchResults) -> None:
        """Use LLM to synthesize and rank results."""
        if not self.llm_client:
            return
        
        try:
            # Prepare results summary for LLM
            results_summary = []
            for i, result in enumerate(search_results.results[:10]):  # Limit for prompt size
                results_summary.append({
                    "index": i,
                    "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    "content_type": result.content_type,
                    "similarity": f"{result.similarity_score:.3f}",
                    "metadata": {k: v for k, v in result.metadata.items() if k in ["title", "case_number"]}
                })
            
            prompt = RESULT_SYNTHESIS_PROMPT.format(
                query=search_results.query,
                strategies=[s.value for s in search_results.strategies_used],
                results=json.dumps(results_summary, indent=2)
            )
            
            response = await self.llm_client.generate_async(prompt)
            synthesis_data = json.loads(response)
            
            # Update SearchResults with synthesis
            search_results.synthesis_summary = synthesis_data.get("synthesis_summary")
            search_results.key_themes = synthesis_data.get("key_themes", [])
            search_results.confidence_score = synthesis_data.get("confidence_score", 0.0)
            
            # Apply LLM ranking
            ranked_results = synthesis_data.get("ranked_results", [])
            for rank_info in ranked_results:
                result_idx = rank_info.get("result_index")
                if result_idx < len(search_results.results):
                    search_results.results[result_idx].rank = rank_info.get("rank")
                    search_results.results[result_idx].relevance_explanation = rank_info.get("relevance_explanation")
            
            # Sort by LLM rank if available
            ranked_items = [r for r in search_results.results if r.rank is not None]
            unranked_items = [r for r in search_results.results if r.rank is None]
            ranked_items.sort(key=lambda x: x.rank)
            
            search_results.results = ranked_items + unranked_items
            
        except Exception as e:
            logger.warning(f"Result synthesis failed: {e}")
            # Set basic synthesis info
            search_results.synthesis_summary = f"Found {len(search_results.results)} relevant results"
            search_results.confidence_score = 0.7