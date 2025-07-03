"""
Enhanced RAG Agent with intelligent search planning and LLM-driven synthesis.

This agent provides advanced RAG capabilities that go beyond simple vector similarity:
1. LLM-driven query analysis and search planning
2. Multi-field search across titles, summaries, content, and metadata
3. Intelligent result synthesis and ranking
4. Configurable search strategies per use case
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from buttermilk.debug.error_capture import ErrorCapture, capture_enhanced_rag_errors, safe_isinstance_check

from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import AgentInput, AgentOutput, ToolOutput
from buttermilk.agents.rag.rag_agent import RagAgent
from buttermilk._core.log import logger
from buttermilk._core.types import Record, UserMessage
from buttermilk._core.mcp_decorators import tool
from buttermilk.agents.rag.search_planning import (
    SearchPlan,
    SearchResults,
    SearchStrategy,
    QueryAnalysis,
    SearchResult,
    QUERY_ANALYSIS_PROMPT,
    RESULT_SYNTHESIS_PROMPT,
    create_search_plan_from_analysis,
)


class EnhancedRagAgent(RagAgent):
    """
    Enhanced RAG Agent with intelligent search capabilities.

    This agent extends the base RagAgent to provide LLM-driven search strategies:
    - Analyze queries and understand user intent
    - Plan optimal search strategies across multiple content fields
    - Execute targeted searches with metadata filtering
    - Synthesize and rank results intelligently
    - Generate contextual responses based on findings
    
    The agent exposes multiple search strategies as tools that can be called
    independently or orchestrated through intelligent planning.
    """

    # Enhanced RAG configuration
    enable_query_planning: bool = Field(default=True, description="Enable LLM-driven query planning and analysis")
    enable_result_synthesis: bool = Field(default=True, description="Enable LLM-driven result synthesis and ranking")
    search_strategies: List[str] = Field(
        default_factory=lambda: ["semantic", "title", "summary", "hybrid"], description="Available search strategies for this agent"
    )
    max_search_rounds: int = Field(default=3, description="Maximum number of search strategies to try")
    planning_model: Optional[str] = Field(default=None, description="LLM model to use for query planning (defaults to main model)")
    synthesis_model: Optional[str] = Field(default=None, description="LLM model to use for result synthesis (defaults to main model)")
    confidence_threshold: float = Field(default=0.5, description="Minimum similarity threshold for search results")
    max_results_per_strategy: int = Field(default=5, description="Maximum results to return per search strategy")
    include_search_explanation: bool = Field(default=True, description="Include explanation of search strategy in response")

    def __init__(self, **data):
        super().__init__(**data)
        self._error_capture = ErrorCapture(capture_locals=True)
        # Override parent's output model to None for flexible responses
        self._output_model = None

    @tool(name="analyze_query", description="Analyze a query to understand search intent and suggest strategies")
    async def analyze_query(self, query: str) -> str:
        """Use LLM to analyze query and understand search intent.
        
        Args:
            query: User's search query
            
        Returns:
            JSON string with query analysis including intent, key concepts, and suggested strategies
        """
        await self.ensure_chromadb_ready()
        
        # Use parent's model client if available
        if hasattr(self, "_model_client") and self._model_client:
            try:
                prompt = QUERY_ANALYSIS_PROMPT.format(query=query)
                response = await self._model_client.generate_async(prompt)
                
                # Parse and validate response
                analysis_data = json.loads(response)
                analysis_data["query"] = query
                return json.dumps(analysis_data)
                
            except Exception as e:
                logger.warning(f"Query analysis failed, using fallback: {e}")
        
        # Fallback analysis
        return json.dumps({
            "query": query,
            "intent": "Basic search query",
            "query_type": "factual",
            "key_concepts": [query],
            "suggested_strategies": ["semantic"],
            "expected_answer_type": "information"
        })
            

    @tool(name="semantic_search", description="Perform semantic vector search across all content")
    async def semantic_search(
        self, 
        query: str, 
        n_results: int = 5,
        content_type: Optional[str] = None,
        metadata_filters: Optional[str] = None
    ) -> str:
        """Perform semantic vector search across all content.
        
        Args:
            query: Search query
            n_results: Maximum number of results
            content_type: Optional filter by content type (title, summary, content)
            metadata_filters: Optional JSON string of metadata filters
            
        Returns:
            JSON string with search results
        """
        await self.ensure_chromadb_ready()
        
        where_clause = {}
        if content_type:
            where_clause["content_type"] = content_type
            
        if metadata_filters:
            try:
                filters = json.loads(metadata_filters)
                where_clause.update(filters)
            except json.JSONDecodeError:
                logger.warning(f"Invalid metadata_filters JSON: {metadata_filters}")
        
        # Use parent's _query_db method
        tool_output = await self._query_db(query)
        
        # Apply additional filtering if needed
        if where_clause and tool_output.results:
            filtered_results = [
                r for r in tool_output.results
                if all(r.metadata.get(k) == v for k, v in where_clause.items())
            ]
            tool_output.results = filtered_results[:n_results]
            tool_output.content = "\n\n".join([str(r) for r in filtered_results])
        
        # Convert results to JSON for tool response
        results_data = [
            {
                "document_id": r.document_id,
                "chunk_id": r.id,
                "content": r.full_text[:500],  # Truncate for response
                "content_type": r.metadata.get("content_type", "unknown"),
                "metadata": r.metadata
            }
            for r in (tool_output.results or [])[:n_results]
        ]
        
        return json.dumps({
            "query": query,
            "results": results_data,
            "total_found": len(results_data)
        })
    
    @tool(name="field_search", description="Search within a specific content field (title, summary, or content)")
    async def field_search(
        self,
        query: str,
        content_type: str,
        n_results: int = 5,
        metadata_filters: Optional[str] = None
    ) -> str:
        """Search within a specific content field.
        
        Args:
            query: Search query
            content_type: Type of content to search (title, summary, content)
            n_results: Maximum number of results
            metadata_filters: Optional JSON string of metadata filters
            
        Returns:
            JSON string with search results from the specified field
        """
        return await self.semantic_search(
            query=query,
            n_results=n_results,
            content_type=content_type,
            metadata_filters=metadata_filters
        )
    
    @tool(name="hybrid_search", description="Perform hybrid search across multiple content types")
    async def hybrid_search(
        self,
        query: str,
        n_results: int = 10,
        metadata_filters: Optional[str] = None
    ) -> str:
        """Perform hybrid search across multiple content types and combine results.
        
        Args:
            query: Search query
            n_results: Total maximum number of results
            metadata_filters: Optional JSON string of metadata filters
            
        Returns:
            JSON string with combined and deduplicated search results
        """
        # Search across different content types
        tasks = [
            self.field_search(query, "title", n_results // 3, metadata_filters),
            self.field_search(query, "summary", n_results // 3, metadata_filters),
            self.field_search(query, "content", n_results // 3, metadata_filters),
        ]
        
        results_json_list = await asyncio.gather(*tasks)
        
        # Combine and deduplicate results
        all_results = []
        seen_chunk_ids = set()
        
        for results_json in results_json_list:
            results_data = json.loads(results_json)
            for result in results_data.get("results", []):
                if result["chunk_id"] not in seen_chunk_ids:
                    all_results.append(result)
                    seen_chunk_ids.add(result["chunk_id"])
        
        # Limit results
        all_results = all_results[:n_results]
        
        return json.dumps({
            "query": query,
            "results": all_results,
            "total_found": len(all_results),
            "search_type": "hybrid"
        })
    
    @tool(name="create_search_plan", description="Create an intelligent search plan based on query analysis")
    async def create_search_plan(self, query: str) -> str:
        """Create an intelligent search plan based on query analysis.
        
        Args:
            query: User's search query
            
        Returns:
            JSON string with search plan including strategies and metadata
        """
        # First analyze the query
        analysis_json = await self.analyze_query(query)
        analysis = json.loads(analysis_json)
        
        # Create search plan based on analysis
        plan = {
            "query": query,
            "intent": analysis.get("intent", "Basic search"),
            "primary_strategy": analysis.get("suggested_strategies", ["semantic"])[0],
            "secondary_strategies": analysis.get("suggested_strategies", [])[1:3],  # Up to 2 secondary
            "key_concepts": analysis.get("key_concepts", []),
            "metadata_filters": {},  # Could be enhanced based on query type
            "max_results_per_strategy": self.max_results_per_strategy,
            "confidence_threshold": self.confidence_threshold
        }
        
        # Add metadata filters based on query type
        if "legal" in analysis.get("query_type", "").lower():
            plan["metadata_filters"]["category"] = "legal"
        elif "technical" in analysis.get("query_type", "").lower():
            plan["metadata_filters"]["category"] = "technical"
            
        return json.dumps(plan)
    
    @tool(name="execute_search_plan", description="Execute a complete search plan with multiple strategies")
    async def execute_search_plan(self, plan_json: str) -> str:
        """Execute a complete search plan with multiple strategies.
        
        Args:
            plan_json: JSON string containing the search plan
            
        Returns:
            JSON string with combined results from all strategies
        """
        plan = json.loads(plan_json)
        query = plan.get("query", "")
        primary_strategy = plan.get("primary_strategy", "semantic")
        secondary_strategies = plan.get("secondary_strategies", [])
        metadata_filters = json.dumps(plan.get("metadata_filters", {})) if plan.get("metadata_filters") else None
        max_results = plan.get("max_results_per_strategy", self.max_results_per_strategy)
        
        all_results = []
        strategies_used = []
        
        # Execute primary strategy
        if primary_strategy == "hybrid":
            primary_json = await self.hybrid_search(query, max_results, metadata_filters)
        else:
            primary_json = await self.semantic_search(
                query,
                max_results,
                content_type=primary_strategy if primary_strategy in ["title", "summary", "content"] else None,
                metadata_filters=metadata_filters
            )
        
        primary_data = json.loads(primary_json)
        all_results.extend(primary_data.get("results", []))
        strategies_used.append(primary_strategy)
        
        # Execute secondary strategies
        for strategy in secondary_strategies:
            try:
                if strategy == "hybrid":
                    secondary_json = await self.hybrid_search(query, max_results // 2, metadata_filters)
                else:
                    secondary_json = await self.semantic_search(
                        query,
                        max_results // 2,
                        content_type=strategy if strategy in ["title", "summary", "content"] else None,
                        metadata_filters=metadata_filters
                    )
                
                secondary_data = json.loads(secondary_json)
                all_results.extend(secondary_data.get("results", []))
                strategies_used.append(strategy)
                
            except Exception as e:
                logger.warning(f"Secondary strategy {strategy} failed: {e}")
        
        # Deduplicate results
        seen_ids = set()
        unique_results = []
        for result in all_results:
            chunk_id = result.get("chunk_id")
            if chunk_id not in seen_ids:
                unique_results.append(result)
                seen_ids.add(chunk_id)
        
        # Apply confidence threshold if specified
        confidence_threshold = plan.get("confidence_threshold", self.confidence_threshold)
        # Note: We don't have similarity scores in the simplified results, so we'll skip this filter
        
        return json.dumps({
            "query": query,
            "plan": plan,
            "results": unique_results[:max_results],  # Limit final results
            "total_found": len(unique_results),
            "strategies_used": strategies_used
        })
    
    async def _process(self, *, message: AgentInput) -> AgentOutput:
        """
        Process user query with enhanced RAG capabilities.

        Args:
            message: AgentInput containing user query and parameters

        Returns:
            AgentOutput with search results and synthesized response
        """
        try:
            with self._error_capture.capture_context("process_query", agent_type="enhanced_rag", message_type=type(message).__name__):
                # Ensure ChromaDB is ready
                await self.ensure_chromadb_ready()

                # Extract query from input
                query = self._extract_query(message)
                if not query:
                    return AgentOutput(
                        agent_id=self.agent_id, outputs="No query provided. Please provide a search query.", metadata={"error": "missing_query"}
                    )

                # Determine search strategy
                if self.enable_query_planning:
                    # Create intelligent search plan
                    plan_json = await self.create_search_plan(query)
                    plan = json.loads(plan_json)
                    
                    # Execute primary strategy
                    primary_strategy = plan.get("primary_strategy", "semantic")
                    if primary_strategy == "hybrid":
                        results_json = await self.hybrid_search(query, self.max_results_per_strategy)
                    else:
                        results_json = await self.semantic_search(
                            query, 
                            self.max_results_per_strategy,
                            content_type=primary_strategy if primary_strategy in ["title", "summary", "content"] else None
                        )
                else:
                    # Fallback to basic semantic search
                    results_json = await self.semantic_search(query, self.max_results_per_strategy)
                
                results_data = json.loads(results_json)
                
                # Generate response
                response = self._generate_simple_response(query, results_data)
                
                # Prepare output metadata
                output_metadata = {
                    "query": query,
                    "total_results": results_data.get("total_found", 0),
                    "search_type": results_data.get("search_type", "semantic")
                }
                
                if self.enable_query_planning and self.include_search_explanation:
                    output_metadata["search_plan"] = plan

                return AgentOutput(agent_id=self.agent_id, outputs=response, metadata=output_metadata)

        except Exception as e:
            # Capture the error with full context
            error_context = self._error_capture.capture_error(e, {
                "component": "enhanced_rag_agent",
                "operation": "process_query",
                "agent_id": getattr(self, 'agent_id', 'unknown'),
                "message_inputs": getattr(message, 'inputs', {}) if hasattr(message, 'inputs') else {}
            })

            logger.error(f"Enhanced RAG agent error: {e}")

            return AgentOutput(agent_id=self.agent_id, outputs=f"Search failed: {str(e)}", metadata={"error": str(e), "error_context": error_context.model_dump()})

    def _extract_query(self, message: AgentInput) -> str:
        """Extract search query from AgentInput."""
        # Try different input formats from message.inputs
        if message.inputs.get("query"):
            return str(message.inputs["query"])
        elif message.inputs.get("question"):
            return str(message.inputs["question"])
        elif message.inputs.get("search"):
            return str(message.inputs["search"])
        elif message.inputs.get("text"):
            return str(message.inputs["text"])
        elif message.inputs.get("content"):
            return str(message.inputs["content"])
        else:
            # Use first non-empty input value
            for key, value in message.inputs.items():
                if value and isinstance(value, str):
                    return value
        return ""

    def _generate_simple_response(self, query: str, results_data: dict) -> str:
        """
        Generate contextual response based on search results.

        Args:
            query: Original user query
            results_data: Dictionary with search results

        Returns:
            Formatted response string
        """
        results = results_data.get("results", [])
        if not results:
            return f"No relevant information found for: {query}"

        # Create basic response
        response_parts = [f"Found {len(results)} relevant results for: {query}"]
        response_parts.append("\n**Relevant Information:**")

        for i, result in enumerate(results[:5]):  # Top 5 results
            doc_id = result.get("document_id", "Unknown")
            content_preview = result.get("content", "")
            metadata = result.get("metadata", {})
            title = metadata.get("title", f"Document {doc_id}")
            
            result_text = f"\n{i+1}. **{title}**"
            if doc_id != "Unknown":
                result_text += f" (ID: {doc_id})"
            result_text += f"\n   {content_preview}"
            
            response_parts.append(result_text)

        return "\n".join(response_parts)



