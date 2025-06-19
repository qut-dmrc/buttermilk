"""
Enhanced RAG Agent with intelligent search planning and LLM-driven synthesis.

This agent provides advanced RAG capabilities that go beyond simple vector similarity:
1. LLM-driven query analysis and search planning
2. Multi-field search across titles, summaries, content, and metadata
3. Intelligent result synthesis and ranking
4. Configurable search strategies per use case
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from buttermilk.debug.error_capture import ErrorCapture, capture_enhanced_rag_errors, safe_isinstance_check

from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import AgentInput, AgentOutput
from buttermilk.agents.llm import LLMAgent
from buttermilk._core.log import logger
from buttermilk._core.types import Record
from buttermilk._core.mcp_decorators import tool
from buttermilk.agents.rag.enhanced_search import EnhancedVectorSearch
from buttermilk.agents.rag.search_planning import (
    SearchPlan,
    SearchResults,
    SearchStrategy,
    QueryAnalysis,
)


class EnhancedRagAgent(LLMAgent):
    """
    Enhanced RAG Agent with intelligent search capabilities.

    This agent uses LLM reasoning to:
    - Analyze queries and understand user intent
    - Plan optimal search strategies across multiple content fields
    - Execute targeted searches with metadata filtering
    - Synthesize and rank results intelligently
    - Generate contextual responses based on findings
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
        self._enhanced_search: Optional[EnhancedVectorSearch] = None
        self._error_capture = ErrorCapture(capture_locals=True)

    async def _initialize_search_tools(self) -> None:
        """Initialize enhanced search tools with vectorstore and LLM."""
        with self._error_capture.capture_context("initialize_search_tools", agent_type="enhanced_rag"):
            if self._enhanced_search is not None:
                return

            # Get vectorstore from data configuration
            if not self.data:
                raise ValueError("Enhanced RAG agent requires data configuration with vectorstore")

            # Load vectorstore from first data source
            from buttermilk._core.dmrc import get_bm

            bm = get_bm()

            vectorstore_key = list(self.data.keys())[0]
            storage_config = self.data[vectorstore_key]

            vectorstore = await bm.get_storage_async(storage_config)

            # Initialize enhanced search with vectorstore and LLM client from parent
            from buttermilk import buttermilk as bm_instance

            model_client = bm_instance.llms.get_autogen_chat_client(self._model)
            self._enhanced_search = EnhancedVectorSearch(vectorstore=vectorstore, llm_client=model_client)

            logger.info(f"Enhanced RAG agent initialized with vectorstore: {vectorstore.collection_name}")

    @tool(name="search", description="Search the knowledge base with an intelligent RAG system")
    async def search(self, query: str, max_results: int = 10) -> str:
        """Search the knowledge base using enhanced RAG capabilities.
        
        Args:
            query: The search query or question
            max_results: Maximum number of results to return
            
        Returns:
            A synthesized response based on the search results
        """
        # Create an AgentInput for backward compatibility
        message = AgentInput(
            inputs={"query": query, "max_results": max_results}
        )
        
        # Use the existing _process logic
        result = await self._process(message=message)
        
        # Return just the output string for tool compatibility
        return result.outputs if isinstance(result.outputs, str) else str(result.outputs)
    
    async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
        """
        Process user query with enhanced RAG capabilities.

        Args:
            message: AgentInput containing user query and parameters

        Returns:
            AgentOutput with search results and synthesized response
        """
        try:
            with self._error_capture.capture_context("process_query", agent_type="enhanced_rag", message_type=type(message).__name__):
                await self._initialize_search_tools()

                # Extract query from input
                query = self._extract_query(message)
                if not query:
                    return AgentOutput(
                        agent_id=self.agent_id, outputs="No query provided. Please provide a search query.", metadata={"error": "missing_query"}
                    )

                # Step 1: Create search plan (if enabled)
                if self.enable_query_planning:
                    search_plan = await self._enhanced_search.create_search_plan(query)
                    logger.info(f"Created search plan: {search_plan.primary_strategy} + {len(search_plan.secondary_strategies)} secondary")
                else:
                    # Fallback to basic semantic search
                    search_plan = SearchPlan(
                        query=query,
                        intent="Basic search query",
                        primary_strategy=SearchStrategy.SEMANTIC,
                        max_results_per_strategy=self.max_results_per_strategy,
                        confidence_threshold=self.confidence_threshold,
                    )

                # Step 2: Execute search plan
                search_results = await self._enhanced_search.execute_search_plan(search_plan)

                # Step 3: Generate response
                response = await self._generate_response(query, search_results)

                # Step 4: Prepare output
                output_metadata = {
                    "query": query,
                    "total_results": search_results.total_found,
                    "strategies_used": [s.value for s in search_results.strategies_used],
                    "confidence_score": search_results.confidence_score,
                    "key_themes": search_results.key_themes,
                }

                if self.include_search_explanation:
                    output_metadata["search_explanation"] = self._create_search_explanation(search_plan, search_results)

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

    async def _generate_response(self, query: str, search_results: SearchResults) -> str:
        """
        Generate contextual response based on search results.

        Args:
            query: Original user query
            search_results: SearchResults from enhanced search

        Returns:
            Formatted response string
        """
        if not search_results.results:
            return f"No relevant information found for: {query}"

        # Create basic response
        response_parts = []

        # Add synthesis summary if available
        if search_results.synthesis_summary:
            response_parts.append(f"**Summary:** {search_results.synthesis_summary}")

        # Add key themes if available
        if search_results.key_themes:
            response_parts.append(f"**Key Themes:** {', '.join(search_results.key_themes)}")

        # Add top results
        response_parts.append("**Relevant Information:**")

        for i, result in enumerate(search_results.results[:5]):  # Top 5 results
            title = result.metadata.get("title", "Document")
            case_number = result.metadata.get("case_number", "")
            content_preview = result.content[:200] + "..." if len(result.content) > 200 else result.content

            result_text = f"{i+1}. **{title}**"
            if case_number:
                result_text += f" ({case_number})"
            result_text += f"\n   {content_preview}"

            if result.relevance_explanation:
                result_text += f"\n   *Relevance: {result.relevance_explanation}*"

            response_parts.append(result_text)

        # Add confidence note
        if search_results.confidence_score:
            confidence_level = "high" if search_results.confidence_score > 0.8 else "medium" if search_results.confidence_score > 0.6 else "low"
            response_parts.append(f"*Confidence: {confidence_level} ({search_results.confidence_score:.2f})*")

        return "\n\n".join(response_parts)

    def _create_search_explanation(self, plan: SearchPlan, results: SearchResults) -> str:
        """Create explanation of search strategy used."""
        explanation_parts = [
            f"Search Intent: {plan.intent}",
            f"Primary Strategy: {plan.primary_strategy.value}",
        ]

        if plan.secondary_strategies:
            explanation_parts.append(f"Secondary Strategies: {', '.join(s.value for s in plan.secondary_strategies)}")

        if plan.metadata_filters:
            explanation_parts.append(f"Applied Filters: {plan.metadata_filters}")

        explanation_parts.append(f"Results Found: {results.total_found}")

        return " | ".join(explanation_parts)


# Enhanced RAG Tool Configuration
class EnhancedRagTool:
    """Tool configuration for enhanced RAG capabilities."""

    @staticmethod
    def create_agent_config(
        vectorstore_config, enable_planning: bool = True, enable_synthesis: bool = True, strategies: Optional[List[str]] = None
    ) -> AgentConfig:
        """
        Create AgentConfig for Enhanced RAG Agent.

        Args:
            vectorstore_config: Storage configuration for vectorstore
            enable_planning: Enable LLM-driven query planning
            enable_synthesis: Enable LLM-driven result synthesis
            strategies: List of search strategies to enable

        Returns:
            AgentConfig configured for enhanced RAG
        """
        if strategies is None:
            strategies = ["semantic", "title", "summary", "hybrid"]

        return AgentConfig(
            role="ENHANCED_RESEARCHER",
            agent_obj="EnhancedRagAgent",
            description="Advanced research assistant with intelligent search capabilities",
            data={"vectorstore": vectorstore_config},
            parameters={
                "enable_query_planning": enable_planning,
                "enable_result_synthesis": enable_synthesis,
                "search_strategies": strategies,
                "max_search_rounds": 3,
                "confidence_threshold": 0.5,
                "max_results_per_strategy": 5,
                "include_search_explanation": True,
            },
        )
