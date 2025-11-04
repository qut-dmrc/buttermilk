"""Simplified RAG agent that uses composition over inheritance.

This module provides a clean RAG agent that:
- Uses external search tools instead of embedded logic
- Guarantees structured outputs with citations
- Relies on templates for orchestration
"""

from typing import Any

from pydantic import BaseModel, Field

from buttermilk.agents.llm import LLMAgent


class Reference(BaseModel):
    """Represents a single cited reference within a research result."""

    summary: str = Field(..., description="Summary of the key information from this reference.")
    citation: str = Field(..., description="Source identifier for the reference.")


class ResearchResult(BaseModel):
    """Structured output of a RAG process with citations."""

    literature: list[Reference] = Field(
        ...,
        description="List of literature references used in generating the response.",
    )
    response: str = Field(
        ...,
        description="The synthesized textual response (markdown permitted).",
    )
    summary: str = Field(
        ...,
        description="A brief summary (plain text)",
    )

    def as_markdown(self, agent_id: str = None, call_id: str = None) -> str:
        """Returns a Markdown formatted string for insertion into templates.

        Format follows the standard: agent identifier on first line, followed by
        content-specific fields without empty lines between components.

        Args:
            agent_id: The agent identifier (e.g., "RAG-gpt4")
            call_id: The call identifier for this execution

        Returns:
            str: Formatted markdown string suitable for template insertion
        """
        header = ""
        if agent_id and call_id:
            # Use only the last 8 characters of call_id for brevity
            short_call_id = call_id[-8:] if len(call_id) > 8 else call_id
            header = f"**{agent_id} #{short_call_id}**\n"

        # Format literature references
        lit_str = ""
        if self.literature:
            lit_parts = []
            for ref in self.literature[:3]:  # Show first 3 references
                lit_parts.append(f"- {ref.citation}: {ref.summary[:80]}...")
            lit_str = "\n".join(lit_parts)
            if len(self.literature) > 3:
                lit_str += f"\n- ... and {len(self.literature) - 3} more references"

        return f"{header}" f"{self.summary}\n" f"Response: {self.response[:200]}...\n" f"References:\n" f"{lit_str if lit_str else '- No references'}"

    def __str__(self) -> str:
        """Returns a Markdown formatted string representation.

        When agent context is available (via _agent_id and _call_id attributes),
        includes the full header. Otherwise returns the summary.
        """
        # Check if agent context is available (set by ExecutionTrace)
        agent_id = getattr(self, "_agent_id", None)
        call_id = getattr(self, "_call_id", None)

        if agent_id and call_id:
            return self.as_markdown(agent_id, call_id)

        # Fallback to summary
        return self.summary


class RagAgent(LLMAgent):
    """Base RAG agent ensuring structured outputs with citations.

    This simplified agent:
    - Extends LLMAgent to inherit all tool usage capabilities
    - Forces ResearchResult as the output format
    - Uses external search tools configured in YAML
    - Relies on Jinja2 templates for search orchestration

    The template is responsible for:
    1. Calling search tools
    2. Extracting citations from results
    3. Iterating up to max_tool_iterations
    4. Synthesizing the response
    5. Returning in ResearchResult format
    """

    def __init__(self, *, output_model: type[BaseModel] = None, **kwargs: Any) -> None:
        """Initialize RagAgent with template configuration."""
        if output_model is None:
            output_model = ResearchResult
        super().__init__(output_model=output_model, **kwargs)
