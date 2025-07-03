"""RAG (Retrieval Augmented Generation) agents."""

from .simple_rag_agent import RagAgent, Reference, ResearchResult
from .rag_zotero import RagZotero, ZoteroReference, ZoteroResearchResult

# Legacy imports - to be removed
from .rag_agent import RagAgent as LegacyRagAgent
from .ragzot import RagZot
from .enhanced_rag_agent import EnhancedRagAgent

__all__ = [
    # New simplified agents
    "RagAgent",
    "Reference", 
    "ResearchResult",
    "RagZotero",
    "ZoteroReference",
    "ZoteroResearchResult",
    # Legacy - to be removed
    "LegacyRagAgent",
    "RagZot",
    "EnhancedRagAgent",
]