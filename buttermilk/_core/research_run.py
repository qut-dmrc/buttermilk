"""Research Run management for grouping related sessions.

This module provides the ResearchRun class, which represents the middle tier
in Buttermilk's three-tier architecture:
- ExecutionContext (process-level infrastructure)
- ResearchRun (logical grouping of related research tasks)  
- Session (individual task execution)

A ResearchRun groups multiple sessions that logically belong together for
research purposes, such as all tasks in a batch job, or a series of related
analyses for a project.
"""

from __future__ import annotations

import datetime
import platform
from enum import Enum
from typing import Any

import psutil
import shortuuid
from pydantic import BaseModel, Field

from buttermilk._core.log import logger


class ResearchRunStatus(str, Enum):
    """Status enumeration for research run lifecycle management."""
    
    INITIALIZING = "initializing"  # Run created, sessions being set up
    ACTIVE = "active"  # Sessions running
    COMPLETED = "completed"  # All sessions completed successfully
    FAILED = "failed"  # One or more sessions failed
    CANCELLED = "cancelled"  # Run was cancelled
    MIXED = "mixed"  # Some sessions completed, some failed


def _make_research_run_id() -> str:
    """Generates a unique research run ID.

    The ID is constructed using the current UTC timestamp, a short UUID,
    the machine's node name, and the current username.

    Returns:
        str: A unique string identifier for this research run.
    """
    node_name = platform.uname().node
    username = psutil.Process().username()
    username = str.split(username, "\\")[-1]  # Strip domain if present

    # Format timestamp for use in filenames (simplified ISO 8601)
    run_time = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%MZ")

    research_run_id = f"run-{run_time}-{shortuuid.uuid()[:4]}-{node_name}-{username}"
    return research_run_id


class ResearchRunMetadata(BaseModel):
    """Metadata for a research run.
    
    Contains information about the research context, methodology,
    and parameters that apply to all sessions in the run.
    """
    
    # Basic identification
    research_run_id: str = Field(
        default_factory=_make_research_run_id,
        description="Unique identifier for this research run."
    )
    name: str = Field(..., description="Human-readable name for the research run.")
    description: str | None = Field(
        default=None,
        description="Detailed description of the research objectives."
    )
    
    # Research context
    project: str | None = Field(
        default=None, 
        description="Project this research run belongs to."
    )
    researcher: str | None = Field(
        default=None,
        description="Primary researcher or team responsible."
    )
    methodology: str | None = Field(
        default=None,
        description="Research methodology or approach used."
    )
    
    # Execution context
    platform: str = Field(
        default="local",
        description="Platform where the research run is executed."
    )
    node_name: str = Field(
        default_factory=lambda: platform.uname().node,
        description="Machine/node where run was initiated."
    )
    
    # Timestamps
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
        description="When the research run was created."
    )
    started_at: datetime.datetime | None = Field(
        default=None,
        description="When the first session started."
    )
    completed_at: datetime.datetime | None = Field(
        default=None,
        description="When the last session completed."
    )
    
    # Configuration
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Research parameters that apply to all sessions."
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing and filtering runs."
    )


class SessionSummary(BaseModel):
    """Summary information about a session within a research run."""
    
    session_id: str = Field(..., description="Unique session identifier.")
    name: str = Field(..., description="Session name.")
    job: str = Field(..., description="Job/task name.")
    status: str = Field(..., description="Current session status.")
    
    started_at: datetime.datetime | None = Field(
        default=None,
        description="When the session started."
    )
    completed_at: datetime.datetime | None = Field(
        default=None,
        description="When the session completed."
    )
    
    error_message: str | None = Field(
        default=None,
        description="Error message if session failed."
    )
    
    # Results summary
    records_processed: int | None = Field(
        default=None,
        description="Number of records processed by this session."
    )
    outputs_generated: int | None = Field(
        default=None,
        description="Number of outputs generated."
    )


class ResearchRun(BaseModel):
    """Represents a logical grouping of related research sessions.
    
    A ResearchRun coordinates multiple sessions that belong together for
    research purposes. It tracks overall progress, manages shared metadata,
    and provides aggregated reporting across all sessions in the run.
    
    Examples:
    - All tasks in a batch job analyzing a dataset
    - Multiple related experiments in a research project
    - A series of iterative refinements on a research question
    """
    
    # Core metadata
    metadata: ResearchRunMetadata = Field(..., description="Research run metadata.")
    
    # Status tracking
    status: ResearchRunStatus = Field(
        default=ResearchRunStatus.INITIALIZING,
        description="Current status of the research run."
    )
    
    # Session tracking
    sessions: dict[str, SessionSummary] = Field(
        default_factory=dict,
        description="Summary of all sessions in this research run."
    )
    
    # Results aggregation
    total_sessions: int = Field(
        default=0,
        description="Total number of sessions planned for this run."
    )
    completed_sessions: int = Field(
        default=0,
        description="Number of sessions completed successfully."
    )
    failed_sessions: int = Field(
        default=0,
        description="Number of sessions that failed."
    )
    
    def add_session(
        self,
        session_id: str,
        name: str,
        job: str,
        **kwargs
    ) -> None:
        """Add a session to this research run.
        
        Args:
            session_id: Unique identifier for the session.
            name: Session name.
            job: Job/task name.
            **kwargs: Additional session summary fields.
        """
        session_summary = SessionSummary(
            session_id=session_id,
            name=name,
            job=job,
            status="initializing",
            **kwargs
        )
        
        self.sessions[session_id] = session_summary
        self.total_sessions = len(self.sessions)
        
        # Update run status if needed
        if self.status == ResearchRunStatus.INITIALIZING and self.total_sessions > 0:
            self.status = ResearchRunStatus.ACTIVE
            self.metadata.started_at = datetime.datetime.now(datetime.UTC)
        
        logger.info(
            "Added session to research run",
            research_run_id=self.metadata.research_run_id,
            session_id=session_id,
            total_sessions=self.total_sessions
        )
    
    def update_session_status(
        self,
        session_id: str,
        status: str,
        **kwargs
    ) -> None:
        """Update the status of a session in this research run.
        
        Args:
            session_id: Session to update.
            status: New session status.
            **kwargs: Additional fields to update.
        """
        if session_id not in self.sessions:
            logger.warning(
                "Attempted to update unknown session",
                research_run_id=self.metadata.research_run_id,
                session_id=session_id
            )
            return
        
        session = self.sessions[session_id]
        old_status = session.status
        session.status = status
        
        # Update additional fields
        for key, value in kwargs.items():
            if hasattr(session, key):
                setattr(session, key, value)
        
        # Update timestamps
        if status in ["active", "running"] and session.started_at is None:
            session.started_at = datetime.datetime.now(datetime.UTC)
        elif status in ["completed", "failed", "error"] and session.completed_at is None:
            session.completed_at = datetime.datetime.now(datetime.UTC)
        
        # Update run-level counters
        self._update_run_status()
        
        logger.info(
            "Updated session status",
            research_run_id=self.metadata.research_run_id,
            session_id=session_id,
            old_status=old_status,
            new_status=status
        )
    
    def _update_run_status(self) -> None:
        """Update the overall run status based on session statuses."""
        if not self.sessions:
            return
        
        # Count session statuses
        status_counts = {}
        for session in self.sessions.values():
            status_counts[session.status] = status_counts.get(session.status, 0) + 1
        
        self.completed_sessions = status_counts.get("completed", 0)
        self.failed_sessions = status_counts.get("failed", 0) + status_counts.get("error", 0)
        
        # Determine run status
        total = len(self.sessions)
        
        if self.completed_sessions == total:
            # All sessions completed successfully
            self.status = ResearchRunStatus.COMPLETED
            if self.metadata.completed_at is None:
                self.metadata.completed_at = datetime.datetime.now(datetime.UTC)
        elif self.failed_sessions > 0 and (self.completed_sessions + self.failed_sessions) == total:
            # All sessions done, but some failed
            if self.completed_sessions > 0:
                self.status = ResearchRunStatus.MIXED
            else:
                self.status = ResearchRunStatus.FAILED
            if self.metadata.completed_at is None:
                self.metadata.completed_at = datetime.datetime.now(datetime.UTC)
        elif status_counts.get("cancelled", 0) > 0:
            self.status = ResearchRunStatus.CANCELLED
            if self.metadata.completed_at is None:
                self.metadata.completed_at = datetime.datetime.now(datetime.UTC)
        else:
            # Still have active sessions
            self.status = ResearchRunStatus.ACTIVE
    
    def get_progress_summary(self) -> dict[str, Any]:
        """Get a summary of research run progress.
        
        Returns:
            Dict containing progress metrics and timing information.
        """
        total = len(self.sessions)
        completed = self.completed_sessions
        failed = self.failed_sessions
        active = total - completed - failed
        
        # Calculate timing
        duration = None
        if self.metadata.started_at:
            end_time = self.metadata.completed_at or datetime.datetime.now(datetime.UTC)
            duration = (end_time - self.metadata.started_at).total_seconds()
        
        return {
            "research_run_id": self.metadata.research_run_id,
            "name": self.metadata.name,
            "status": self.status,
            "progress": {
                "total_sessions": total,
                "completed": completed,
                "failed": failed,
                "active": active,
                "completion_rate": completed / total if total > 0 else 0,
            },
            "timing": {
                "created_at": self.metadata.created_at.isoformat(),
                "started_at": self.metadata.started_at.isoformat() if self.metadata.started_at else None,
                "completed_at": self.metadata.completed_at.isoformat() if self.metadata.completed_at else None,
                "duration_seconds": duration,
            },
            "metadata": {
                "project": self.metadata.project,
                "researcher": self.metadata.researcher,
                "methodology": self.metadata.methodology,
                "tags": self.metadata.tags,
            }
        }


# Factory functions for creating research runs

def create_research_run(
    name: str,
    description: str | None = None,
    project: str | None = None,
    researcher: str | None = None,
    methodology: str | None = None,
    parameters: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    **kwargs
) -> ResearchRun:
    """Create a new research run.
    
    Args:
        name: Human-readable name for the research run.
        description: Detailed description of research objectives.
        project: Project this research run belongs to.
        researcher: Primary researcher or team responsible.
        methodology: Research methodology or approach.
        parameters: Research parameters that apply to all sessions.
        tags: Tags for categorizing and filtering runs.
        **kwargs: Additional metadata fields.
        
    Returns:
        ResearchRun: A new research run instance.
    """
    metadata_data = {
        "name": name,
        "description": description,
        "project": project,
        "researcher": researcher,
        "methodology": methodology,
        "parameters": parameters or {},
        "tags": tags or [],
        **kwargs
    }
    
    metadata = ResearchRunMetadata(**metadata_data)
    
    research_run = ResearchRun(metadata=metadata)
    
    logger.info(
        "Created research run",
        research_run_id=research_run.metadata.research_run_id,
        name=name,
        project=project
    )
    
    return research_run


def create_batch_research_run(
    batch_name: str,
    dataset_info: dict[str, Any],
    **kwargs
) -> ResearchRun:
    """Create a research run specifically for batch processing.
    
    Args:
        batch_name: Name for the batch processing run.
        dataset_info: Information about the dataset being processed.
        **kwargs: Additional research run parameters.
        
    Returns:
        ResearchRun: A research run configured for batch processing.
    """
    description = f"Batch processing run for {batch_name}"
    if "description" in dataset_info:
        description += f": {dataset_info['description']}"
    
    parameters = {
        "dataset_info": dataset_info,
        "processing_type": "batch",
        **kwargs.get("parameters", {})
    }
    
    tags = ["batch", "automated"] + kwargs.get("tags", [])
    
    return create_research_run(
        name=batch_name,
        description=description,
        methodology="batch_processing",
        parameters=parameters,
        tags=tags,
        **kwargs
    )