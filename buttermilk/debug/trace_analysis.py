"""Core trace analysis functions."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from buttermilk._core.contract import ExecutionTrace

from .trace_models import ErrorContext, TimelineEvent, TraceDiff, TraceFile, TraceSummary


def load_traces(path: Path) -> list[ExecutionTrace]:
    """Load traces from JSON or JSONL file.

    Handles both JSON arrays and JSONL formats. Parses nested JSON strings
    in agent_info, inputs, and messages fields.

    Args:
        path: Path to the trace file

    Returns:
        List of ExecutionTrace objects

    Raises:
        FileNotFoundError: If path does not exist
        ValueError: If file format is invalid or parsing fails
    """
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    content = path.read_text()

    # Try parsing as JSON array first
    try:
        raw_data = json.loads(content)
        if not isinstance(raw_data, list):
            raw_data = [raw_data]
    except json.JSONDecodeError:
        # Try JSONL format
        raw_data = []
        for line_num, line in enumerate(content.strip().split("\n"), 1):
            if not line.strip():
                continue
            try:
                raw_data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e

    # Parse each trace, handling nested JSON strings
    traces = []
    for i, raw_trace in enumerate(raw_data):
        try:
            # Handle nested JSON strings in certain fields
            trace_dict = raw_trace.copy()

            # Parse agent_info if it's a string
            if isinstance(trace_dict.get("agent_info"), str):
                trace_dict["agent_info"] = json.loads(trace_dict["agent_info"])

            # Parse inputs if it's a string
            if isinstance(trace_dict.get("inputs"), str):
                trace_dict["inputs"] = json.loads(trace_dict["inputs"])

            # Parse messages if they're strings
            if "messages" in trace_dict and isinstance(trace_dict["messages"], list):
                parsed_messages = []
                for msg in trace_dict["messages"]:
                    if isinstance(msg, str):
                        parsed_messages.append(json.loads(msg))
                    else:
                        parsed_messages.append(msg)
                trace_dict["messages"] = parsed_messages

            # Parse timestamp if it's a string
            if isinstance(trace_dict.get("timestamp"), str):
                # Try multiple datetime formats
                timestamp_str = trace_dict["timestamp"]
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"]:
                    try:
                        trace_dict["timestamp"] = datetime.strptime(timestamp_str, fmt)
                        break
                    except ValueError:
                        continue

            traces.append(ExecutionTrace(**trace_dict))
        except Exception as e:
            raise ValueError(f"Failed to parse trace at index {i}: {e}") from e

    return traces


def load_trace_file(path: Path) -> TraceFile:
    """Load traces with file metadata.

    Args:
        path: Path to the trace file

    Returns:
        TraceFile object containing traces and metadata
    """
    traces = load_traces(path)
    return TraceFile(path=path, traces=traces, loaded_at=datetime.now())


def get_trace(traces: list[ExecutionTrace], call_id: str) -> ExecutionTrace | None:
    """Find trace by call_id.

    Args:
        traces: List of traces to search
        call_id: Unique call identifier

    Returns:
        ExecutionTrace if found, None otherwise
    """
    for trace in traces:
        if trace.call_id == call_id:
            return trace
    return None


def get_traces_by_agent(traces: list[ExecutionTrace], agent_name: str) -> list[ExecutionTrace]:
    """Get traces by agent role or component_name (case-insensitive).

    Args:
        traces: List of traces to search
        agent_name: Agent role or component name to match

    Returns:
        List of matching traces
    """
    agent_name_lower = agent_name.lower()
    matching_traces = []

    for trace in traces:
        # Check agent_info.role
        role = trace.agent_info.get("role", "").lower()
        if role == agent_name_lower:
            matching_traces.append(trace)
            continue

        # Check agent_info.component_name
        component_name = trace.agent_info.get("component_name", "").lower()
        if component_name == agent_name_lower:
            matching_traces.append(trace)
            continue

        # Check agent_info.agent_name
        agent_name_field = trace.agent_info.get("agent_name", "").lower()
        if agent_name_field == agent_name_lower:
            matching_traces.append(trace)

    return matching_traces


def filter_traces(
    traces: list[ExecutionTrace],
    *,
    has_error: bool | None = None,
    agent_name: str | None = None,
    after: datetime | None = None,
    before: datetime | None = None,
) -> list[ExecutionTrace]:
    """Filter traces by criteria.

    Args:
        traces: List of traces to filter
        has_error: If True, return only traces with errors; if False, only without errors
        agent_name: Filter by agent role or component name
        after: Return only traces after this timestamp
        before: Return only traces before this timestamp

    Returns:
        List of traces matching all specified criteria
    """
    filtered = traces

    if has_error is not None:
        filtered = [t for t in filtered if t.is_error == has_error]

    if agent_name is not None:
        filtered = get_traces_by_agent(filtered, agent_name)

    if after is not None:
        filtered = [t for t in filtered if t.timestamp >= after]

    if before is not None:
        filtered = [t for t in filtered if t.timestamp <= before]

    return filtered


def summarize(traces: list[ExecutionTrace]) -> TraceSummary:
    """Generate summary statistics.

    Args:
        traces: List of traces to summarize

    Returns:
        TraceSummary with aggregated statistics
    """
    if not traces:
        return TraceSummary(
            total_traces=0,
            error_count=0,
            agents=[],
            execution_path=[],
            time_range=None,
            session_id=None,
            by_agent={},
            by_error_status={"success": 0, "error": 0},
        )

    # Count errors
    error_count = sum(1 for t in traces if t.is_error)

    # Collect unique agents
    agents = set()
    for trace in traces:
        role = trace.agent_info.get("role")
        if role:
            agents.add(role)
        component_name = trace.agent_info.get("component_name")
        if component_name:
            agents.add(component_name)

    # Build execution path (chronological order of call_ids)
    sorted_traces = sorted(traces, key=lambda t: t.timestamp)
    execution_path = [t.call_id for t in sorted_traces]

    # Time range
    timestamps = [t.timestamp for t in traces]
    time_range = (min(timestamps), max(timestamps))

    # Session ID (use first non-None)
    session_id = None
    for trace in traces:
        if trace.session_id:
            session_id = trace.session_id
            break

    # Count by agent
    by_agent: dict[str, int] = {}
    for trace in traces:
        role = trace.agent_info.get("role", "unknown")
        by_agent[role] = by_agent.get(role, 0) + 1

    # Count by error status
    by_error_status = {"success": len(traces) - error_count, "error": error_count}

    return TraceSummary(
        total_traces=len(traces),
        error_count=error_count,
        agents=sorted(agents),
        execution_path=execution_path,
        time_range=time_range,
        session_id=session_id,
        by_agent=by_agent,
        by_error_status=by_error_status,
    )


def get_errors(traces: list[ExecutionTrace]) -> list[ErrorContext]:
    """Extract all errors with context.

    Args:
        traces: List of traces to analyze

    Returns:
        List of ErrorContext objects for all errors found
    """
    errors = []

    for trace in traces:
        if not trace.is_error:
            continue

        # Extract error information
        error_dict = trace.error or {}
        error_message = error_dict.get("event", "Unknown error")
        error_details = error_dict.get("details", {})

        # Extract agent info
        agent_name = trace.agent_info.get("component_name", "unknown")
        agent_role = trace.agent_info.get("role", "unknown")

        # Summarize inputs
        inputs_summary = str(trace.inputs)[:200] if trace.inputs else "No inputs"

        # Find preceding traces (same session, earlier timestamp)
        preceding = [t.call_id for t in traces if t.session_id == trace.session_id and t.timestamp < trace.timestamp]

        errors.append(
            ErrorContext(
                call_id=trace.call_id,
                agent_name=agent_name,
                agent_role=agent_role,
                error_message=error_message,
                error_details=error_details,
                timestamp=trace.timestamp,
                inputs_summary=inputs_summary,
                parent_call_id=trace.parent_call_id,
                preceding_trace_ids=preceding,
            )
        )

    return errors


def get_timeline(traces: list[ExecutionTrace]) -> list[TimelineEvent]:
    """Generate chronological timeline.

    Args:
        traces: List of traces to analyze

    Returns:
        List of TimelineEvent objects in chronological order
    """
    events = []

    for trace in traces:
        agent_name = trace.agent_info.get("component_name", "unknown")
        agent_role = trace.agent_info.get("role", "unknown")

        # Determine event type
        event_type: Any = "success"
        summary = f"{agent_role} completed"

        if trace.is_error:
            event_type = "error"
            error_dict = trace.error or {}
            error_msg = error_dict.get("event", "Unknown error")
            summary = f"{agent_role} failed: {error_msg}"

        # Calculate duration if available (would need start/end timestamps)
        duration_ms = None

        events.append(
            TimelineEvent(
                call_id=trace.call_id,
                timestamp=trace.timestamp,
                agent_name=agent_name,
                agent_role=agent_role,
                event_type=event_type,
                duration_ms=duration_ms,
                summary=summary,
            )
        )

    # Sort chronologically
    events.sort(key=lambda e: e.timestamp)

    return events


def get_llm_conversation(trace: ExecutionTrace) -> list[dict]:
    """Extract LLM messages.

    Args:
        trace: ExecutionTrace to extract messages from

    Returns:
        List of message dictionaries with role and content
    """
    messages = []

    for msg in trace.messages:
        # Handle both dict and object formats
        if isinstance(msg, dict):
            messages.append({"type": msg.get("type", "unknown"), "content": msg.get("content", ""), "source": msg.get("source", "")})
        else:
            # Handle Pydantic models
            messages.append({"type": getattr(msg, "type", "unknown"), "content": getattr(msg, "content", ""), "source": getattr(msg, "source", "")})

    return messages


def get_inputs_outputs(trace: ExecutionTrace) -> dict:
    """Get readable I/O.

    Args:
        trace: ExecutionTrace to extract I/O from

    Returns:
        Dictionary with 'inputs' and 'outputs' keys
    """
    return {"inputs": trace.inputs, "outputs": trace.outputs}


def diff_traces(trace1: ExecutionTrace, trace2: ExecutionTrace) -> TraceDiff:
    """Compare two traces.

    Args:
        trace1: First trace to compare
        trace2: Second trace to compare

    Returns:
        TraceDiff object describing differences
    """
    # Get dict representations
    dict1 = trace1.model_dump()
    dict2 = trace2.model_dump()

    # Find fields only in one trace
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    only_in_trace1 = sorted(keys1 - keys2)
    only_in_trace2 = sorted(keys2 - keys1)

    # Find differences in common fields
    differences = {}
    for key in keys1 & keys2:
        val1 = dict1[key]
        val2 = dict2[key]

        if val1 != val2:
            differences[key] = {"trace1": val1, "trace2": val2}

    return TraceDiff(
        trace1_id=trace1.call_id, trace2_id=trace2.call_id, differences=differences, only_in_trace1=only_in_trace1, only_in_trace2=only_in_trace2
    )
