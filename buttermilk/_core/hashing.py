"""Unified hash computation module for all Buttermilk hash operations.

This module provides centralized hash computation for:
- Record content hashes
- Ground truth data hashes  
- Template content hashes
- Flow configuration hashes

All hash functions use SHA256 and consistent formatting for maintainability
and to eliminate duplication across the codebase.
"""

import hashlib
import json
from pathlib import Path
from typing import Any


def compute_sha256_hash(content: str) -> str:
    """Core SHA256 computation used by all hash functions.
    
    Args:
        content: String content to hash
        
    Returns:
        SHA256 hexdigest of the content
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_record_hash(record_markdown: str) -> str:
    """Compute hash of record's markdown representation.
    
    Args:
        record_markdown: The as_markdown() output from a Record
        
    Returns:
        SHA256 hash of the markdown content
    """
    return compute_sha256_hash(record_markdown)


def compute_ground_truth_hash(ground_truth: dict[str, Any] | list[Any] | None) -> str | None:
    """Compute hash of ground truth data.
    
    Args:
        ground_truth: Ground truth data (dict, list, or None)
        
    Returns:
        SHA256 hash of JSON-serialized ground truth, or None if input is None
    """
    if ground_truth is None:
        return None
        
    # Convert to consistent JSON string for hashing
    # Sort keys to ensure consistent hash for same data
    gt_json = json.dumps(ground_truth, sort_keys=True, separators=(",", ":"))
    return compute_sha256_hash(gt_json)


def compute_template_hash(template_content: str) -> str:
    """Compute hash of template content.
    
    Args:
        template_content: The raw template file content
        
    Returns:
        SHA256 hash of the template content (no prefix)
    """
    return compute_sha256_hash(template_content)


def compute_flow_hash(flow_config: dict[str, Any]) -> str:
    """Compute hash of flow configuration.
    
    Args:
        flow_config: Flow configuration dictionary
        
    Returns:
        SHA256 hash of normalized flow configuration
    """
    # Extract and normalize flow config for consistent hashing
    normalized_config = normalize_flow_config(flow_config)
    config_json = json.dumps(normalized_config, sort_keys=True, separators=(",", ":"))
    return compute_sha256_hash(config_json)


def normalize_flow_config(flow_config: dict[str, Any]) -> dict[str, Any]:
    """Extract relevant parts of flow config for hashing.
    
    Includes components that affect flow behavior for A/B testing:
    - Flow name, description, orchestrator
    - Parameters and criteria
    - Agent configurations (names, roles)
    - Storage configurations that affect behavior
    
    Excludes runtime and session-specific data:
    - Session IDs, timestamps
    - Local environment configuration
    - Runtime parameters
    
    Args:
        flow_config: Raw flow configuration
        
    Returns:
        Normalized configuration dictionary for consistent hashing
    """
    normalized = {}
    
    # Include core flow identification
    for field in ["name", "description", "orchestrator"]:
        if field in flow_config:
            normalized[field] = flow_config[field]
    
    # Include parameters that affect flow behavior
    if "parameters" in flow_config:
        normalized["parameters"] = flow_config["parameters"]
    
    # Include agent configurations (names and core config, not runtime params)
    if "agents" in flow_config:
        agents = flow_config["agents"]
        if isinstance(agents, dict):
            # Normalize agent configs by extracting core configuration
            normalized_agents = {}
            for agent_name, agent_config in agents.items():
                if isinstance(agent_config, dict):
                    # Include core agent configuration, exclude runtime parameters
                    normalized_agent = {}
                    for field in ["name", "role", "instructions", "template"]:
                        if field in agent_config:
                            normalized_agent[field] = agent_config[field]
                    if normalized_agent:
                        normalized_agents[agent_name] = normalized_agent
                else:
                    # Agent config is not a dict, include as-is
                    normalized_agents[agent_name] = agent_config
            if normalized_agents:
                normalized["agents"] = normalized_agents
    
    # Include observer configurations if present
    if "observers" in flow_config:
        normalized["observers"] = flow_config["observers"]
    
    # Include storage configurations that affect flow behavior
    # Exclude save configurations that are just for output routing
    if "storage" in flow_config:
        storage = flow_config["storage"]
        if isinstance(storage, dict):
            # Only include storage configs that affect flow behavior
            normalized_storage = {}
            for key, value in storage.items():
                # Include storage configs that affect data access/behavior
                # Exclude pure output/save configurations
                if not key.startswith("save") and value is not None:
                    normalized_storage[key] = value
            if normalized_storage:
                normalized["storage"] = normalized_storage
    
    return normalized


def compute_template_hash_from_file(template_path: str | Path) -> str:
    """Compute template hash by reading file content.

    Args:
        template_path: Path to template file

    Returns:
        SHA256 hash of template file content

    Raises:
        FileNotFoundError: If template file doesn't exist
        IOError: If template file cannot be read
    """
    template_path = Path(template_path)
    template_content = template_path.read_text(encoding="utf-8")
    return compute_template_hash(template_content)


def compute_processor_config_hash(processor_config: dict[str, Any]) -> str:
    """Compute hash of processor configuration for cache invalidation.

    This function creates a stable, deterministic hash of processor configuration
    parameters to enable cache invalidation when processor configs change.

    Includes configuration that affects processor behavior:
    - Model names, templates, mappings
    - Processing parameters (batch sizes, thresholds, etc.)
    - Any other parameters that affect output

    Excludes runtime and internal state:
    - Private attributes (starting with _)
    - Semaphores, locks, clients
    - Cached compiled objects (_compiled_expressions, etc.)

    Args:
        processor_config: Processor configuration dictionary

    Returns:
        First 8 characters of SHA256 hash for readability in cache paths

    Example:
        >>> config = {"model": "gpt-4", "template": "summarize", "temperature": 0.7}
        >>> compute_processor_config_hash(config)
        'a3f8b2c1'
    """
    # Extract only serializable configuration parameters
    normalized = {}

    for key, value in processor_config.items():
        # Skip private attributes and internal state
        if key.startswith("_"):
            continue

        # Skip non-serializable objects
        if callable(value):
            continue

        # Skip None values for consistency
        if value is None:
            continue

        # Include serializable configuration
        try:
            # Test if value is JSON-serializable
            json.dumps(value)
            normalized[key] = value
        except (TypeError, ValueError):
            # Skip non-serializable values (semaphores, locks, clients, etc.)
            continue

    # Convert to consistent JSON string for hashing
    # Sort keys to ensure consistent hash for same config
    config_json = json.dumps(normalized, sort_keys=True, separators=(",", ":"))

    # Return first 8 characters for readability in cache paths
    full_hash = compute_sha256_hash(config_json)
    return full_hash[:8]


def hash_dict(data: dict[str, Any]) -> str:
    """Compute deterministic hash of dictionary.

    Sorts keys and values to ensure consistent hashing.

    Args:
        data: Dictionary to hash

    Returns:
        SHA256 hex hash of dictionary

    Example:
        >>> hash_dict({"model": "gpt-4", "temp": 0.7})
        'a3f8b2c1...'
    """
    # Sort keys for deterministic hashing
    serialized = json.dumps(data, sort_keys=True)
    return compute_sha256_hash(serialized)


def hash_content(content: str | bytes) -> str:
    """Compute hash of content for tracing.

    Args:
        content: String or bytes to hash

    Returns:
        Hex string of content hash

    Example:
        >>> hash_content("test content")
        'abc123def456...'
    """
    if isinstance(content, str):
        content = content.encode("utf-8")

    return hashlib.sha256(content).hexdigest()


class HashCollector:
    """Collects all hashes for a trace.

    Provides systematic hash capture for OpenTelemetry spans,
    ensuring all critical hashes (template, config, record, etc.)
    are logged consistently.

    Example:
        >>> collector = HashCollector()
        >>> collector.add_template_hash("judge.jinja2", template_content)
        >>> collector.add_config_hash(agent.parameters)
        >>> span_attrs = collector.to_span_attributes()
        >>> # Use span_attrs in OTEL span creation
    """

    def __init__(self) -> None:
        """Initialize empty hash collector."""
        self.hashes: dict[str, str] = {}

    def add_template_hash(self, template_name: str, template_content: str) -> None:
        """Add template content hash.

        Args:
            template_name: Name of the template (for reference)
            template_content: Template content to hash
        """
        self.hashes["hash.template"] = compute_template_hash(template_content)
        self.hashes["hash.template.name"] = template_name

    def add_config_hash(self, config: dict[str, Any]) -> None:
        """Add configuration hash.

        Args:
            config: Configuration dictionary to hash
        """
        self.hashes["hash.config"] = hash_dict(config)

    def add_record_hash(self, record: Any) -> None:
        """Add record hash.

        Args:
            record: Record object with record_hash and record_id attributes
        """
        if hasattr(record, "record_hash") and record.record_hash:
            self.hashes["hash.record"] = record.record_hash

        if hasattr(record, "record_id") and record.record_id:
            self.hashes["hash.record.id"] = record.record_id

    def add_output_hash(self, output: str | dict[str, Any]) -> None:
        """Add output hash.

        Args:
            output: Agent output to hash
        """
        if isinstance(output, dict):
            self.hashes["hash.output"] = hash_dict(output)
        else:
            self.hashes["hash.output"] = hash_content(str(output))

    def add_custom_hash(self, key: str, value: str) -> None:
        """Add custom hash with specified key.

        Args:
            key: Hash key (should start with "hash.")
            value: Hash value
        """
        if not key.startswith("hash."):
            key = f"hash.{key}"
        self.hashes[key] = value

    def to_span_attributes(self) -> dict[str, str]:
        """Convert to OTEL span attributes.

        Returns:
            Dictionary of hash attributes, excluding None values
        """
        return {k: v for k, v in self.hashes.items() if v is not None}
