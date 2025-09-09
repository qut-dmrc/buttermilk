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