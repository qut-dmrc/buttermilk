"""Helper utilities for batch processing in Buttermilk.

This module contains helper functions and classes for extracting records
and managing batch operations.
"""

from typing import TYPE_CHECKING

from buttermilk._core.orchestrator import Orchestrator
from buttermilk.bm import logger

if TYPE_CHECKING:
    from buttermilk.runner.flowrunner import FlowRunner


class BatchRunnerHelper:
    """Helper class for batch processing operations.
    
    This class provides utility methods for extracting record IDs and
    managing batch operations across different orchestrator types.
    """

    @staticmethod
    async def get_record_ids_for_flow(flow_runner: "FlowRunner", flow_name: str) -> list[dict[str, str]]:
        """Get record IDs for a specific flow.
        
        Args:
            flow_runner: The FlowRunner instance
            flow_name: The name of the flow
            
        Returns:
            List of record ID dictionaries, each containing at least 'record_id'
            
        Raises:
            ValueError: If flow not found or record extraction fails

        """
        if flow_name not in flow_runner.flows:
            raise ValueError(f"Flow '{flow_name}' not found. Available flows: {list(flow_runner.flows.keys())}")

        orchestrator = flow_runner.flows[flow_name]

        try:
            # Try to use the get_record_ids method directly from the orchestrator
            if hasattr(orchestrator, "get_record_ids"):
                return await orchestrator.get_record_ids()

            # Fall back to a method that should be available on Orchestrator instances
            if isinstance(orchestrator, Orchestrator):
                return await orchestrator.get_record_ids()

            # If all else fails, try to look in the data sources
            if hasattr(orchestrator, "data") and orchestrator.data:
                # Handle standard record extraction from data sources
                record_ids = []
                for source_name, source_config in orchestrator.data.items():
                    # This is just a placeholder - actual implementation would depend on
                    # the structure of data sources in your specific application
                    logger.warning(f"Using fallback record ID extraction for data source '{source_name}'")
                    # Add dummy record as fallback
                    record_ids.append({"record_id": f"dummy-{source_name}", "name": f"Dummy Record for {source_name}"})

                return record_ids

        except Exception as e:
            logger.error(f"Failed to extract record IDs for flow '{flow_name}': {e}")
            raise ValueError(f"Failed to extract record IDs: {e}")

        raise ValueError(f"Could not find any method to extract record IDs for flow '{flow_name}'")
