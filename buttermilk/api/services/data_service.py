from typing import Any, Protocol

from buttermilk.bm import logger
from buttermilk.runner.helpers import prepare_step_df


class FlowRunner(Protocol):
    """Protocol for FlowRunner to avoid circular imports"""

    flows: dict[str, Any]

    async def get_records_for_flow(self) -> list[dict[str, Any]]:
        ...


class DataService:
    """Service for handling data-related operations"""

    @staticmethod
    async def get_criteria_for_flow(flow_name: str, flow_runner: FlowRunner) -> list[str]:
        """Get criteria for a flow
        
        Args:
            flow_name: The flow name
            flow_runner: The flow runner instance
            
        Returns:
            List[str]: The list of criteria

        """
        try:
            return list(flow_runner.flows[flow_name].parameters.get("criteria", []))
        except Exception as e:
            logger.warning(f"Error getting criteria for flow {flow_name}: {e}")
            return []

    @staticmethod
    async def get_models_for_flow(flow_name: str, flow_runner: FlowRunner) -> list[str]:
        """Get models for a flow
        
        Args:
            flow_name: The flow name
            flow_runner: The flow runner instance
            
        Returns:
            List[str]: The list of models

        """
        return ["lite", "full"]
        try:
            return list(flow_runner.flows[flow_name].parameters.get("model", []))
        except Exception as e:
            logger.warning(f"Error getting models for flow {flow_name}: {e}")
            return []

    @staticmethod
    async def get_records_for_flow(flow_name: str, flow_runner: FlowRunner) -> list[dict[str, str]]:
        """Get records for a flow
        
        Args:
            flow_name: The flow name
            flow_runner: The flow runner instance
            
        Returns:
            List[dict[str, str]]: The list of record IDs and names.

        """
        try:
            record_ids = []
            datasets = await prepare_step_df(flow_runner.flows[flow_name].data)

            for name, df in datasets.items():
                df_temp = df.copy().reset_index()
                if "name" not in df_temp.columns:
                    df_temp["name"] = df_temp["record_id"]

                record_ids.extend(df_temp[["record_id", "name"]].to_dict(orient="records"))

            return record_ids

        except Exception as e:
            logger.warning(f"Error getting records for flow {flow_name}: {e}")
            return []

    @staticmethod
    async def get_run_history(flow_name: str, criteria: str, record_id: str, flow_runner) -> list[dict[str, Any]]:
        """Get run history for a flow, criteria, and record
        
        Args:
            flow_name: The flow name
            criteria: The criteria
            record_id: The record ID
            flow_runner: The flow runner instance
            
        Returns:
            List[Dict[str, Any]]: The list of history entries

        """
        try:
            # This is a placeholder for actual history retrieval
            # In a real implementation, this would query a data store

            # For the demo, return empty list
            return []
        except Exception as e:
            logger.warning(f"Error getting run history: {e}")
            return []

    @staticmethod
    def safely_get_session_data(websocket_manager, session_id: str) -> dict[str, Any]:
        """Safely get session data, ensuring default values if keys don't exist
        
        Args:
            websocket_manager: The WebSocketManager instance
            session_id: The session ID
            
        Returns:
            Dict[str, Any]: The sanitized session data

        """
        try:
            if not session_id or not hasattr(websocket_manager, "session_data"):
                return {
                    "scores": {},
                    "outcomes": [],
                    "pending_agents": [],
                    "progress": {"current_step": 0, "total_steps": 100, "status": "waiting"},
                }

            session = websocket_manager.session_data.get(session_id)
            if not session:
                return {
                    "scores": {},
                    "outcomes": [],
                    "pending_agents": [],
                    "progress": {"current_step": 0, "total_steps": 100, "status": "waiting"},
                }

            # Ensure progress has all required fields with defaults
            progress = session.get("progress", {})
            progress.setdefault("current_step", 0)
            progress.setdefault("total_steps", 100)
            progress.setdefault("status", "waiting")
            progress.setdefault("pending_agents", [])

            # Sanitize the response to ensure all expected keys are present
            return {
                "scores": {},      # This will be populated later by MessageService
                "outcomes": [],    # This will be populated later by MessageService
                "pending_agents": progress.get("pending_agents", []),
                "progress": progress,
            }
        except Exception as e:
            logger.warning(f"Error safely getting session data: {e}")
            return {
                "scores": {},
                "outcomes": [],
                "pending_agents": [],
                "progress": {"current_step": 0, "total_steps": 100, "status": "waiting"},
            }
