import datetime
from typing import Any, Protocol

from buttermilk._core.config import AgentConfig, DataSourceConfig
from buttermilk._core.contract import AgentInput, AgentTrace
from buttermilk._core.log import logger
from buttermilk._core.query import QueryRunner
from buttermilk._core.types import Record
from buttermilk.data.loaders import create_data_loader


class FlowRunner(Protocol):
    """Protocol for FlowRunner to avoid circular imports"""

    flows: dict[str, Any]

    async def get_records_for_flow(self) -> list[dict[str, Any]]:
        ...


class DataService:
    """Service for handling data-related operations"""

    @staticmethod
    def _convert_to_data_source_config(storage_config_raw: Any) -> DataSourceConfig:
        """Convert raw configuration (OmegaConf, dict, etc.) to DataSourceConfig.
        
        Args:
            storage_config_raw: Raw storage configuration from flow definition
            
        Returns:
            DataSourceConfig: Properly validated configuration object
        """
        if isinstance(storage_config_raw, DataSourceConfig):
            return storage_config_raw
        elif hasattr(storage_config_raw, '__dict__'):
            # Handle OmegaConf objects
            return DataSourceConfig(**dict(storage_config_raw))
        elif isinstance(storage_config_raw, dict):
            return DataSourceConfig(**storage_config_raw)
        else:
            # Fallback: try to convert to dict first
            return DataSourceConfig(**dict(storage_config_raw))

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
    async def get_datasets_for_flow(flow_name: str, flow_runner: FlowRunner) -> list[str]:
        """Get available dataset names for a flow

        Args:
            flow_name: The flow name
            flow_runner: The flow runner instance

        Returns:
            List[str]: The list of dataset names

        """
        try:
            return list(flow_runner.flows[flow_name].storage.keys())
        except Exception as e:
            logger.warning(f"Error getting datasets for flow {flow_name}: {e}")
            return []

    @staticmethod
    async def get_records_for_flow(flow_name: str, flow_runner: FlowRunner, include_scores: bool = False, dataset_name: str | None = None) -> list[Record]:
        """Get records for a flow

        Args:
            flow_name: The flow name
            flow_runner: The flow runner instance
            include_scores: Whether to include summary scores in metadata
            dataset_name: Optional specific dataset name to load from

        Returns:
            List[Record]: The list of Record objects with optional score summaries.

        """
        try:
            records = []

            # Get the appropriate storage configuration
            if dataset_name:
                if dataset_name not in flow_runner.flows[flow_name].storage:
                    raise ValueError(f"Dataset '{dataset_name}' not found in flow '{flow_name}'")
                storage_config_raw = flow_runner.flows[flow_name].storage[dataset_name]
            else:
                # Fallback to first storage configuration for backward compatibility
                storage_config_raw = list(flow_runner.flows[flow_name].storage.values())[0]

            # Convert OmegaConf/dict to proper DataSourceConfig
            storage_config = DataService._convert_to_data_source_config(storage_config_raw)

            loader = create_data_loader(storage_config)
            for record in loader:
                # Use the actual Record object, optionally enhancing metadata
                if include_scores:
                    # Add placeholder summary scores to metadata - in a real implementation,
                    # this would query BigQuery for actual scores
                    record.metadata["summary_scores"] = {
                        "off_shelf_accuracy": 0.75,
                        "custom_average": 0.635,
                        "total_evaluations": 8,
                        "has_detailed_responses": True
                    }

                records.append(record)

            return records

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

    @staticmethod
    async def get_record_by_id(record_id: str, flow_name: str, flow_runner, dataset_name: str | None = None) -> Record | None:
        """Get a single record by ID for a specific flow

        Args:
            record_id: The record ID to fetch
            flow_name: The flow name
            flow_runner: The flow runner instance
            dataset_name: Optional specific dataset name to load from

        Returns:
            Record object or None if not found
        """
        try:
            # Get the appropriate storage configuration
            if dataset_name:
                if dataset_name not in flow_runner.flows[flow_name].storage:
                    raise ValueError(f"Dataset '{dataset_name}' not found in flow '{flow_name}'")
                storage_config_raw = flow_runner.flows[flow_name].storage[dataset_name]
            else:
                # Fallback to first storage configuration for backward compatibility
                storage_config_raw = list(flow_runner.flows[flow_name].storage.values())[0]

            # Convert OmegaConf/dict to proper DataSourceConfig
            storage_config = DataService._convert_to_data_source_config(storage_config_raw)

            loader = create_data_loader(storage_config)
            for record in loader:
                if record.record_id == record_id:
                    # Enhance the existing Record object with computed metadata
                    record.metadata.update({
                        "dataset": flow_name,
                        "word_count": len(str(record.content).split()) if isinstance(record.content, str) else 0,
                        "char_count": len(str(record.content)) if isinstance(record.content, str) else 0
                    })
                    return record
            return None
        except Exception as e:
            logger.warning(f"Error getting record {record_id} for flow {flow_name}: {e}")
            return None

    @staticmethod
    async def get_scores_for_record(record_id: str, flow_name: str, session_id: str | None = None) -> list[AgentTrace]:
        """Get toxicity scores for a specific record as AgentTrace objects

        Args:
            record_id: The record ID
            flow_name: The flow name for table partitioning
            session_id: Optional session ID for filtering

        Returns:
            List[AgentTrace]: List of AgentTrace objects containing the scoring results
        """
        try:
            # Get BigQuery client from BM instance
            from buttermilk._core.dmrc import get_bm
            bm_instance = get_bm()
            bq_client = bm_instance.bq
            query_runner = QueryRunner(bq_client=bq_client)

            # Use BM instance's storage defaults for dataset configuration
            dataset_id = bm_instance.save.dataset_id

            # Build the query for scores
            where_clause = f"WHERE record_id = '{record_id}'"
            if session_id:
                where_clause += f" AND session_id = '{session_id}'"

            # Query the full AgentTrace data from the flows table
            sql = f"""
            SELECT
                session_id,
                call_id,
                timestamp,
                agent_info,
                inputs,
                outputs,
                metadata,
                run_info,
                parent_call_id,
                tracing_link,
                error,
                messages
            FROM `{bq_client.project}.{dataset_id}.flows`
            {where_clause}
            AND JSON_VALUE(agent_info, '$.role') IN ('JUDGE', 'SYNTHESISER', 'SCORERS')
            AND JSON_QUERY_ARRAY(inputs, '$.records') IS NOT NULL
            ORDER BY timestamp DESC
            """

            result = query_runner.run_query(sql, return_df=False)

            if not result:
                return []

            agent_traces = []
            for row in result:
                try:
                    # Reconstruct AgentTrace from the database row
                    # Parse JSON fields
                    import json
                    agent_info_data = json.loads(row["agent_info"]) if row["agent_info"] else {}
                    inputs_data = json.loads(row["inputs"]) if row["inputs"] else {}
                    outputs_data = row["outputs"]
                    metadata_data = json.loads(row["metadata"]) if row["metadata"] else {}
                    run_info_data = json.loads(row["run_info"]) if row["run_info"] else {}
                    messages_data = json.loads(row["messages"]) if row["messages"] else []
                    error_data = json.loads(row["error"]) if row["error"] else []

                    # Create AgentConfig
                    agent_config = AgentConfig(**agent_info_data)

                    # Create AgentInput
                    agent_input = AgentInput(
                        inputs=inputs_data.get("inputs", {}),
                        parameters=inputs_data.get("parameters", {}),
                        context=inputs_data.get("context", []),
                        records=[Record(**rec) for rec in inputs_data.get("records", [])],
                        parent_call_id=row.get("parent_call_id")
                    )

                    # Create AgentTrace
                    agent_trace = AgentTrace(
                        timestamp=row["timestamp"] if isinstance(row["timestamp"], datetime.datetime)
                                 else datetime.datetime.fromisoformat(row["timestamp"]),
                        call_id=row["call_id"],
                        agent_id=agent_config.agent_id,
                        metadata=metadata_data,
                        outputs=outputs_data,
                        run_info=run_info_data,
                        agent_info=agent_config,
                        session_id=row["session_id"],
                        parent_call_id=row.get("parent_call_id"),
                        tracing_link=row.get("tracing_link"),
                        inputs=agent_input,
                        messages=messages_data,
                        error=error_data
                    )

                    agent_traces.append(agent_trace)

                except Exception as e:
                    logger.warning(f"Error reconstructing AgentTrace from row: {e}")
                    continue

            return agent_traces

        except Exception as e:
            logger.error(f"Error getting scores for record {record_id}: {e}", exc_info=True)
            return []

    @staticmethod
    async def get_responses_for_record(record_id: str, flow_name: str, session_id: str | None = None, include_reasoning: bool = True) -> list[AgentTrace]:
        """Get detailed AI responses for a specific record as AgentTrace objects

        Args:
            record_id: The record ID
            flow_name: The flow name for table partitioning
            session_id: Optional session ID for filtering
            include_reasoning: Whether to include detailed reasoning (preserved for API compatibility)

        Returns:
            List[AgentTrace]: List of AgentTrace objects containing the detailed responses
        """
        try:
            # Get BigQuery client from BM instance
            from buttermilk._core.dmrc import get_bm
            bm_instance = get_bm()
            bq_client = bm_instance.bq
            query_runner = QueryRunner(bq_client=bq_client)

            # Use BM instance's storage defaults for dataset configuration
            dataset_id = bm_instance.save.dataset_id

            # Build the query for detailed responses
            where_clause = f"WHERE record_id = '{record_id}'"
            if session_id:
                where_clause += f" AND session_id = '{session_id}'"

            # Reuse the same query as get_scores_for_record since we want the full AgentTrace data
            # The include_reasoning parameter is ignored since AgentTrace contains all data
            sql = f"""
            SELECT
                session_id,
                call_id,
                timestamp,
                agent_info,
                inputs,
                outputs,
                metadata,
                run_info,
                parent_call_id,
                tracing_link,
                error,
                messages
            FROM `{bq_client.project}.{dataset_id}.flows`
            {where_clause}
            AND JSON_VALUE(agent_info, '$.role') IN ('JUDGE', 'SYNTHESISER', 'SCORERS')
            AND JSON_QUERY_ARRAY(inputs, '$.records') IS NOT NULL
            ORDER BY timestamp DESC
            """

            result = query_runner.run_query(sql, return_df=False)

            if not result:
                return []

            agent_traces = []
            for row in result:
                try:
                    # Reconstruct AgentTrace from the database row (same logic as get_scores_for_record)
                    import json
                    agent_info_data = json.loads(row["agent_info"]) if row["agent_info"] else {}
                    inputs_data = json.loads(row["inputs"]) if row["inputs"] else {}
                    outputs_data = row["outputs"]
                    metadata_data = json.loads(row["metadata"]) if row["metadata"] else {}
                    run_info_data = json.loads(row["run_info"]) if row["run_info"] else {}
                    messages_data = json.loads(row["messages"]) if row["messages"] else []
                    error_data = json.loads(row["error"]) if row["error"] else []

                    # Create AgentConfig
                    agent_config = AgentConfig(**agent_info_data)

                    # Create AgentInput
                    agent_input = AgentInput(
                        inputs=inputs_data.get("inputs", {}),
                        parameters=inputs_data.get("parameters", {}),
                        context=inputs_data.get("context", []),
                        records=[Record(**rec) for rec in inputs_data.get("records", [])],
                        parent_call_id=row.get("parent_call_id")
                    )

                    # Create AgentTrace
                    agent_trace = AgentTrace(
                        timestamp=row["timestamp"] if isinstance(row["timestamp"], datetime.datetime)
                                 else datetime.datetime.fromisoformat(row["timestamp"]),
                        call_id=row["call_id"],
                        agent_id=agent_config.agent_id,  # Use the correct attribute name
                        metadata=metadata_data,
                        outputs=outputs_data,
                        run_info=run_info_data,
                        agent_info=agent_config,
                        session_id=row["session_id"],
                        parent_call_id=row.get("parent_call_id"),
                        tracing_link=row.get("tracing_link"),
                        inputs=agent_input,
                        messages=messages_data,
                        error=error_data
                    )

                    agent_traces.append(agent_trace)

                except Exception as e:
                    logger.warning(f"Error reconstructing AgentTrace from row: {e}")
                    continue

            return agent_traces

        except Exception as e:
            logger.error(f"Error getting responses for record {record_id}: {e}", exc_info=True)
            return []
