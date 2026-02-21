import datetime
from typing import Any, Protocol

from buttermilk import bm, logger
from buttermilk._core.config import AgentConfig, SessionConfig
from buttermilk._core.contract import AgentInput, ExecutionTrace
from buttermilk._core.query import QueryRunner
from buttermilk._core.types import Record


class FlowRunner(Protocol):
    """Protocol for FlowRunner to avoid circular imports"""

    flows: dict[str, Any]

    async def get_records_for_flow(self) -> list[dict[str, Any]]: ...


class DataService:
    """Service for handling data-related operations"""

    # Session configuration with defaults from YAML config
    _session_config = SessionConfig()

    @staticmethod
<<<<<<< HEAD
    async def get_criteria_for_flow(flow_name: str, flow_runner: FlowRunner) -> list[str]:
=======
    async def get_criteria_for_flow(
        flow_name: str, flow_runner: FlowRunner
    ) -> list[str]:
>>>>>>> origin/stable
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
        try:
            return list(flow_runner.flows[flow_name].parameters.get("model", []))
        except Exception as e:
            logger.warning(f"Error getting models for flow {flow_name}: {e}")
            return []

    @staticmethod
<<<<<<< HEAD
    async def get_datasets_for_flow(flow_name: str, flow_runner: FlowRunner) -> list[str]:
=======
    async def get_datasets_for_flow(
        flow_name: str, flow_runner: FlowRunner
    ) -> list[str]:
>>>>>>> origin/stable
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
    async def get_records_for_flow(
        flow_name: str,
        flow_runner: FlowRunner,
        dataset_key: str,
        include_scores: bool = False,
    ) -> list[Record]:
        """Get records for a flow

        Args:
            flow_name: The flow name
            flow_runner: The flow runner instance
            include_scores: Whether to include summary scores in metadata
            dataset_key: Required dataset configuration key (the top-level key in the storage config)

        Returns:
            List[Record]: The list of Record objects with optional score summaries.

        """
        try:
            records = []

            # Get the specified storage configuration
            if dataset_key not in flow_runner.flows[flow_name].storage:
                available_datasets = list(flow_runner.flows[flow_name].storage.keys())
<<<<<<< HEAD
                raise ValueError(f"Dataset '{dataset_key}' not found in flow '{flow_name}'. Available datasets: {available_datasets}")
=======
                raise ValueError(
                    f"Dataset '{dataset_key}' not found in flow '{flow_name}'. Available datasets: {available_datasets}"
                )
>>>>>>> origin/stable

            # Use unified storage system instead of deprecated create_data_loader
            storage = bm.get_storage(flow_runner.flows[flow_name].storage[dataset_key])

            for record in storage:
                # Use the actual Record object, optionally enhancing metadata
                if include_scores:
<<<<<<< HEAD
                    raise NotImplementedError("Score inclusion feature is not yet implemented")
=======
                    raise NotImplementedError(
                        "Score inclusion feature is not yet implemented"
                    )
>>>>>>> origin/stable

                records.append(record)

            return records

        except Exception as e:
            logger.error(f"Error getting records for flow {flow_name}: {e}")
            return []

    @staticmethod
<<<<<<< HEAD
    async def get_run_history(flow_name: str, criteria: str, record_id: str, flow_runner) -> list[dict[str, Any]]:
=======
    async def get_run_history(
        flow_name: str, criteria: str, record_id: str, flow_runner
    ) -> list[dict[str, Any]]:
>>>>>>> origin/stable
        """Get run history for a flow, criteria, and record

        Args:
            flow_name: The flow name
            criteria: The criteria
            record_id: The record ID
            flow_runner: The flow runner instance

        Returns:
            List[Dict[str, Any]]: The list of history entries

        """
        raise NotImplementedError("Run history retrieval is not yet implemented")

    @classmethod
<<<<<<< HEAD
    def safely_get_session_data(cls, websocket_manager, session_id: str) -> dict[str, Any]:
=======
    def safely_get_session_data(
        cls, websocket_manager, session_id: str
    ) -> dict[str, Any]:
>>>>>>> origin/stable
        """Safely get session data, ensuring default values if keys don't exist

        Args:
            websocket_manager: The WebSocketManager instance
            session_id: The session ID

        Returns:
            Dict[str, Any]: The sanitized session data

        """
        try:
            # Get default values from configuration
            defaults = cls._session_config.defaults.model_dump()

            if not session_id or not hasattr(websocket_manager, "session_data"):
                return defaults

            session = websocket_manager.session_data.get(session_id)
            if not session:
                return defaults

            # Ensure progress has all required fields with defaults from config
            progress = session.get("progress", {})
            progress_defaults = cls._session_config.defaults.progress.model_dump()
            for key, default_value in progress_defaults.items():
                progress.setdefault(key, default_value)

            # Sanitize the response to ensure all expected keys are present
            return {
<<<<<<< HEAD
                "scores": defaults["scores"],  # This will be populated later by MessageService
                "outcomes": defaults["outcomes"],  # This will be populated later by MessageService
                "pending_agents": progress.get("pending_agents", defaults["pending_agents"]),
=======
                "scores": defaults[
                    "scores"
                ],  # This will be populated later by MessageService
                "outcomes": defaults[
                    "outcomes"
                ],  # This will be populated later by MessageService
                "pending_agents": progress.get(
                    "pending_agents", defaults["pending_agents"]
                ),
>>>>>>> origin/stable
                "progress": progress,
            }
        except Exception as e:
            logger.warning(f"Error safely getting session data: {e}")
            # Return defaults from configuration on error
            return cls._session_config.defaults.model_dump()

    @staticmethod
<<<<<<< HEAD
    async def get_record_by_id(record_id: str, flow_name: str, flow_runner, dataset_key: str | None = None) -> Record | None:
=======
    async def get_record_by_id(
        record_id: str, flow_name: str, flow_runner, dataset_key: str | None = None
    ) -> Record | None:
>>>>>>> origin/stable
        """Get a single record by ID for a specific flow

        Args:
            record_id: The record ID to fetch
            flow_name: The flow name
            flow_runner: The flow runner instance
            dataset_key: Optional specific dataset configuration key to load from

        Returns:
            Record object or None if not found

        """
        try:
            # Get the appropriate storage configuration
            if dataset_key:
                if dataset_key not in flow_runner.flows[flow_name].storage:
<<<<<<< HEAD
                    raise ValueError(f"Dataset '{dataset_key}' not found in flow '{flow_name}'")
                storage_config_raw = flow_runner.flows[flow_name].storage[dataset_key]
            else:
                # Fallback to first storage configuration for backward compatibility
                storage_config_raw = list(flow_runner.flows[flow_name].storage.values())[0]
=======
                    raise ValueError(
                        f"Dataset '{dataset_key}' not found in flow '{flow_name}'"
                    )
                storage_config_raw = flow_runner.flows[flow_name].storage[dataset_key]
            else:
                # Fallback to first storage configuration for backward compatibility
                storage_config_raw = list(
                    flow_runner.flows[flow_name].storage.values()
                )[0]
>>>>>>> origin/stable

            # Use unified storage system instead of deprecated create_data_loader
            storage = bm.get_storage(storage_config_raw)

            # Use storage's get_record_by_id method (handles iteration internally)
            record = storage.get_record_by_id(record_id)

            if record:
                # Enhance the existing Record object with computed metadata
                record.metadata.update(
                    {
                        "dataset": flow_name,
<<<<<<< HEAD
                        "word_count": len(str(record.content).split()) if isinstance(record.content, str) else 0,
                        "char_count": len(str(record.content)) if isinstance(record.content, str) else 0,
=======
                        "word_count": len(str(record.content).split())
                        if isinstance(record.content, str)
                        else 0,
                        "char_count": len(str(record.content))
                        if isinstance(record.content, str)
                        else 0,
>>>>>>> origin/stable
                    },
                )
                return record
            return None
        except Exception as e:
<<<<<<< HEAD
            logger.warning(f"Error getting record {record_id} for flow {flow_name}: {e}")
=======
            logger.warning(
                f"Error getting record {record_id} for flow {flow_name}: {e}"
            )
>>>>>>> origin/stable
            return None

    @staticmethod
    def _reconstruct_agent_trace_from_row(row: dict) -> ExecutionTrace:
        """Convenience method to reconstruct ExecutionTrace from database row.

        This method centralizes the logic for reconstructing ExecutionTrace objects
        from database query results, eliminating code duplication.

        Args:
            row: Database row containing ExecutionTrace data

        Returns:
            ExecutionTrace object reconstructed from the row data

        Raises:
            Exception: If reconstruction fails due to invalid data

        """
        import json

        # Parse JSON fields
        agent_info_data = json.loads(row["agent_info"]) if row["agent_info"] else {}
        inputs_data = json.loads(row["inputs"]) if row["inputs"] else {}
        outputs_data = row["outputs"]
        metadata_data = json.loads(row["metadata"]) if row["metadata"] else {}
<<<<<<< HEAD
        session_info_data = json.loads(row["session_info"]) if row["session_info"] else {}
=======
        session_info_data = (
            json.loads(row["session_info"]) if row["session_info"] else {}
        )
>>>>>>> origin/stable
        messages_data = json.loads(row["messages"]) if row["messages"] else []
        error_data = json.loads(row["error"]) if row["error"] else []

        # Create AgentConfig
        agent_config = AgentConfig(**agent_info_data)

        # Create AgentInput
        agent_input = AgentInput(
            inputs=inputs_data.get("inputs", {}),
            parameters=inputs_data.get("parameters", {}),
            context=inputs_data.get("context", []),
            record=Record(**inputs_data.get("records", {})),
            parent_call_id=row.get("parent_call_id"),
        )

        # Create ExecutionTrace
        agent_trace = ExecutionTrace(
<<<<<<< HEAD
            timestamp=row["timestamp"] if isinstance(row["timestamp"], datetime.datetime) else datetime.datetime.fromisoformat(row["timestamp"]),
=======
            timestamp=row["timestamp"]
            if isinstance(row["timestamp"], datetime.datetime)
            else datetime.datetime.fromisoformat(row["timestamp"]),
>>>>>>> origin/stable
            call_id=row["call_id"],
            metadata=metadata_data,
            outputs=outputs_data,
            session_info=session_info_data,
            agent_info=agent_config,
            session_id=row["session_id"],
            parent_call_id=row.get("parent_call_id"),
            inputs=agent_input,
            messages=messages_data,
            error=error_data,
        )

        return agent_trace

    @staticmethod
    async def get_scores_for_record(
        record_id: str,
        flow_name: str,
        flow_runner: FlowRunner,
        session_id: str | None = None,
    ) -> list[ExecutionTrace]:
        """Get toxicity scores for a specific record as ExecutionTrace objects

        Args:
            record_id: The record ID
            flow_name: The flow name to get save configuration from
            flow_runner: FlowRunner instance to access flow configuration
            session_id: Optional session ID for filtering

        Returns:

        """
        try:
            bq_client = bm.bq
<<<<<<< HEAD
            query_runner = QueryRunner(bq_client=bq_client)  # Use the provided flow runner to access flow configuration and save settings

            # Get the save configuration from the flow parameters
            if flow_name not in flow_runner.flows:
                raise ValueError(f"Flow '{flow_name}' not found in flow runner. Available flows: {list(flow_runner.flows.keys())}")
=======
            query_runner = QueryRunner(
                bq_client=bq_client
            )  # Use the provided flow runner to access flow configuration and save settings

            # Get the save configuration from the flow parameters
            if flow_name not in flow_runner.flows:
                raise ValueError(
                    f"Flow '{flow_name}' not found in flow runner. Available flows: {list(flow_runner.flows.keys())}"
                )
>>>>>>> origin/stable

            flow_config = flow_runner.flows[flow_name]
            save_config = flow_config.parameters.get("save", {})

            if not save_config or save_config.get("type") != "bigquery":
<<<<<<< HEAD
                raise ValueError(f"Flow '{flow_name}' does not have BigQuery save configuration. Cannot query scores.")
=======
                raise ValueError(
                    f"Flow '{flow_name}' does not have BigQuery save configuration. Cannot query scores."
                )
>>>>>>> origin/stable

            dataset_id = save_config.get("dataset_id")
            table_id = save_config.get("table_id")

            if not dataset_id:
<<<<<<< HEAD
                raise ValueError(f"Flow '{flow_name}' is missing required 'dataset_id' in save configuration")
            if not table_id:
                raise ValueError(f"Flow '{flow_name}' is missing required 'table_id' in save configuration")
=======
                raise ValueError(
                    f"Flow '{flow_name}' is missing required 'dataset_id' in save configuration"
                )
            if not table_id:
                raise ValueError(
                    f"Flow '{flow_name}' is missing required 'table_id' in save configuration"
                )
>>>>>>> origin/stable

            # Build the query for scores using the correct table reference
            where_clause = f"WHERE record_id = '{record_id}'"
            if session_id:
                where_clause += f" AND session_id = '{session_id}'"

            # Query the full ExecutionTrace data from the configured flows table
            sql = f"""
                SELECT
                    session_id,
                    call_id,
                    timestamp,
                    agent_info,
                    inputs,
                    outputs,
                    metadata,
                    session_info,
                    parent_call_id,
                    tracing_link,
                    error,
                    messages
                FROM `{bq_client.project}.{dataset_id}.{table_id}`
                {where_clause}
                AND JSON_VALUE(agent_info, '$.role') IN ('JUDGE', 'SYNTHESISER', 'SCORERS')
                AND JSON_QUERY_ARRAY(inputs, '$.record') IS NOT NULL
                ORDER BY timestamp DESC
            """

            result = query_runner.run_query(sql, return_df=False)

            if not result:
                return []

            agent_traces = []
            for row in result:
                try:
                    agent_trace = DataService._reconstruct_agent_trace_from_row(row)
                    agent_traces.append(agent_trace)
                except Exception as e:
                    logger.warning(f"Error reconstructing ExecutionTrace from row: {e}")
                    continue

            return agent_traces

        except Exception as e:
<<<<<<< HEAD
            logger.error(f"Error getting scores for record {record_id}: {e}", exc_info=True)
=======
            logger.error(
                f"Error getting scores for record {record_id}: {e}", exc_info=True
            )
>>>>>>> origin/stable
            return []

    @staticmethod
    async def get_responses_for_record(
        record_id: str,
        flow_name: str,
        flow_runner: FlowRunner,
        session_id: str | None = None,
        include_reasoning: bool = True,
    ) -> list[ExecutionTrace]:
        """Get detailed AI responses for a specific record as ExecutionTrace objects

        Args:
            record_id: The record ID
            flow_name: The flow name to get save configuration from
            flow_runner: FlowRunner instance to access flow configuration
            session_id: Optional session ID for filtering
            include_reasoning: Whether to include detailed reasoning (preserved for API compatibility)

        Returns:
            List[ExecutionTrace]: List of ExecutionTrace objects containing the detailed responses

        """
        try:
            # Get BigQuery client from BM instance
            bq_client = bm.bq
            query_runner = QueryRunner(bq_client=bq_client)

            # Use the provided flow runner to access flow configuration and save settings

            # Get the save configuration from the flow parameters
            if flow_name not in flow_runner.flows:
<<<<<<< HEAD
                raise ValueError(f"Flow '{flow_name}' not found in flow runner. Available flows: {list(flow_runner.flows.keys())}")
=======
                raise ValueError(
                    f"Flow '{flow_name}' not found in flow runner. Available flows: {list(flow_runner.flows.keys())}"
                )
>>>>>>> origin/stable

            flow_config = flow_runner.flows[flow_name]
            save_config = flow_config.parameters.get("save", {})

            if not save_config or save_config.get("type") != "bigquery":
<<<<<<< HEAD
                raise ValueError(f"Flow '{flow_name}' does not have BigQuery save configuration. Cannot query scores.")
=======
                raise ValueError(
                    f"Flow '{flow_name}' does not have BigQuery save configuration. Cannot query scores."
                )
>>>>>>> origin/stable

            dataset_id = save_config.get("dataset_id")
            table_id = save_config.get("table_id")

            if not dataset_id:
<<<<<<< HEAD
                raise ValueError(f"Flow '{flow_name}' is missing required 'dataset_id' in save configuration")
            if not table_id:
                raise ValueError(f"Flow '{flow_name}' is missing required 'table_id' in save configuration")
=======
                raise ValueError(
                    f"Flow '{flow_name}' is missing required 'dataset_id' in save configuration"
                )
            if not table_id:
                raise ValueError(
                    f"Flow '{flow_name}' is missing required 'table_id' in save configuration"
                )
>>>>>>> origin/stable

            # Build the query for detailed responses
            where_clause = f"WHERE record_id = '{record_id}'"
            if session_id:
                where_clause += f" AND session_id = '{session_id}'"

            # Reuse the same query as get_scores_for_record since we want the full ExecutionTrace data
            # The include_reasoning parameter is ignored since ExecutionTrace contains all data
            sql = f"""
            SELECT
                session_id,
                call_id,
                timestamp,
                agent_info,
                inputs,
                outputs,
                metadata,
                session_info,
                parent_call_id,
                tracing_link,
                error,
                messages
            FROM `{bq_client.project}.{dataset_id}.{table_id}`
            {where_clause}
            AND JSON_VALUE(agent_info, '$.role') IN ('JUDGE', 'SYNTHESISER', 'SCORERS')
            AND JSON_QUERY_ARRAY(inputs, '$.record') IS NOT NULL
            ORDER BY timestamp DESC
            """

            result = query_runner.run_query(sql, return_df=False)

            if not result:
                return []

            agent_traces = []
            for row in result:
                try:
                    agent_trace = DataService._reconstruct_agent_trace_from_row(row)
                    agent_traces.append(agent_trace)
                except Exception as e:
                    logger.warning(f"Error reconstructing ExecutionTrace from row: {e}")
                    continue

            return agent_traces

        except Exception as e:
<<<<<<< HEAD
            logger.error(f"Error getting responses for record {record_id}: {e}", exc_info=True)
=======
            logger.error(
                f"Error getting responses for record {record_id}: {e}", exc_info=True
            )
>>>>>>> origin/stable
            return []
