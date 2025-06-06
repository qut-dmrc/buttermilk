from typing import Any, Protocol

from buttermilk._core.log import logger
from buttermilk._core.query import QueryRunner
from buttermilk.data.loaders import create_data_loader


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
    async def get_records_for_flow(flow_name: str, flow_runner: FlowRunner, include_scores: bool = False) -> list[dict[str, str]]:
        """Get records for a flow

        Args:
            flow_name: The flow name
            flow_runner: The flow runner instance

        Returns:
            List[dict[str, str]]: The list of record IDs and names.

        """
        try:
            records = []

            loader = create_data_loader(list(flow_runner.flows[flow_name].data.values())[0])
            for record in loader:
                record_data = {"record_id": record.record_id, "name": record.title}

                if include_scores:
                    # Add placeholder summary scores - in a real implementation,
                    # this would query BigQuery for actual scores
                    record_data["summary_scores"] = {
                        "off_shelf_accuracy": 0.75,
                        "custom_average": 0.635,
                        "total_evaluations": 8,
                        "has_detailed_responses": True
                    }

                records.append(record_data)

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
    async def get_record_by_id(record_id: str, flow_name: str, flow_runner) -> dict[str, Any] | None:
        """Get a single record by ID for a specific flow

        Args:
            record_id: The record ID to fetch
            flow_name: The flow name
            flow_runner: The flow runner instance

        Returns:
            Dict containing record data or None if not found
        """
        try:
            loader = create_data_loader(list(flow_runner.flows[flow_name].data.values())[0])
            for record in loader:
                if record.record_id == record_id:
                    return {
                        "id": record.record_id,
                        "name": getattr(record, "title", record.record_id),
                        "content": record.content,
                        "metadata": {
                            "created_at": getattr(record, "created_at", None),
                            "dataset": flow_name,
                            "word_count": len(str(record.content).split()) if isinstance(record.content, str) else 0,
                            "char_count": len(str(record.content)) if isinstance(record.content, str) else 0
                        }
                    }
            return None
        except Exception as e:
            logger.warning(f"Error getting record {record_id} for flow {flow_name}: {e}")
            return None

    @staticmethod
    async def get_scores_for_record(record_id: str, flow_name: str, bm_instance, session_id: str | None = None) -> dict[str, Any]:
        """Get toxicity scores for a specific record

        Args:
            record_id: The record ID
            flow_name: The flow name for table partitioning
            bm_instance: Buttermilk instance for BigQuery access
            session_id: Optional session ID for filtering

        Returns:
            Dict containing off-shelf and custom results with summary
        """
        try:
            # Get BigQuery client from BM instance
            bq_client = bm_instance.bq
            query_runner = QueryRunner(bq_client=bq_client)

            # Build the query for scores
            where_clause = f"WHERE record_id = '{record_id}'"
            if session_id:
                where_clause += f" AND session_id = '{session_id}'"

            sql = f"""
            SELECT
                session_id,
                call_id,
                scorer_call_id,
                record_id,
                judge,
                judge_model,
                judge_template,
                judge_criteria,
                violating,
                confidence,
                correctness,
                scorer,
                scoring_model,
                scoring_template
            FROM `{bq_client.project}.buttermilk.judge_scores`
            {where_clause}
            ORDER BY timestamp DESC
            """

            df = query_runner.run_query(sql, return_df=True)

            if df.empty:
                return {
                    "record_id": record_id,
                    "off_shelf_results": {},
                    "custom_results": {},
                    "summary": {
                        "off_shelf_accuracy": 0.0,
                        "custom_average_score": 0.0,
                        "total_evaluations": 0,
                        "agreement_rate": 0.0
                    }
                }

            # Transform data to API format
            off_shelf_results = {}
            custom_results = {}

            for _, row in df.iterrows():
                # Map confidence levels to numeric values
                confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
                confidence_val = confidence_map.get(row.get("confidence", "medium"), 0.7)

                # Determine if this is off-shelf or custom
                judge_name = row.get("judge", "")
                model_name = row.get("judge_model", "")

                # Custom results (from buttermilk agents)
                if row.get("scorer") or row.get("correctness") is not None:
                    agent_key = f"{judge_name}-{model_name}"
                    custom_results[agent_key] = {
                        "step": "judge" if not row.get("scorer") else "score",
                        "score": float(row.get("correctness", 0.0)) if row.get("correctness") is not None else (1.0 if row.get("violating") else 0.0),
                        "model": model_name,
                        "template": row.get("judge_template", ""),
                        "criteria": row.get("judge_criteria", "")
                    }
                else:
                    # Off-shelf style results
                    model_key = model_name or judge_name
                    if model_key:
                        off_shelf_results[model_key] = {
                            "correct": bool(row.get("violating", False)),
                            "score": 1.0 if row.get("violating") else 0.0,
                            "label": "TOXIC" if row.get("violating") else "SAFE",
                            "confidence": confidence_val,
                            "model_version": model_name
                        }

            # Calculate summary statistics
            total_evaluations = len(df)
            custom_scores = [r["score"] for r in custom_results.values() if "score" in r]
            custom_average = sum(custom_scores) / len(custom_scores) if custom_scores else 0.0

            # Simple agreement calculation (could be more sophisticated)
            predictions = [row.get("violating", False) for _, row in df.iterrows()]
            agreement_rate = len(set(predictions)) / len(predictions) if predictions else 0.0

            return {
                "record_id": record_id,
                "off_shelf_results": off_shelf_results,
                "custom_results": custom_results,
                "summary": {
                    "off_shelf_accuracy": 0.75,  # Placeholder - would need ground truth
                    "custom_average_score": custom_average,
                    "total_evaluations": total_evaluations,
                    "agreement_rate": 1.0 - agreement_rate  # Convert to agreement percentage
                }
            }

        except Exception as e:
            logger.error(f"Error getting scores for record {record_id}: {e}", exc_info=True)
            return {
                "record_id": record_id,
                "off_shelf_results": {},
                "custom_results": {},
                "summary": {
                    "off_shelf_accuracy": 0.0,
                    "custom_average_score": 0.0,
                    "total_evaluations": 0,
                    "agreement_rate": 0.0
                }
            }

    @staticmethod
    async def get_responses_for_record(record_id: str, flow_name: str, bm_instance, session_id: str | None = None, include_reasoning: bool = True) -> dict[str, Any]:
        """Get detailed AI responses for a specific record

        Args:
            record_id: The record ID
            flow_name: The flow name for table partitioning
            bm_instance: Buttermilk instance for BigQuery access
            session_id: Optional session ID for filtering
            include_reasoning: Whether to include detailed reasoning

        Returns:
            Dict containing detailed agent responses
        """
        try:
            # Get BigQuery client from BM instance
            bq_client = bm_instance.bq
            query_runner = QueryRunner(bq_client=bq_client)

            # Build the query for detailed responses
            where_clause = f"WHERE record_id = '{record_id}'"
            if session_id:
                where_clause += f" AND session_id = '{session_id}'"

            sql = f"""
            SELECT
                session_id,
                call_id,
                timestamp,
                record_id,
                judge,
                judge_model,
                judge_template,
                judge_criteria,
                judge_role,
                reasons,
                conclusion,
                violating,
                confidence
            FROM `{bq_client.project}.buttermilk.judge_reasons`
            {where_clause}
            ORDER BY timestamp DESC
            """

            df = query_runner.run_query(sql, return_df=True)

            responses = []
            for _, row in df.iterrows():
                # Map confidence to numeric
                confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
                confidence_val = confidence_map.get(row.get("confidence", "medium"), 0.7)

                response = {
                    "agent": f"{row.get('judge', '')}-{row.get('judge_model', '')}",
                    "type": row.get("judge_role", "").lower() or "judge",
                    "model": row.get("judge_model", ""),
                    "content": row.get("conclusion", ""),
                    "score": 1.0 if row.get("violating") else 0.0,
                    "criteria_used": row.get("judge_criteria", ""),
                    "template": row.get("judge_template", ""),
                    "timestamp": row.get("timestamp").isoformat() if row.get("timestamp") else None,
                    "confidence": confidence_val,
                    "prediction": bool(row.get("violating", False))
                }

                if include_reasoning and row.get("reasons"):
                    # Join reasons array into a single string
                    reasons = row.get("reasons", [])
                    if isinstance(reasons, list):
                        response["reasoning"] = " ".join(reasons)
                    else:
                        response["reasoning"] = str(reasons)

                responses.append(response)

            return {
                "record_id": record_id,
                "responses": responses
            }

        except Exception as e:
            logger.error(f"Error getting responses for record {record_id}: {e}", exc_info=True)
            return {
                "record_id": record_id,
                "responses": []
            }
