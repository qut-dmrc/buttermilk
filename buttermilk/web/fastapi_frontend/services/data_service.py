from typing import Any

import pandas as pd

from buttermilk.bm import logger
from buttermilk.runner.helpers import prepare_step_df


class DataService:
    """Service for handling data processing operations"""

    @staticmethod
    async def get_criteria_for_flow(flow_name: str, flow_runner) -> list[str]:
        """Get criteria options for a specific flow
        
        Args:
            flow_name: The name of the flow
            flow_runner: The flow runner instance
            
        Returns:
            List[str]: The list of criteria options

        """
        criteria = []

        if not flow_name:
            logger.warning("Request for criteria missing 'flow' parameter.")
            return criteria

        try:
            if flow_name in flow_runner.flows:
                criteria = flow_runner.flows[flow_name].parameters.get("criteria", [])
            else:
                logger.warning(f"Flow '{flow_name}' not found.")
        except Exception as e:
            logger.error(f"Error loading criteria for flow '{flow_name}': {e}")

        return criteria

    @staticmethod
    async def get_records_for_flow(flow_name: str, flow_runner) -> list[str]:
        """Get record IDs for a specific flow
        
        Args:
            flow_name: The name of the flow
            flow_runner: The flow runner instance
            
        Returns:
            List[str]: The list of record IDs

        """
        record_ids = []

        if not flow_name:
            logger.warning("Request for records missing 'flow' parameter.")
            return record_ids

        try:
            if flow_name in flow_runner.flows:
                flow_obj = flow_runner.flows[flow_name]

                # Try to use get_record_ids method if it exists
                if hasattr(flow_obj, "get_record_ids") and callable(flow_obj.get_record_ids):
                    try:
                        # Check if it's an async function
                        import inspect
                        if inspect.iscoroutinefunction(flow_obj.get_record_ids):
                            # It's an async function
                            df = await flow_obj.get_record_ids()
                        else:
                            # It's a regular function
                            df = flow_obj.get_record_ids()

                        return DataService._extract_record_ids(df)
                    except Exception as e:
                        logger.error(f"Error calling get_record_ids: {e}")

                # Otherwise try to use the data directly
                elif hasattr(flow_obj, "data") and flow_obj.data:
                    # Check if we have mock data first
                    if "mock_data" in flow_obj.data and "record_ids" in flow_obj.data["mock_data"]:
                        return flow_obj.data["mock_data"]["record_ids"]
                    try:
                        logger.debug(f"Loading data for flow: {flow_name}")
                        df_dict = await prepare_step_df(flow_obj.data)
                        if df_dict and isinstance(df_dict, dict) and len(df_dict) > 0:
                            df = list(df_dict.values())[-1]
                            return DataService._extract_record_ids(df)
                    except Exception as e:
                        logger.error(f"Error preparing data: {e}")
            else:
                logger.warning(f"Flow '{flow_name}' not found.")

        except Exception as e:
            logger.error(f"Error loading data for flow '{flow_name}': {e}")

        return record_ids

    @staticmethod
    def _extract_record_ids(df: Any) -> list[str]:
        """Extract record IDs from a data object
        
        Args:
            df: The data object to extract record IDs from
            
        Returns:
            List[str]: The list of record IDs

        """
        record_ids = []

        try:
            # Try directly casting to list first
            if df is None:
                logger.warning("Data source returned None")
            elif isinstance(df, list):
                record_ids = df
            elif isinstance(df, pd.DataFrame):
                # Handle pandas DataFrame
                record_ids = df.index.tolist()
            else:
                # For any other type, try various approaches
                try:
                    # Try accessing .index attribute if it exists
                    if hasattr(df, "index"):
                        # Suppress type checking warnings with cast
                        from typing import cast
                        df_with_index = cast("pd.DataFrame", df)
                        record_ids = df_with_index.index.tolist()
                    # Try to_dict if it's a dataframe-like object
                    elif hasattr(df, "to_dict") and callable(df.to_dict):
                        # Suppress type checking warnings with cast
                        from typing import Any, cast
                        df_with_dict = cast("pd.DataFrame", df)
                        # Using intermediate variable to handle typing issues
                        raw_records = df_with_dict.to_dict("records")
                        records: list[dict[str, Any]] = []
                        for r in raw_records:
                            # Convert all keys to strings
                            records.append({str(k): v for k, v in r.items()})
                        record_ids = [str(r.get("id", i)) for i, r in enumerate(records)]
                    # Try direct iteration
                    elif hasattr(df, "__iter__"):
                        try:
                            # Convert to list only if it's truly iterable
                            record_ids = list(df)
                        except TypeError:
                            # Fall back to string representation if iteration fails
                            record_ids = [str(df)]
                    # Last resort: convert to string and use as single ID
                    else:
                        str_val = str(df)
                        if str_val:
                            record_ids = [str_val]
                        else:
                            logger.warning(f"Could not extract record IDs from: {type(df)}")
                except Exception as e:
                    logger.error(f"Error extracting record IDs: {e}")
                    # Try one last approach - convert to string
                    try:
                        str_val = str(df)
                        if str_val:
                            record_ids = [str_val]
                    except:
                        pass
        except Exception as e:
            logger.error(f"Unhandled error processing get_record_ids result: {e}")

        return record_ids

    @staticmethod
    async def get_run_history(flow_name: str, criteria: str, record_id: str, flow_runner) -> list[dict[str, Any]]:
        """Get run history for a specific flow, criteria, and record
        
        Args:
            flow_name: The name of the flow
            criteria: The criteria to filter by
            record_id: The record ID to filter by
            flow_runner: The flow runner instance
            
        Returns:
            List[Dict[str, Any]]: The list of run history records

        """
        if not flow_name or not criteria or not record_id:
            logger.warning("Request for run history missing required parameters.")
            return []

        try:
            # Format of the SQL query to get judge and synth runs
            sql = """
            SELECT
                *
            FROM
                `prosocial-443205.testing.flow_score_results`
            """
            # Execute the query
            if hasattr(flow_runner, "bm") and hasattr(flow_runner.bm, "run_query"):
                results_df = flow_runner.bm.run_query(sql)

                # If the results are empty
                if not isinstance(results_df, pd.DataFrame) or results_df.empty:
                    return []

                # Convert to list of dicts with string keys
                records = []
                for record in results_df.to_dict("records"):
                    records.append({str(k): v for k, v in record.items()})
                return records
            logger.warning("FlowRunner doesn't have the bm.run_query method.")
            return []

        except Exception as e:
            logger.error(f"Error loading run history: {e}")
            return []
