"""BigQuery operations and cost tracking utilities."""

import datetime
from typing import Any

import humanfriendly
import pandas as pd
from google.cloud import bigquery
from pydantic import BaseModel

from buttermilk._core.log import logger

# https://cloud.google.com/bigquery/pricing
GOOGLE_BQ_PRICE_PER_BYTE = 5 / 10e12  # $5 per TB


class QueryRunner(BaseModel):
    """Manages BigQuery operations with cost tracking."""

    class Config:
        arbitrary_types_allowed = True

    _bq_client: bigquery.Client

    def __init__(self, bq_client: bigquery.Client):
        """Initialize the query runner with a BigQuery client.
        
        Args:
            bq_client: Authenticated BigQuery client

        """
        super().__init__(_bq_client=bq_client)

    def run_query(
        self,
        sql: str,
        destination: str = None,
        overwrite: bool = False,
        do_not_return_results: bool = False,
        save_to_gcs: bool = False,
        return_df: bool = True,
        save_dir: str = None,
    ) -> pd.DataFrame | Any | bool:
        """Run a BigQuery SQL query with cost tracking.
        
        Args:
            sql: SQL query to execute
            destination: Optional destination table for query results
            overwrite: Whether to overwrite the destination table if it exists
            do_not_return_results: If True, don't fetch results (for DDL/DML)
            save_to_gcs: If True, save results to Google Cloud Storage
            return_df: If True, return results as a pandas DataFrame
            save_dir: Directory to save results to if save_to_gcs is True
            
        Returns:
            Query results as pandas DataFrame, BigQuery result object, or boolean status
            
        Raises:
            RuntimeError: If query execution fails

        """
        t0 = datetime.datetime.now()

        # Create job config based on parameters
        job_config = bigquery.QueryJobConfig(use_legacy_sql=False)

        # Cannot set write_disposition if saving to GCS
        if save_to_gcs:
            # Tell BigQuery to save the results to a specific GCS location
            if not save_dir:
                logger.error("save_dir is not set. Cannot save query results to GCS.")
                return False  # Indicate failure

            import shortuuid
            gcs_results_uri = f"{save_dir}/query_{shortuuid.uuid()}/*.json"
            job_config.destination_uris = [gcs_results_uri]
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

        elif destination:
            # Configure destination table if specified
            job_config.destination = destination
            job_config.write_disposition = (
                bigquery.WriteDisposition.WRITE_TRUNCATE
                if overwrite
                else bigquery.WriteDisposition.WRITE_EMPTY
            )

        # Execute query
        try:
            job = self._bq_client.query(sql, job_config=job_config)
        except Exception as e:
            logger.error(f"BigQuery query failed: {e}")
            return False  # Indicate failure

        # Collect statistics for logging
        bytes_billed = job.total_bytes_billed
        cache_hit = job.cache_hit

        if bytes_billed:
            approx_cost = bytes_billed * GOOGLE_BQ_PRICE_PER_BYTE
            bytes_billed_str = humanfriendly.format_size(bytes_billed)
            approx_cost_str = humanfriendly.format_number(approx_cost)
        else:
            bytes_billed_str = "N/A"
            approx_cost_str = "unknown"

        time_taken = datetime.datetime.now() - t0
        logger.info(
            f"Query stats: Ran in {time_taken} seconds, cache hit: {cache_hit}, "
            f"billed {bytes_billed_str}, approx cost ${approx_cost_str}.",
        )

        if do_not_return_results:
            return True  # Return success status

        # Get query results
        try:
            result = job.result()  # This blocks until query completes
            if return_df:
                if result.total_rows and result.total_rows > 0:
                    return result.to_dataframe()
                return pd.DataFrame()  # Return empty DataFrame if no rows
            return result  # Return raw BigQuery result object
        except Exception as e:
            logger.error(f"Failed to get BigQuery query results: {e}")
            return False  # Indicate failure

    def estimate_query_cost(self, sql: str) -> tuple[int, float]:
        """Estimate the cost of a query without executing it.
        
        Args:
            sql: SQL query to estimate
            
        Returns:
            Tuple of (bytes_billed, estimated_cost_usd)
            
        Raises:
            RuntimeError: If cost estimation fails

        """
        try:
            job_config = bigquery.QueryJobConfig(dry_run=True)
            job = self._bq_client.query(sql, job_config=job_config)

            bytes_processed = job.total_bytes_processed or 0
            estimated_cost = bytes_processed * GOOGLE_BQ_PRICE_PER_BYTE

            bytes_processed_str = humanfriendly.format_size(bytes_processed)
            estimated_cost_str = humanfriendly.format_number(estimated_cost)

            logger.info(f"Query cost estimate: {bytes_processed_str}, approx ${estimated_cost_str}")
            return bytes_processed, estimated_cost
        except Exception as e:
            logger.error(f"Failed to estimate query cost: {e}")
            raise RuntimeError(f"Failed to estimate query cost: {e}") from e
