"""Provides utilities for executing Google BigQuery queries with cost tracking.

This module includes the `QueryRunner` class, which facilitates running SQL queries
against BigQuery, estimating their cost, and handling results, including options
for saving to BigQuery tables or Google Cloud Storage (GCS), and returning
results as Pandas DataFrames.
"""

import datetime

import humanfriendly  # For human-readable data sizes and numbers
import pandas as pd
from google.cloud import bigquery  # Google Cloud BigQuery client library
from pydantic import BaseModel, ConfigDict, Field  # Pydantic for model validation

from buttermilk._core.log import logger  # Centralized logger

# Pricing reference: https://cloud.google.com/bigquery/pricing
GOOGLE_BQ_PRICE_PER_BYTE = 5 / (10**12)  # $5 per Terabyte (1 TB = 10^12 bytes)
"""Estimated cost per byte for Google BigQuery queries, based on $5 per TB."""


class QueryRunner(BaseModel):
    """Manages Google BigQuery query execution and provides cost tracking.

    This class wraps the Google BigQuery client to offer methods for running
    SQL queries, estimating their costs, and handling results in various formats.
    It logs query statistics, including execution time, data processed, and
    approximate cost.

    Attributes:
        bq_client (bigquery.Client): An instance of the authenticated Google
            BigQuery client.
        model_config (ConfigDict): Pydantic model configuration.
            - `arbitrary_types_allowed`: True - Allows `bigquery.Client` type.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bq_client: bigquery.Client = Field(..., description="Authenticated BigQuery client instance.")

    def run_query(
        self,
        sql: str,
        destination: str | None = None,
        overwrite: bool = False,
        do_not_return_results: bool = False,
        save_to_gcs: bool = False,
        return_df: bool = True,
        save_dir: str | None = None,
    ) -> pd.DataFrame | bigquery.table.RowIterator | bool | None:
        """Runs a BigQuery SQL query with options for destination, result handling, and cost tracking.

        Args:
            sql (str): The SQL query string to execute.
            destination (str | None): Optional. The destination BigQuery table ID
                (format: "project.dataset.table") where results should be saved.
                If None, results are typically returned in memory or to a temporary table.
            overwrite (bool): If True and `destination` is specified, the destination
                table will be overwritten if it already exists. Defaults to False.
            do_not_return_results (bool): If True, the query is executed, but results
                are not fetched back to the client (e.g., for DDL/DML statements or
                when results are large and saved to a destination table).
                Defaults to False.
            save_to_gcs (bool): If True, results are saved to Google Cloud Storage (GCS)
                instead of a BigQuery table. `save_dir` must specify the GCS path prefix.
                Defaults to False.
            return_df (bool): If True (and `do_not_return_results` is False), attempts
                to return the query results as a Pandas DataFrame. Defaults to True.
            save_dir (str | None): The directory path (local or GCS URI) where results
                should be saved if `save_to_gcs` is True. A unique filename will be
                generated within this directory.

        Returns:
            pd.DataFrame | bigquery.table.RowIterator | bool | None:
            - If `return_df` is True and results are fetched: A Pandas DataFrame.
              Returns an empty DataFrame if the query yields no rows.
            - If `return_df` is False and results are fetched: A `bigquery.table.RowIterator`.
            - If `do_not_return_results` is True: `True` for successful execution,
              `False` for failure.
            - `None` or `False` can also indicate failure in other scenarios.

        Raises:
            RuntimeError: Can be implicitly raised by the BigQuery client for
                various API errors, though this method attempts to catch common
                exceptions and log them, returning `False` or `None`.

        """
        t_start = datetime.datetime.now(datetime.UTC)  # Use UTC for consistency

        job_config = bigquery.QueryJobConfig(use_legacy_sql=False)

        if save_to_gcs:
            if not save_dir:
                logger.error("`save_dir` must be provided when `save_to_gcs` is True.")
                return None  # Indicate failure due to missing configuration

            import shortuuid  # For unique filenames
            # Ensure save_dir ends with a slash for proper GCS path construction
            gcs_path_prefix = save_dir if save_dir.endswith("/") else save_dir + "/"
            gcs_results_uri = f"{gcs_path_prefix}query_results_{shortuuid.uuid()}/results-*.json"
            job_config.destination_uris = [gcs_results_uri]
            # For GCS, typically WRITE_TRUNCATE is used to ensure clean output directory
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
            logger.info(f"Query results will be saved to GCS: {gcs_results_uri}")

        elif destination:
            job_config.destination = destination
            job_config.write_disposition = (
                bigquery.WriteDisposition.WRITE_TRUNCATE if overwrite else bigquery.WriteDisposition.WRITE_APPEND  # Or WRITE_EMPTY if preferred
            )
            logger.info(f"Query results will be saved to BigQuery table: {destination} (Overwrite: {overwrite})")

        try:
            query_job = self.bq_client.query(sql, job_config=job_config)
            logger.info(f"BigQuery job submitted: {query_job.job_id}")

            # Wait for the job to complete to get statistics
            query_job.result()  # This blocks until the query completes

        except Exception as e:
            logger.error(f"BigQuery query submission or execution failed: {e!s}")
            return None  # Indicate failure

        # Collect and log statistics
        bytes_billed = query_job.total_bytes_billed or 0  # Ensure not None
        cache_hit = query_job.cache_hit
        approx_cost = bytes_billed * GOOGLE_BQ_PRICE_PER_BYTE

        bytes_billed_str = humanfriendly.format_size(bytes_billed)
        approx_cost_str = f"${approx_cost:.2f}"  # Format cost to 2 decimal places

        time_taken = datetime.datetime.now(datetime.UTC) - t_start
        logger.info(
            f"Query '{query_job.job_id}' stats: Execution time: {time_taken}, Cache hit: {cache_hit}, "
            f"Bytes billed: {bytes_billed_str}, Approx. cost: {approx_cost_str}.",
        )

        if do_not_return_results or save_to_gcs:  # If results saved to GCS, also don't return them directly
            return True  # Indicate successful execution without returning data

        try:
            if return_df:
                if query_job.total_rows and query_job.total_rows > 0:
                    return query_job.to_dataframe()
                return pd.DataFrame()  # Return empty DataFrame for no rows
            return query_job  # Return the job object which contains RowIterator
        except Exception as e:
            logger.error(f"Failed to retrieve BigQuery query results for job '{query_job.job_id}': {e!s}")
            return None  # Indicate failure

    def estimate_query_cost(self, sql: str) -> tuple[int | None, float | None]:
        """Estimates the cost of a BigQuery SQL query without actually executing it.

        This is done by performing a "dry run" of the query.

        Args:
            sql (str): The SQL query string to estimate.

        Returns:
            tuple[int | None, float | None]: A tuple containing:
                - `bytes_billed` (int | None): The estimated number of bytes the query
                  would process. None if estimation fails.
                - `estimated_cost_usd` (float | None): The estimated cost in USD.
                  None if estimation fails.

        Raises:
            RuntimeError: If the BigQuery client fails to perform the dry run
                or if cost estimation encounters an unexpected error. This behavior
                is preserved from the original code, though returning (None, None)
                might be an alternative for handling failures.

        """
        try:
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            # use_query_cache=False ensures it estimates actual processing cost

            query_job = self.bq_client.query(sql, job_config=job_config)  # Perform dry run

            bytes_processed = query_job.total_bytes_processed  # This is the key metric from dry run
            if bytes_processed is None:  # Should not happen for successful dry run, but defensive
                logger.warning("Dry run for query cost estimation did not return total_bytes_processed.")
                return None, None

            estimated_cost_usd = bytes_processed * GOOGLE_BQ_PRICE_PER_BYTE

            # Log human-readable estimates
            bytes_processed_str = humanfriendly.format_size(bytes_processed)
            estimated_cost_str = f"${estimated_cost_usd:.2f}"  # Format cost

            logger.info(
                f"Query cost estimation: Bytes to be processed: {bytes_processed_str}, Approximate cost: {estimated_cost_str}",
            )
            return bytes_processed, estimated_cost_usd
        except Exception as e:
            logger.error(f"Failed to estimate query cost for SQL: '{sql[:100]}...': {e!s}")
            # Original code raised RuntimeError, preserving that behavior.
            # Alternatively, could return (None, None) to indicate failure.
            raise RuntimeError(f"Failed to estimate query cost: {e!s}") from e
