"""Provides an agent and tool configuration for fetching and processing records.

This module defines `FetchAgent`, a Buttermilk `Agent` that can act
autonomously to fetch records based on incoming messages.
"""

import datetime
from functools import partial
from typing import Any

from autogen_core import (
    message_handler,
)
from autogen_core.tools import FunctionTool, Tool

from buttermilk import bm, logger
from buttermilk._core.agent import Agent
from buttermilk._core.contract import AgentInput, AgentOutput, AgentTrace, StepRequest  # Buttermilk message contracts
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.storage_config import BaseStorageConfig
from buttermilk._core.types import Record
from buttermilk.utils.media import download_and_convert  # Media utilities
from buttermilk.utils.utils import URL_PATTERN

MATCH_PATTERNS = rf"^(![\d\w_]+)|<({URL_PATTERN})>"
"""Regex pattern to match command symbols or URLs.

Used to identify potential record IDs (prefixed with `COMMAND_SYMBOL`) or URIs
within text inputs.
"""


class FetchAgent(Agent):
    """An agent that fetches records, either as a tool or through direct processing.

    Attributes:
        storage (dict[str, BaseStorageConfig]): Datasets that can be used to fetch records.
    """

    def __init__(self, storage: dict[str, BaseStorageConfig] = None, **data):
        super().__init__(**data)
        if storage:
            self._data_sources = {source_name: bm.get_storage(config) for source_name, config in storage.items()}
        else:
            self._data_sources = {}
        self._tools = []

    async def fetch_uri(self, uri: str) -> Record:
        """Fetches a record based on a given URI.

        Args:
            uri (str): The URI of the record to fetch.

        Returns:
            Record: The fetched record.

        Raises:
            ProcessingError: If no record could be found or fetched.
        """
        record = await download_and_convert(uri)
        if record:  # Check if download_and_convert succeeded
            # Ensure metadata exists and add provenance
            if not record.metadata:
                record.metadata = {}
            record.metadata["fetch_source_uri"] = uri
            record.metadata["fetch_timestamp_utc"] = datetime.now(datetime.UTC).isoformat()
            return record
        # Use original_uri for the error message
        raise ProcessingError(f"Record not found for URI: {uri}")

    async def fetch_record(self, record_id: str, dataset_name: str) -> Record:
        """Fetches a record based on `record_id`.

        Args:
            record_id (str): The ID of the record to fetch from loaded data sources.
            dataset (str): The name of the dataset to use to fetch the record.

        Returns:
            Record: The fetched record.

        Raises:
            ProcessingError: If no record could be found or fetched.
        """
        try:
            return self._data_sources[dataset_name].get_record(record_id)
        except Exception as e:
            raise ProcessingError(f"Record not found for ID: {record_id}: {e}") from e

    @message_handler(match=lambda msg, ctx: msg.role == "FETCH")
    async def fetch_request(self, message: StepRequest, ctx) -> AgentOutput | AgentTrace | None:
        if message.role != self.role:
            logger.debug(
                f"Agent {self.agent_name} skipped StepRequest due to role mismatch: requested {message.role}, agent is {self.role}"
            )
            return None

        return await self.invoke(message=message)

    async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput | None:
        """Process the message and return an AgentOutput or ErrorEvent."""
        result = None

        # Check both inputs and parameters for record_id, uri, url
        uri = (
            message.inputs.get("url")
            or message.inputs.get("uri")
            or message.parameters.get("url")
            or message.parameters.get("uri")
        )
        record_id = message.inputs.get("record_id") or message.parameters.get("record_id")

        if uri and record_id:
            raise ProcessingError("Cannot provide both uri and record_id.")

        try:
            if uri:
                result = await self.fetch_uri(uri=uri)
            elif record_id:
                result = await self.fetch_record(record_id=record_id)
        except ProcessingError as e:
            logger.error(f"FetchAgent '{self.agent_id}': {e}")
            raise

        if result and isinstance(result, Record):
            # Wrap the Record in an AgentOutput
            return AgentOutput(
                agent_id=self.agent_id,
                outputs=result,
                metadata=result.metadata if hasattr(result, "metadata") else {},
            )

        # No result found
        raise ProcessingError("No result found in _process")

    def get_tool_definitions(self) -> list[Tool]:
        """Generate structured tool definitions for this agent."""
        internal_tools = [
            FunctionTool(
                name="fetch_uri",
                description=("Get a record from a given URI."),
                func=self.fetch_uri,
                strict=True,
            ),
        ]
        
        # Create dataset-specific fetch_record tools using partial
        dataset_tools = [
            FunctionTool(
                name=f"fetch_record_from_{dataset_name}",
                description=f"Get a record from the {dataset_name} dataset by record ID.",
                func=partial(self.fetch_record, dataset_name=dataset_name),
                strict=True,
            )
            for dataset_name in self._data_sources.keys()
        ]
        
        return internal_tools + dataset_tools
