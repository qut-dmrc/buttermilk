"""Provides an agent and tool configuration for fetching and processing records.

This module defines `FetchAgent`, a Buttermilk `Agent` that can act
autonomously to fetch records based on incoming messages.
"""

import datetime
from typing import Any

from buttermilk import bm, logger
from buttermilk._core.agent import Agent
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
    ExecutionTrace,
    StepRequest,
)  # Buttermilk message contracts
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.runtime_types import message_handler
from buttermilk._core.tool_types import FunctionTool, Tool
from buttermilk._core.types import BaseRecord, Record
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

    def __init__(self, **data):
        super().__init__(**data)
        if storage := data.get("parameters", {}).get("storage"):
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
            record = self._data_sources[dataset_name].get_record_by_id(record_id)
            if record is None:
                raise ProcessingError(f"Record not found for ID: {record_id}")
            return record
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Record not found for ID: {record_id}: {e}") from e

    @message_handler(match=lambda msg, ctx: msg.role == "FETCH")
    async def fetch_request(self, message: StepRequest, ctx) -> AgentOutput | ExecutionTrace | None:
        return await self.invoke(message=message)

    async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput | None:
        """Process the message and return an AgentOutput or ErrorEvent."""
        result = None

        # DEBUG: Log what we receive for RFC #311 debugging
        logger.debug(f"FETCH {self.agent_id} received inputs keys: {list(message.inputs.keys()) if message.inputs else []}")
        logger.debug(f"FETCH {self.agent_id} received parameters keys: {list(message.parameters.keys()) if message.parameters else []}")
        logger.debug(f"FETCH {self.agent_id} required_inputs config: {self.required_inputs}")

        # Check both inputs and parameters for record_id, uri, url
        uri = message.inputs.get("url") or message.inputs.get("uri") or message.parameters.get("url") or message.parameters.get("uri")

        # Check for record_id in inputs/parameters, or extract from message.record field
        record_id = (
            message.inputs.get("record_id") or message.parameters.get("record_id") or message.inputs.get("record") or message.parameters.get("record")
        )

        # If no record_id found but message.record exists, extract record_id from it
        if not record_id and message.record:
            record_id = getattr(message.record, "record_id", None)

        # DEBUG: Log what we found
        logger.debug(f"FETCH {self.agent_id} found uri={uri!r}, record_id={record_id!r}")

        if uri and record_id:
            raise ProcessingError("Cannot provide both uri and record_id.")

        # If message.record already has content, use it directly (no need to re-fetch)
        # This handles the case where records are pre-loaded via ParameterExpansionProcessor
        if message.record and hasattr(message.record, "content") and message.record.content:
            logger.debug(f"FETCH {self.agent_id}: Using pre-loaded record, skipping fetch")
            return AgentOutput(
                agent_id=self.agent_id,
                outputs=message.record,
                metadata=message.record.metadata if hasattr(message.record, "metadata") else {},
            )

        try:
            if uri:
                result = await self.fetch_uri(uri=uri)
            elif record_id:
                if not (dataset_name := message.inputs.get("dataset")):
                    dataset_name = list(self._data_sources)[0]  # use first dataset as default
                result = await self.fetch_record(record_id=record_id, dataset_name=dataset_name)
        except ProcessingError as e:
            logger.error(f"FetchAgent '{self.agent_id}': {e}")
            raise

        if result and isinstance(result, BaseRecord):
            # Wrap the Record in an AgentOutput
            return AgentOutput(
                agent_id=self.agent_id,
                outputs=result,
                metadata=result.metadata if hasattr(result, "metadata") else {},
            )

        # No result found
        raise ProcessingError("No result found in _process")

    def get_tool_definitions(self) -> list[Tool]:
        """Generate structured tool definitions for this agent.

        Returns list of tool definitions as Tool objects."""
        internal_tools = [
            FunctionTool(
                name="fetch_uri",
                description=("Get a record from a given URI."),
                func=self.fetch_uri,
                strict=True,
            )
        ]

        datasets = list(self._data_sources.keys())
        if datasets:
            internal_tools.append(
                FunctionTool(
                    name="fetch_record",
                    description=f"Get a record from a dataset (literal: {', '.join(datasets)}) by record ID.",
                    func=self.fetch_record,
                    strict=True,
                )
            )

        return internal_tools
