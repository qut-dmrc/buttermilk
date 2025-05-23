"""Provides an agent and tool configuration for fetching and processing records.

This module defines `FetchRecord`, a `ToolConfig` subclass that can be used by
LLMs to fetch data from specified URIs or internal datasets by ID. It also defines
`FetchAgent`, a Buttermilk `Agent` that can either operate as this tool or act
autonomously to fetch records based on incoming messages.
"""

import asyncio
from collections.abc import (
    Awaitable,
    Callable,  # For typing callables
)
from datetime import UTC, datetime  # For timestamping fetched records
from typing import Any

import pydantic  # Pydantic core
import regex as re  # Regular expression operations
from autogen_core.tools import FunctionTool  # Autogen's FunctionTool for LLM integration
from shortuuid import uuid  # For generating short unique IDs

from buttermilk._core.agent import Agent, AgentOutput, CancellationToken  # Buttermilk base agent and types
from buttermilk._core.config import ToolConfig  # Base class for tool configurations
from buttermilk._core.contract import (  # Buttermilk message contracts
    AgentInput,
    ErrorEvent,
    GroupchatMessageTypes,
    ManagerMessage,
)
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.types import Record
from buttermilk.runner.helpers import prepare_step_df
from buttermilk.utils.media import download_and_convert
from buttermilk.utils.utils import URL_PATTERN, extract_url

MATCH_PATTERNS = rf"^(![\d\w_]+)|<({URL_PATTERN})>"
"""Regex pattern to match command symbols or URLs.

Used to identify potential record IDs (prefixed with `COMMAND_SYMBOL`) or URIs
within text inputs.
"""


class FetchRecord(ToolConfig):
    """A tool configuration for fetching records from data sources or URIs.

    This class, inheriting from `ToolConfig`, defines the structure and logic
    for a tool that can be invoked (e.g., by an LLM agent) to retrieve a `Record`.
    It can fetch from pre-configured data sources using a `record_id` or download
    and convert content from a given `uri`.

    The `data` attribute (from `ToolConfig`) can be configured with
    `DataSourceConfig` instances. `load_data` populates `_data_sources` from these.

    Attributes:
        _data_sources (dict[str, Any]): Private attribute storing loaded data sources,
            typically Pandas DataFrames, keyed by their configured names.
        _data_task (asyncio.Task): Private attribute for an asyncio task, potentially
            for asynchronous data loading (though not explicitly used in `load_data`).
        _pat (Any): Private attribute storing a compiled regex pattern (`MATCH_PATTERNS`)
            used for parsing inputs.
        _fns (list[FunctionTool]): Private attribute caching the list of Autogen
            `FunctionTool` definitions generated for this tool.

    """

    _data_sources: dict[str, Any] = pydantic.PrivateAttr(default_factory=dict)
    _data_task: asyncio.Task[Any] = pydantic.PrivateAttr()  # type: ignore # Needs default or factory
    _pat: Any = pydantic.PrivateAttr(default_factory=lambda: re.compile(MATCH_PATTERNS))
    _fns: list[FunctionTool] = pydantic.PrivateAttr(default_factory=list)

    async def load_data(self) -> None:
        """Loads and prepares data sources defined in `self.data`.
        
        Populates the `self._data_sources` attribute with processed datasets,
        typically Pandas DataFrames, making them available for querying by `record_id`.
        This method is usually called before the tool needs to access internal datasets.
        """
        if self.data:  # self.data is from ToolConfig, a Mapping[str, DataSourceConfig]
            self._data_sources = await prepare_step_df(self.data)

    def get_functions(self) -> list[FunctionTool]:  # Return type changed to list[FunctionTool]
        """Creates and returns Autogen `FunctionTool` definitions for this tool.

        This method makes the `_run` method callable by an LLM agent (e.g.,
        via Autogen's tool use mechanism). It generates a `FunctionTool`
        with the description and role name defined in this `FetchRecord` instance's
        configuration. The generated tools are cached in `self._fns`.

        Returns:
            list[FunctionTool]: A list containing the `FunctionTool` definition(s)
            for this record fetching tool.

        """
        if not self._fns:
            # self.role is from AgentConfig, inherited via Agent -> ToolConfig (if FetchRecord used as Agent)
            # If FetchRecord is used purely as ToolConfig, 'role' might not be standard.
            # Assuming 'name' or a dedicated tool_name field might be more appropriate from ToolConfig.
            # For now, using self.role, assuming it's set appropriately in the config.
            tool_name = getattr(self, "role", "fetch_record_tool")  # Fallback name
            if not self.description:  # Ensure description is set for the tool
                tool_description = "Fetches a record by its ID from a dataset or by its URI."
            else:
                tool_description = self.description

            self._fns = [
                FunctionTool(
                    self.fetch,
                    description=self.description,
                    name="FetchRecord",
                    strict=False,
                ),
            ]
        return self._fns

    async def fetch(self, record_id: str | None = None, uri: str | None = None, prompt: str | None = None) -> Record:
        """Fetches a record based on `record_id`, `uri`, or a `prompt` containing a URI/ID.

        This is the core logic when `FetchRecord` is used as a tool.
        The priority for fetching is:
        1.  Explicit `record_id` (queries loaded `_data_sources`).
        2.  Explicit `uri` (downloads and converts content).
        3.  If `prompt` is provided and `record_id`/`uri` are not:
            a.  Extracts a URL from the `prompt` and uses it as `uri`.
            b.  If no URL, treats the stripped `prompt` (after removing `COMMAND_SYMBOL`)
                as `record_id`.

        An assertion ensures that either `record_id` or `uri` is ultimately determined,
        but not both. Fetched records have metadata updated with fetch source and timestamp.

        Args:
            record_id (str | None): The ID of the record to fetch from loaded data sources.
            uri (str | None): The URI (URL, file path) to fetch and convert.
            prompt (str | None): A text prompt that might contain a URI or a command-like
                record ID (e.g., "!my_record_123").

        Returns:
            Record | ErrorEvent: The fetched (and potentially converted) `Record` object
            if successful, or an `ErrorEvent` if no record could be found or fetched.
        
        Raises:
            AssertionError: If it cannot resolve to either a `record_id` or a `uri`,
                            or if both are somehow provided.

        """
        # Determine original lookup type for error messaging, before uri might be set from prompt
        original_uri = uri
        original_record_id = None  # ensure original_record_id is None if uri is from prompt

        assert (record_id or uri) and not (record_id and uri), "You must provide EITHER record_id OR uri."

        record: Record | None = None
        if record_id:
            # This breaks now because the code was moved to Orchestrator
            record = await self._get_record_dataset(record_id)
            if record:
                # Ensure metadata exists and add provenance
                if not record.metadata:
                    record.metadata = {}
                record.metadata["fetch_source_id"] = record_id
                record.metadata["fetch_timestamp_utc"] = datetime.now(UTC).isoformat()
                return record
            # Use original_record_id for the error message if record_id was from prompt
            raise ProcessingError(f"Record not found for ID: {original_record_id or record_id}")
        if uri:  # uri case
            record = await download_and_convert(uri)
            if record:  # Check if download_and_convert succeeded
                # Ensure metadata exists and add provenance
                if not record.metadata:
                    record.metadata = {}
                record.metadata["fetch_source_uri"] = uri
                record.metadata["fetch_timestamp_utc"] = datetime.now(UTC).isoformat()
                return record
            # Use original_uri for the error message
            raise ProcessingError(f"Record not found for URI: {original_uri or uri}")

        # This part should ideally not be reached due to the assertion and logic above.
        # If it is, it means neither record_id nor uri led to a record or an error for not finding one.
        # However, the logic above ensures that if a record is not found, an error is raised.
        # If record is None here, it means neither record_id nor uri was set, which contradicts the assertion.
        # For safety, though, if we somehow end up here without a record:
        if original_uri:
             raise ProcessingError(f"Record not found for URI: {original_uri}")
        if original_record_id:
             raise ProcessingError(f"Record not found for ID: {original_record_id}")
        # Fallback if prompt didn't yield URI or ID.
        raise ProcessingError("Record not found, and no URI or ID was effectively specified for the fetch attempt.")


class FetchAgent(FetchRecord, Agent):
    """An agent that fetches records, either as a tool or through direct processing.

    This agent combines the record-fetching capabilities of `FetchRecord` (making
    it usable as an LLM tool) with the standard Buttermilk `Agent` lifecycle.
    It can:
    1.  Be configured as a tool for other agents (via `FetchRecord.get_functions`).
    2.  Passively listen for messages (`_listen`) containing URIs or record IDs
        in their `AgentInput.inputs` and, if found, fetch the record and publish
        it as an `AgentOutput`.
    3.  Actively process an `AgentInput` (`_process`) to fetch a record based on
        `uri` or `record_id` in the input, returning the fetched `Record` or an
        `ErrorEvent`.

    Attributes:
        id (str): A unique identifier for the agent instance, typically prefixed
            with "fetch_record_". Defaults to a generated ID.

    """

    id: str = pydantic.Field(
        default_factory=lambda: f"fetch_record_{uuid()[:4]}",
        description="Unique identifier for the FetchAgent instance.",
    )

    async def _listen(
        self,
        message: AgentInput | GroupchatMessageTypes,  # More specific input type
        *,
        cancellation_token: CancellationToken | None = None,  # Standard arg
        source: str = "",  # Standard arg
        public_callback: Callable[[Any], Awaitable[None]] | None = None,  # Made optional
        message_callback: Callable[[Any], Awaitable[None]] | None = None,  # Made optional
        **kwargs: Any,  # Standard arg
    ) -> None:
        """Listens for `AgentInput` messages and attempts to fetch a record if URI/ID is provided.

        If running as an agent, watch for URLs or record ids from the
        user and inject them into the chat.

        If the incoming `message` is an `AgentInput` and contains either a `uri`
        or `record_id` in its `inputs` dictionary, this method calls the `_run`
        (inherited from `FetchRecord`) to fetch the record. If successful and a
        `public_callback` is provided, it wraps the fetched `Record` in an
        `AgentOutput` and publishes it.

        Args:
            message: The incoming message. Expected to be an `AgentInput` for
                active fetching logic.
            cancellation_token: Optional cancellation token.
            source: Identifier of the message sender.
            public_callback: Optional callback function to publish results (e.g.,
                the fetched `Record` wrapped in `AgentOutput`).
            message_callback: Optional callback (typically not used by `_listen`).
            **kwargs: Additional keyword arguments.

        """
        result = None
        if isinstance(message, ManagerMessage):
            if message.content:
                # Check if the message is a command
                match = self._pat.search(message.content)
                if match:
                    # Extract the record_id or URL from the message
                    record_id = match.group(1)
                    uri = match.group(2)
                    if uri:
                        result = await self.fetch(uri=uri)
                    elif record_id:
                        result = await self.fetch(record_id=record_id)
                elif uri := extract_url(message.content):
                    result = await self.fetch(uri=uri)

        if result and isinstance(result, Record):
            # output = AgentOutput(agent_id=self.agent_id,
            #     outputs=result,
            #     metadata=result.metadata,
            # )
            await public_callback(result)

    async def _process(self, *, message: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs) -> AgentOutput | ErrorEvent:
        """Process the message and return an AgentOutput or ErrorEvent."""
        result = None
        if isinstance(message, AgentInput):
            uri = message.inputs.get("uri")
            record_id = message.inputs.get("record_id")
            if uri or record_id:
                result = await self.fetch(record_id=record_id, uri=uri, prompt=message.inputs.get("prompt"))

        if result:
            return result

        # Return an ErrorEvent instead of None
        return ErrorEvent(source=self.id, content="No result found in _process")
