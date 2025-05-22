"""Provides an agent and tool configuration for fetching and processing records.

This module defines `FetchRecord`, a `ToolConfig` subclass that can be used by
LLMs to fetch data from specified URIs or internal datasets by ID. It also defines
`FetchAgent`, a Buttermilk `Agent` that can either operate as this tool or act
autonomously to fetch records based on incoming messages.
"""

import asyncio
from collections.abc import Callable # For typing callables
from datetime import UTC, datetime  # For timestamping fetched records
from typing import Any

import pydantic # Pydantic core
import regex as re # Regular expression operations
from autogen_core.tools import FunctionTool # Autogen's FunctionTool for LLM integration
from shortuuid import uuid # For generating short unique IDs

from buttermilk._core.agent import Agent, AgentOutput, CancellationToken # Buttermilk base agent and types
from buttermilk._core.config import ToolConfig # Base class for tool configurations
from buttermilk._core.constants import COMMAND_SYMBOL # Symbol for commands
from buttermilk._core.contract import ( # Buttermilk message contracts
    AgentInput,
    AgentTrace, # Though not directly returned by _process, it's a possible wrapper output
    ErrorEvent,
    GroupchatMessageTypes,
    ManagerMessage, # Possible input/output types for an Agent
    StepRequest,
    ToolOutput, # Possible input/output types for an Agent
    UIMessage,
)
from buttermilk._core.types import Record # Core Buttermilk Record type
from buttermilk.runner.helpers import prepare_step_df # Helper for data source preparation
from buttermilk.utils.media import download_and_convert # Utility for downloading and converting media
from buttermilk.utils.utils import URL_PATTERN, extract_url # Utilities for URL handling

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
    _data_task: asyncio.Task[Any] = pydantic.PrivateAttr() # type: ignore # Needs default or factory
    _pat: Any = pydantic.PrivateAttr(default_factory=lambda: re.compile(MATCH_PATTERNS))
    _fns: list[FunctionTool] = pydantic.PrivateAttr(default_factory=list)

    async def load_data(self) -> None:
        """Loads and prepares data sources defined in `self.data`.
        
        Populates the `self._data_sources` attribute with processed datasets,
        typically Pandas DataFrames, making them available for querying by `record_id`.
        This method is usually called before the tool needs to access internal datasets.
        """
        if self.data: # self.data is from ToolConfig, a Mapping[str, DataSourceConfig]
            self._data_sources = await prepare_step_df(self.data)

    def get_functions(self) -> list[FunctionTool]: # Return type changed to list[FunctionTool]
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
            tool_name = getattr(self, 'role', 'fetch_record_tool') # Fallback name
            if not self.description: # Ensure description is set for the tool
                tool_description = "Fetches a record by its ID from a dataset or by its URI."
            else:
                tool_description = self.description

            self._fns = [
                FunctionTool(
                    self._run, # The method to be called
                    description=tool_description,
                    name=tool_name, # Name for the LLM to call this tool
                    strict=False, # Pydantic validation mode for arguments
                ),
            ]
        return self._fns

    async def _run(
        self, 
        record_id: str | None = None, 
        uri: str | None = None, 
        prompt: str | None = None
    ) -> Record | ErrorEvent:
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
        resolved_record_id = record_id
        resolved_uri = uri

        if prompt and not resolved_record_id and not resolved_uri:
            extracted_uri = extract_url(prompt)
            if extracted_uri:
                resolved_uri = extracted_uri
            else:
                # Try to interpret prompt as a record_id (remove command symbol if present)
                resolved_record_id = prompt.strip().lstrip(COMMAND_SYMBOL)
        
        # Ensure exactly one of record_id or uri is set for fetching.
        if not (resolved_record_id or resolved_uri) or (resolved_record_id and resolved_uri):
            err_msg = "Must provide EITHER record_id OR uri (or a prompt containing one)."
            if resolved_record_id and resolved_uri: # If both somehow got set
                err_msg = f"Both record_id ('{resolved_record_id}') and uri ('{resolved_uri}') were specified or derived; please provide only one."
            return ErrorEvent(source=getattr(self, 'id', "fetch_tool"), content=err_msg)


        fetched_record: Record | None = None
        if resolved_record_id:
            try:
                # _get_record_dataset needs to be available if FetchRecord is used standalone.
                # If FetchRecord is mixed into an Agent, this method would be from Agent.
                # This implies FetchRecord might need its own _get_record_dataset or access to one.
                # For now, assuming it's part of a class that has this method (like FetchAgent).
                # This is a potential design issue if FetchRecord is meant to be fully standalone as a ToolConfig.
                # Let's assume this method exists on `self` (e.g., inherited from FetchAgent or similar context)
                if hasattr(self, "_get_record_dataset") and callable(self._get_record_dataset):
                    fetched_record = await self._get_record_dataset(resolved_record_id) # type: ignore
                else: # Fallback or error if method is missing
                    return ErrorEvent(source=getattr(self, 'id', "fetch_tool"), content=f"_get_record_dataset method not available to fetch record_id '{resolved_record_id}'.")

                if fetched_record:
                    if not fetched_record.metadata: fetched_record.metadata = {}
                    fetched_record.metadata["fetch_source_id"] = resolved_record_id
                    fetched_record.metadata["fetch_timestamp_utc"] = datetime.now(UTC).isoformat()
            except Exception as e:
                return ErrorEvent(source=getattr(self, 'id', "fetch_tool"), content=f"Error fetching record_id '{resolved_record_id}': {e!s}")
        
        elif resolved_uri: # uri case (resolved_uri is guaranteed to be non-None here)
            try:
                fetched_record = await download_and_convert(resolved_uri)
                if fetched_record:
                    if not fetched_record.metadata: fetched_record.metadata = {}
                    fetched_record.metadata["fetch_source_uri"] = resolved_uri
                    fetched_record.metadata["fetch_timestamp_utc"] = datetime.now(UTC).isoformat()
            except Exception as e:
                 return ErrorEvent(source=getattr(self, 'id', "fetch_tool"), content=f"Error downloading/converting URI '{resolved_uri}': {e!s}")

        if fetched_record:
            return fetched_record
        
        return ErrorEvent(source=getattr(self, 'id', "fetch_tool"), content=f"No record found for specified input (id: {resolved_record_id}, uri: {resolved_uri}).")


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
        description="Unique identifier for the FetchAgent instance."
    )

    async def _listen(
        self,
        message: AgentInput | GroupchatMessageTypes, # More specific input type
        *,
        cancellation_token: CancellationToken | None = None, # Standard arg
        source: str = "", # Standard arg
        public_callback: Callable[[Any], Awaitable[None]] | None = None, # Made optional
        message_callback: Callable[[Any], Awaitable[None]] | None = None, # Made optional
        **kwargs: Any, # Standard arg
    ) -> None:
        """Listens for `AgentInput` messages and attempts to fetch a record if URI/ID is provided.

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
        fetched_result: Record | ErrorEvent | None = None
        if isinstance(message, AgentInput):
            uri = message.inputs.get("uri")
            record_id = message.inputs.get("record_id")
            prompt_for_extraction = message.inputs.get("prompt") if not (uri or record_id) else None

            if uri or record_id or prompt_for_extraction:
                fetched_result = await self._run(record_id=record_id, uri=uri, prompt=prompt_for_extraction)
            else:
                logger.debug(f"FetchAgent '{self.id}' listening: No URI or record_id in AgentInput from '{source}'.")
        # else: Not an AgentInput, so _listen does nothing further for now.

        if fetched_result and isinstance(fetched_result, Record):
            if public_callback:
                output_msg = AgentOutput(
                    agent_id=self.id, # Use agent's own ID
                    outputs=fetched_result,
                    metadata=fetched_result.metadata.copy(), # Pass a copy of metadata
                )
                await public_callback(output_msg)
            else:
                logger.warning(f"FetchAgent '{self.id}' fetched record '{fetched_result.record_id}' but no public_callback provided to publish.")
        elif isinstance(fetched_result, ErrorEvent):
            logger.warning(f"FetchAgent '{self.id}' failed to fetch record from message by '{source}': {fetched_result.content}")
            # Optionally, could publish the ErrorEvent via public_callback if desired

    async def _process(
        self, 
        *, 
        message: AgentInput, 
        cancellation_token: CancellationToken | None = None, # Standard arg
        **kwargs: Any # Standard arg
    ) -> AgentOutput | ErrorEvent: # Return type matches base Agent._process more closely
        """Actively processes an `AgentInput` to fetch a record.

        This method is called when the `FetchAgent` is directly invoked. It uses
        the `uri` or `record_id` (or extracts them from `prompt`) found in
        `message.inputs` to fetch a record via the `_run` method (inherited
        from `FetchRecord`).

        Args:
            message: The `AgentInput` containing details for fetching, primarily
                `message.inputs.uri`, `message.inputs.record_id`, or
                `message.inputs.prompt`.
            cancellation_token: Optional cancellation token.
            **kwargs: Additional keyword arguments.

        Returns:
            AgentOutput | ErrorEvent: If fetching is successful, returns an
            `AgentOutput` wrapping the fetched `Record`. If fetching fails or
            no valid input is found, returns an `ErrorEvent`.
        """
        # The original return type included many possibilities, but for an agent's _process,
        # it should typically conform to returning something that becomes AgentOutput/AgentTrace.
        # Directly returning a Record or ErrorEvent is fine if the base Agent.__call__ handles wrapping.
        # For clarity, this will construct AgentOutput or ErrorEvent.
        
        fetched_result: Record | ErrorEvent | None = None
        # Ensure message.inputs exists, though AgentInput defaults it to dict
        inputs_data = message.inputs or {} 

        uri = inputs_data.get("uri")
        record_id = inputs_data.get("record_id")
        # Use prompt from inputs if uri/record_id not directly provided
        prompt_for_extraction = inputs_data.get("prompt") if not (uri or record_id) else None

        if uri or record_id or prompt_for_extraction:
            fetched_result = await self._run(record_id=record_id, uri=uri, prompt=prompt_for_extraction)
        else:
            # No valid parameters to fetch
            return ErrorEvent(source=self.id, content="FetchAgent._process: No 'uri', 'record_id', or 'prompt' with fetchable content found in inputs.")

        if isinstance(fetched_result, Record):
            # The base Agent's __call__ method will wrap this in AgentTrace.
            # We need to return an AgentOutput structure.
            return AgentOutput(
                agent_id=self.id,
                outputs=fetched_result,
                metadata=fetched_result.metadata.copy() # Pass a copy
            )
        elif isinstance(fetched_result, ErrorEvent):
            # Propagate the ErrorEvent
            return fetched_result
        
        # Should not be reached if _run always returns Record or ErrorEvent
        return ErrorEvent(source=self.id, content="FetchAgent._process: Unknown result from _run method.")
