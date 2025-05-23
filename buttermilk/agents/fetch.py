import asyncio
from collections.abc import Callable
from datetime import UTC, datetime  # Added for timestamp
from typing import Any

import pydantic
import regex as re
from autogen_core.tools import FunctionTool
from shortuuid import uuid

from buttermilk._core.agent import Agent, AgentOutput, CancellationToken
from buttermilk._core.config import ToolConfig
from buttermilk._core.constants import COMMAND_SYMBOL
from buttermilk._core.contract import (
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


class FetchRecord(ToolConfig):
    _data_sources: dict[str, Any] = pydantic.PrivateAttr(default={})
    _data_task: asyncio.Task = pydantic.PrivateAttr()
    _pat: Any = pydantic.PrivateAttr(default_factory=lambda: re.compile(MATCH_PATTERNS))
    _fns: list[FunctionTool] = pydantic.PrivateAttr(default=[])

    async def load_data(self):
        self._data_sources = await prepare_step_df(self.data)

    def get_functions(self) -> list[Any]:
        """Create function definitions for this tool."""
        if not self._fns:
            self._fns = [
                FunctionTool(
                    self.fetch,
                    description=self.description,
                    name="FetchRecord",
                    strict=False,
                ),
            ]
        return self._fns

    async def fetch(self, record_id: str | None = None, uri: str | None = None, prompt: str | None = None) -> Record:  # type: ignore
        """Entry point when running as a tool."""
        # Determine original lookup type for error messaging, before uri might be set from prompt
        original_uri = uri
        original_record_id = record_id

        if prompt and not record_id and not uri:
            if not (uri := extract_url(prompt)):
                # Try to get by record_id (remove bang! first)
                record_id = prompt.strip().strip(COMMAND_SYMBOL)
            else: # uri was extracted from prompt
                original_uri = uri # update original_uri to reflect that it came from prompt
                original_record_id = None # ensure original_record_id is None if uri is from prompt

        assert (record_id or uri) and not (record_id and uri), "You must provide EITHER record_id OR uri."

        record: Record | None = None
        if record_id:
            record = await self._get_record_dataset(record_id)
            if record:
                # Ensure metadata exists and add provenance
                if not record.metadata:
                    record.metadata = {}
                record.metadata["fetch_source_id"] = record_id
                record.metadata["fetch_timestamp_utc"] = datetime.now(UTC).isoformat()
                return record
            else:
                # Use original_record_id for the error message if record_id was from prompt
                raise ProcessingError(f"Record not found for ID: {original_record_id or record_id}")
        elif uri:  # uri case
            record = await download_and_convert(uri)
            if record:  # Check if download_and_convert succeeded
                # Ensure metadata exists and add provenance
                if not record.metadata:
                    record.metadata = {}
                record.metadata["fetch_source_uri"] = uri
                record.metadata["fetch_timestamp_utc"] = datetime.now(UTC).isoformat()
                return record
            else:
                # Use original_uri for the error message
                raise ProcessingError(f"Record not found for URI: {original_uri or uri}")
        
        # This part should ideally not be reached due to the assertion and logic above.
        # If it is, it means neither record_id nor uri led to a record or an error for not finding one.
        # However, the logic above ensures that if a record is not found, an error is raised.
        # If record is None here, it means neither record_id nor uri was set, which contradicts the assertion.
        # For safety, though, if we somehow end up here without a record:
        if original_uri:
             raise ProcessingError(f"Record not found for URI: {original_uri}")
        elif original_record_id:
             raise ProcessingError(f"Record not found for ID: {original_record_id}")
        else:
            # Fallback if prompt didn't yield URI or ID.
            raise ProcessingError("Record not found, and no URI or ID was effectively specified for the fetch attempt.")


class FetchAgent(FetchRecord, Agent):
    id: str = pydantic.Field(default_factory=lambda: f"fetch_record_{uuid()[:4]}")

    async def _listen(
        self,
        message: AgentInput | GroupchatMessageTypes,
        *,
        cancellation_token: CancellationToken | None = None,
        source: str = "",
        public_callback: Callable = None,
        message_callback: Callable = None,
        **kwargs,
    ) -> None:
        """Entry point when running this as an agent.

        If running as an agent, watch for URLs or record ids from the
        user and inject them into the chat.
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
