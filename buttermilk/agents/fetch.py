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
    AgentTrace,
    ErrorEvent,
    GroupchatMessageTypes,
    ManagerMessage,
    ManagerRequest,
    StepRequest,
    ToolOutput,
)
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
                    self._run,
                    description=self.description,
                    name=self.role,
                    strict=False,
                ),
            ]
        return self._fns

    async def _run(self, record_id: str | None = None, uri: str | None = None, prompt: str | None = None) -> Record | ErrorEvent:  # type: ignore
        """Entry point when running as a tool."""
        record = None
        if prompt and not record_id and not uri:
            if not (uri := extract_url(prompt)):
                # Try to get by record_id (remove bang! first)
                record_id = prompt.strip().strip(COMMAND_SYMBOL)
        assert (record_id or uri) and not (record_id and uri), "You must provide EITHER record_id OR uri."
        result = None
        if record_id:
            record = await self._get_record_dataset(record_id)
            if record:
                # Ensure metadata exists and add provenance
                if not record.metadata:
                    record.metadata = {}
                record.metadata["fetch_source_id"] = record_id
                record.metadata["fetch_timestamp_utc"] = datetime.now(UTC).isoformat()
                return record
        else:  # uri case
            record = await download_and_convert(uri)
            if record:  # Check if download_and_convert succeeded
                # Ensure metadata exists and add provenance
                if not record.metadata:
                    record.metadata = {}
                record.metadata["fetch_source_uri"] = uri
                record.metadata["fetch_timestamp_utc"] = datetime.now(UTC).isoformat()
                return record

        # Return an ErrorEvent
        return ErrorEvent(source="fetch tool", content="No result found")


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

        If running as an agent, watch for URLs or record ids and inject them
        into the chat.
        """
        result = None
        if isinstance(message, AgentInput):
            uri = message.inputs.get("uri")
            record_id = message.inputs.get("record_id")
            if uri or record_id:
                result = await self._run(record_id=record_id, uri=uri, prompt=message.inputs.get("prompt"))

        if result and isinstance(result, Record):
            output = AgentOutput(agent_id=self.agent_id,
                outputs=result,
                metadata=result.metadata,
            )
            await public_callback(output)

    async def _process(self, *, message: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs) -> AgentTrace | StepRequest | ManagerRequest | ManagerMessage | ToolOutput | ErrorEvent:
        result = None
        if isinstance(message, AgentInput):
            uri = message.inputs.get("uri")
            record_id = message.inputs.get("record_id")
            if uri or record_id:
                result = await self._run(record_id=record_id, uri=uri, prompt=message.inputs.get("prompt"))

        if result:
            return result

        # Return an ErrorEvent instead of None
        return ErrorEvent(source=self.id, content="No result found in _process")
