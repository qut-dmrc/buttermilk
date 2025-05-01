import asyncio
from collections.abc import Callable
from datetime import UTC, datetime  # Added for timestamp
from typing import Any

import pydantic
import regex as re
from autogen_core.tools import FunctionTool
from shortuuid import uuid

from buttermilk._core.agent import Agent, CancellationToken
from buttermilk._core.config import ToolConfig
from buttermilk._core.contract import (
    COMMAND_SYMBOL,
    AgentInput,
    AgentOutput,
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

    async def _get_record_dataset(self, record_id: str) -> Record | None:
        if not self._data_sources:
            await self.load_data()

        for dataset in self._data_sources.values():
            rec = dataset.query("record_id==@record_id")
            if rec.shape[0] == 1:
                data = rec.iloc[0].to_dict()
                if "components" in data:
                    content = "\n".join([d["content"] for d in data["components"]])
                    return Record(
                        content=content, metadata=data.get("metadata"), ground_truth=data.get("ground_truth"), uri=data.get("metadata").get("url"),
                    )
                return Record(**data)
            if rec.shape[0] > 1:
                raise ValueError(
                    f"More than one record found for query record_id == {record_id}",
                )

        return None


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
                result = await self._run(record_id=record_id, uri=uri, prompt=message.prompt)

        if result and isinstance(result, Record):
            await public_callback(result)

    async def _process(self, *, message: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs) -> AgentOutput | StepRequest | ManagerRequest | ManagerMessage | ToolOutput | ErrorEvent:
        result = None
        if isinstance(message, AgentInput):
            uri = message.inputs.get("uri")
            record_id = message.inputs.get("record_id")
            if uri or record_id:
                result = await self._run(record_id=record_id, uri=uri, prompt=message.prompt)

        if result:
            return result

        # Return an ErrorEvent instead of None
        return ErrorEvent(source=self.id, content="No result found in _process")
