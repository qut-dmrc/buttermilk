import asyncio
from collections.abc import AsyncGenerator
from typing import Any, Self

import pydantic
import regex as re
from shortuuid import uuid

from buttermilk._core.agent import Agent, CancellationToken, ToolConfig
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
    GroupchatMessageTypes,
    UserInstructions,
)
from buttermilk._core.runner_types import Record
from buttermilk.runner.helpers import prepare_step_df
from buttermilk.utils.media import download_and_convert
from buttermilk.utils.utils import URL_PATTERN, extract_url

MATCH_PATTERNS = rf"^(![\d\w_]+)|<({URL_PATTERN})>"


class FetchRecord(Agent, ToolConfig):
    id: str = pydantic.Field(default_factory=lambda: f"fetch_record_{uuid()[:4]}" )
    role: str = pydantic.Field(default="") 
    _data: dict[str, Any] = pydantic.PrivateAttr(default={})
    _data_task: asyncio.Task = pydantic.PrivateAttr()
    _pat: Any = pydantic.PrivateAttr(default_factory=lambda: re.compile(MATCH_PATTERNS))

    @pydantic.model_validator(mode="after")
    def load_data_task(self) -> Self:
        self._data_task = asyncio.create_task(self.load_data())
        return self

    async def load_data(self):
        self._data = await prepare_step_df(self.data)

    async def _process(
        self,
        message: GroupchatMessageTypes,
        cancellation_token: CancellationToken | None = None,
        **kwargs,
    ) -> AsyncGenerator[AgentOutput | None, None]:
        """Entry point when running this as an agent."""
        if not isinstance(message, AgentInput):
            return
        record = await self._run(**message.inputs)
        if record:
            yield AgentOutput(
                source=self.id,
                role=self.role,
                content=record.fulltext,
                records=[record],
            )
        return

    async def _run(
        self,
        record_id: str | None = None,
        uri: str | None = None,
        prompt: str | None = None
    ) -> AgentOutput | None:
        """Entry point when running as a tool."""
        record  = None
        if prompt and not record_id and not uri:
            if not (uri := extract_url(prompt)):
                record_id = prompt.strip().strip("!")
        assert (record_id or uri) and not (record_id and uri), (
            "You must provide EITHER record_id OR uri."
        )
        if record_id:
            record = await self.get_record_dataset(record_id)
        else:
            record =await download_and_convert(uri)
        
        if record:
            return AgentOutput(
                source=self.id,
                role=self.role,
                content=record.fulltext,
                records=[record],
            )

    async def get_record_dataset(self, record_id: str) -> Record | None:
        while not self._data_task.done():
            await asyncio.sleep(1)

        for dataset in self._data.values():
            rec = dataset.query("record_id==@record_id")
            if rec.shape[0] == 1:
                return Record(**rec.iloc[0].to_dict())
            if rec.shape[0] > 1:
                raise ValueError(
                    f"More than one record found for query record_id == {record_id}",
                )

        return None

    async def receive_output(
        self,
        message: GroupchatMessageTypes,
        **kwargs,
    ) -> GroupchatMessageTypes | None:
        """If running as an agent, watch for URLs or record ids and inject them into the chat."""
        if not isinstance(message, UserInstructions):
            return None
        if not (match := re.match(self._pat, message.content)):
            return None
        record = None
        if uri := match[2]:
            record = await download_and_convert(uri=uri)
        else:
            # Try to get by record_id (remove bang! first)
            record = await self.get_record_dataset(match[1]).strip("!")

        if record:
            return AgentOutput(
                source=self.id,
                role=self.role,
                content=record.fulltext,
                records=[record],
            )
        return None
