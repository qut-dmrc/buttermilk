import asyncio
from collections.abc import AsyncGenerator
from typing import Any, Callable, Coroutine, Self, Union

from autogen_core import MessageContext
import pydantic
import regex as re
from shortuuid import uuid

from buttermilk._core.agent import Agent, CancellationToken, FunctionTool, ToolConfig
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
    FlowMessage,
    GroupchatMessageTypes,
    ToolOutput,
    UserInstructions,
)
from buttermilk._core.types import Record
from buttermilk.runner.helpers import prepare_step_df
from buttermilk.utils.media import download_and_convert
from buttermilk.utils.utils import URL_PATTERN, extract_url

MATCH_PATTERNS = rf"^(![\d\w_]+)|<({URL_PATTERN})>"


class FetchRecord(Agent, ToolConfig):
    id: str = pydantic.Field(default_factory=lambda: f"fetch_record_{uuid()[:4]}" )
    role: str
    _data: dict[str, Any] = pydantic.PrivateAttr(default={})
    _data_task: asyncio.Task = pydantic.PrivateAttr()
    _pat: Any = pydantic.PrivateAttr(default_factory=lambda: re.compile(MATCH_PATTERNS))
    _fns: list[FunctionTool] = pydantic.PrivateAttr(default=[])
    _trace_this = True

    _message_types_handled: type[Any] = pydantic.PrivateAttr(default=Union[UserInstructions|AgentInput])
    @pydantic.model_validator(mode="after")
    def load_data_task(self) -> Self:
        self._data_task = asyncio.create_task(self.load_data())
        return self

    async def load_data(self):
        self._data = await prepare_step_df(self.data)

    def get_functions(self) -> list[Any]:
        """Create function definitions for this tool."""
        if not self._fns:
            self._fns = [FunctionTool(
                self._run,
                description=self.description, 
                name=self.role,
                strict=False,
            )]
        return self._fns

    async def _listen(
        self, message: GroupchatMessageTypes, cancellation_token: CancellationToken = None, publish_callback: Callable = None, **kwargs
    ) -> AsyncGenerator[GroupchatMessageTypes | None, None]:
        """Entry point when running this as an agent.

        If running as an agent, watch for URLs or record ids and inject them
        into the chat."""

        if not isinstance(message, UserInstructions):
            return
        if not (match := re.match(self._pat, message.content)):
            return

        record = None
        if uri := match[2]:
            record = await download_and_convert(uri=uri)
        else:
            # Try to get by record_id (remove bang! first)
            record_id = match[1].strip("!")
            record = await self._get_record_dataset(record_id=record_id)

        if record:
            output = AgentOutput(
                source=self.id,
                role=self.role,
                content=record._fulltext,
                records=[record],
            )

            yield output
        return

    async def _run( # type: ignore
        self,
        record_id: str | None = None,
        uri: str | None = None,
        prompt: str | None = None
    ) -> ToolOutput | None:
        """Entry point when running as a tool."""
        record  = None
        if prompt and not record_id and not uri:
            if not (uri := extract_url(prompt)):
                record_id = prompt.strip().strip("!")
        assert (record_id or uri) and not (record_id and uri), (
            "You must provide EITHER record_id OR uri."
        )
        result = None
        if record_id:
            record = await self._get_record_dataset(record_id)
            if record:
                result = ToolOutput(
                    role=self.role,
                    name=self.id,
                    results=[record],
                    content=record._fulltext,
                    messages=[record.as_message()],
                    args=dict(record_id=record_id),
                    send_to_ui=True,
                )
        else:
            record =await download_and_convert(uri)
            result = ToolOutput(
                role=self.role,
                name=self.id,
                results=[record],
                content=record._fulltext,
                messages=[record.as_message()],
                args=dict(uri=uri),
                send_to_ui=True,
            )

        if result:
            return  result
        return None

    async def _get_record_dataset(self, record_id: str) -> Record | None:
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
