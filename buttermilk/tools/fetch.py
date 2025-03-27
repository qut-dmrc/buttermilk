import asyncio
from typing import Any, Self

import pydantic
import regex as re

from buttermilk._core.agent import Agent, ToolConfig
from buttermilk._core.contract import AgentInput, AgentMessages, AgentOutput, UserInput
from buttermilk._core.runner_types import Record
from buttermilk.runner.helpers import prepare_step_df
from buttermilk.utils.media import download_and_convert
from buttermilk.utils.utils import URL_PATTERN, extract_url

MATCH_PATTERNS = rf"^(![\d\w_]+)|<({URL_PATTERN})>"


class Fetch(Agent, ToolConfig):
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
        input_data: AgentInput,
        **kwargs,
    ) -> AgentOutput | None:
        """Entry point when running this as an agent."""
        if "uri" in input_data.inputs or "record_id" in input_data.inputs:
            record = await self._run(
                uri=input_data.inputs.get("uri"),
                record_id=input_data.inputs.get("record_id"),
            )
        else:
            prompt = input_data.inputs["prompt"]
            if uri := extract_url(prompt):
                record = await self._run(uri=uri)
            else:
                record = await self._run(record_id=prompt)
        return AgentOutput(
            agent_id=self.id,
            agent_name=self.name,
            content=record.fulltext,
            records=[record],
        )

    async def _run(
        self,
        record_id: str | None = None,
        uri: str | None = None,
    ) -> Record | None:
        """Entry point when running as a tool."""
        assert (record_id or uri) and not (record_id and uri), (
            "You must provide EITHER record_id OR uri."
        )
        if record_id:
            return await self.get_record_dataset(record_id)
        return await download_and_convert(uri)
        return None

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
        message: AgentMessages | UserInput,
        source: str,
        **kwargs,
    ) -> AgentMessages | None:
        """Watch for URLs or record ids and inject them into the chat."""
        # not built yet.
        if not isinstance(message, UserInput):
            return None
        if not (match := re.match(self._pat, message.content)):
            return None
        record = None
        if uri := match[2]:
            record = await download_and_convert(uri=uri)
        else:
            # Try to get by record_id (remove bang! first)
            record = await self.get_record_dataset(match[1])

        if record:
            return AgentOutput(
                agent_id=self.id,
                agent_name=self.name,
                content=record.fulltext,
                records=[record],
            )
        return None
