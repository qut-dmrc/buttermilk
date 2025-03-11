
import asyncio
import re

from autogen_core import (
    MessageContext,
    message_handler,
)

from buttermilk._core.agent import AgentConfig
from buttermilk._core.runner_types import RecordInfo
from buttermilk.runner.chat import (
    BaseGroupChatAgent,
    GroupChatMessage,
    InputRecord,
    NullAnswer,
    RequestToSpeak,
)
from buttermilk.runner.helpers import prepare_step_df
from buttermilk.utils.media import download_and_convert
from buttermilk.utils.utils import extract_url


class Fetch(BaseGroupChatAgent):
    def __init__(
        self,
        *,
        config: AgentConfig,
        group_chat_topic_type: str = "default",
    ) -> None:
        super().__init__(
            config=config,
            group_chat_topic_type=group_chat_topic_type,
        )
        self._data = None
        self._data_task = asyncio.get_running_loop().create_task(self.load_data())

    async def load_data(self):
        self._data = await prepare_step_df(self.config.data)

    async def get_record_dataset(self, record_id: str) -> RecordInfo | None:
        while not self._data_task.done():
            await asyncio.sleep(1)
        for dataset in self._data.values():
            rec = dataset.query("record_id==@record_id")
            if rec.shape[0] == 1:
                return RecordInfo(**rec.iloc[0].to_dict())
            if rec.shape[0] > 1:
                raise ValueError(
                    f"More than one record found for query record_id == {record_id}",
                )

        return None

    async def get_record(self, message: str) -> RecordInfo:
        record = None
        if match := re.match(r"`#([\s\w]+)`", message):
            # Try to get by record_id
            record = await self.get_record_dataset(match.group(1))
        elif uri := extract_url(message):
            # Try to download
            record = await download_and_convert(uri=uri)

        return record

    async def query(
        self,
        request: RequestToSpeak | GroupChatMessage,
    ) -> InputRecord | NullAnswer:
        record = None
        inputs = []
        if isinstance(request, RequestToSpeak):
            # Check the extra fields in the request
            for key, value in self.config.inputs.items():
                inputs.extend([r.content for r in request.placeholders.get(value, [])])
                inputs.extend([r for r in request.inputs.get(value, [])])
                if value == "context" and request.context:
                    inputs.extend([r.content for r in request.context])

        if request.content:
            inputs.append(request.content)
        for data_var in inputs:
            # Get the first matching input we can find
            record = await self.get_record(data_var)
            if record:
                break

        if not record and isinstance(request, RequestToSpeak):
            # We must return an input record if we were asked for one
            # Create a new record with the original text we were given.
            record = RecordInfo(data=request.content)

        if record:
            response = InputRecord(
                content=record.fulltext,
                payload=record,
                step=self.step,
            )
            await self.publish(response)
            return response

        return NullAnswer(step=self.step)

    @message_handler
    async def handle_urls(
        self,
        message: GroupChatMessage,
        ctx: MessageContext,
    ) -> InputRecord | NullAnswer:
        return await self.query(message)
