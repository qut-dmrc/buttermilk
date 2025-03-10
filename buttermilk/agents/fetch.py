
import re
from autogen_core import (
    MessageContext,
    message_handler,
)

from buttermilk._core.agent import AgentConfig
from buttermilk._core.runner_types import RecordInfo
from buttermilk.bm import BM, logger
from buttermilk.runner.chat import (
    Answer,
    BaseGroupChatAgent,
    GroupChatMessage,
    InputRecord,
    RequestToSpeak,
)
from buttermilk.runner.helpers import prepare_step_df
from buttermilk.utils.media import download_and_convert
from buttermilk.utils.utils import extract_url
from re import match


class Fetch(BaseGroupChatAgent):
    def __init__(
        self,
        *,
        config: AgentConfig,
        group_chat_topic_type: str = "default",
    ) -> None:
        super().__init__(
            description=config.description,
            group_chat_topic_type=group_chat_topic_type,
        )
        self.bm = BM()
        self.config = config
        self.step = config.name
        self.parameters = config.parameters

        self._data = await prepare_step_df(self.data)

    def get_record(self, record_id: str) -> RecordInfo:
        rec = self._data.query("record_id==@record_id")
        if rec.shape[0] > 1:
            raise ValueError(
                f"More than one record found for query record_id == {record_id}",
            )

        return RecordInfo(**rec.iloc[0].to_dict())
    
    @message_handler
    async def handle_urls(
        self,
        message: GroupChatMessage | RequestToSpeak,
        ctx: MessageContext,
    ) -> None:
        
        record = None
        if uri := extract_url(message.content):
            record = await download_and_convert(uri=uri)
        elif match := re.match(r"`#([\s\w]+)`", message.content):
            record = self.get_record(match.group(0))
        
        if record:
            response = InputRecord(content=record.fulltext, payload=record, step=self.step)
            await self.publish(response)

            return response
        else:
            return None


    @message_handler
    async def handle_request_to_speak(
        self,
        message: RequestToSpeak,
        ctx: MessageContext,
    ) -> InputRecord:
        log_message = f"{self.id} from {self.step} got request to speak."

        logger.debug(log_message)

        if uri := extract_url(message.content):
            record = await download_and_convert(uri=uri)
        else:
            # check dataset for an ID
            id = re.sub() message.content

        response = InputRecord(content=record.fulltext, payload=record, step=self.step)
        await self.publish(response)

        return answer
