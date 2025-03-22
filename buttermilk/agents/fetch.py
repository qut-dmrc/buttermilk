
import asyncio
import re
from typing import Any

from pydantic import PrivateAttr

from buttermilk._core.agent import Agent
from buttermilk._core.contract import AgentInput, AgentOutput
from buttermilk._core.runner_types import Record
from buttermilk.runner.helpers import prepare_step_df
from buttermilk.utils.media import download_and_convert
from buttermilk.utils.utils import extract_url

# from autogen_core import CancellationToken
# from autogen_core.tools import FunctionTool
# from typing_extensions import Annotated

# async def example():
#     # Initialize a FunctionTool instance for retrieving stock prices.
#     get_record_tool = FunctionTool(
#         get_record, description="Fetch the stock price for a given ticker."
#     )

#     # Execute the tool with cancellation support.
#     cancellation_token = CancellationToken()
#     result = await get_record_tool.run_json(
#         {"ticker": "AAPL", "date": "2021/01/01"}, cancellation_token
#     )

#     # Output the result as a formatted string.
#     print(stock_price_tool.return_value_as_string(result))


class Fetch(Agent):
    _data_task: Any = PrivateAttr(default=None)

    async def load_data(self):
        self._data = await prepare_step_df(self.data)

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

    async def get_record(self, message: str) -> Record | None:
        record = None
        if match := re.match(r"!([\d\w_]+)", message):
            # Try to get by record_id
            record = await self.get_record_dataset(match.group(1))
        elif uri := extract_url(message):
            # Try to download
            record = await download_and_convert(uri=uri)

        return record

    async def process(self, input_data: AgentInput) -> AgentOutput:
        if input_data.prompt:
            record = await self.get_record(input_data.prompt)
            if record:
                return AgentOutput(
                    agent=self.id,
                    records=[record],
                )
            return AgentOutput(
                agent=self.id,
                records=[Record(data=input_data.prompt)],
            )
        return AgentOutput(agent=self.id, error="No input provided.")
