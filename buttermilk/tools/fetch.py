import asyncio
import re
from typing import Any, Self

import pydantic
from pydantic import PrivateAttr

from buttermilk._core.agent import ToolConfig
from buttermilk._core.runner_types import Record
from buttermilk.runner.helpers import prepare_step_df
from buttermilk.utils.media import download_and_convert
from buttermilk.utils.utils import extract_url


class Fetch(ToolConfig):
    _data: dict[str, Any] = PrivateAttr(default={})
    _data_task: asyncio.Task = PrivateAttr()

    @pydantic.model_validator(mode="after")
    def load_data(self) -> Self:
        self._data_task = asyncio.create_task(prepare_step_df(self.data_cfg))
        return self

    async def _run(self, record_id: str) -> Record | None:
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

    async def get_record(
        self,
        record_id: str | None = None,
        uri: str | None = None,
    ) -> Record | None:
        assert (record_id or uri) and not (record_id and uri), (
            "You must provide EITHER record_id OR uri."
        )
        if record_id:
            # Try to get by record_id
            match = re.match(r"!([\d\w_]+)", record_id)
            record = await self.get_record_dataset(match.group(1))
            return record
        if uri:
            uri = extract_url(uri)
            record = download_and_convert(uri=uri)
            return record
        return None
