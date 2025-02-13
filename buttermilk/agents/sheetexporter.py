from typing import Any

import pandas as pd
from pydantic import Field, PrivateAttr

from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.runner_types import Job
from buttermilk.utils.gsheet import GSheet


class GSheetExporter(Agent):
    name: str = Field(default="gsheetexporter", init=False)
    flow: str | None = Field(
        default=None,
        init=False,
        description="The name of the flow or step in the process that this agent is responsible for.",
    )

    convert_json_columns: list[str] = Field(
        default=[],
        description="List of columns to convert to JSON.",
    )

    _gsheet: GSheet = PrivateAttr(default_factory=GSheet)

    class Config:
        extra = "allow"

    async def process_job(
        self,
        *,
        job: Job,
        additional_data: Any = None,
        **kwargs,
    ) -> Job:
        # save the input data from this step to a spreadsheet so that we can compare later.
        from buttermilk.utils.gsheet import format_strings

        if dataset_name := job.parameters.get("from_dataset"):
            dataset = pd.DataFrame.from_records(additional_data[dataset_name])
        else:
            # combine input vars and params together in this instance
            inputs = {}
            inputs.update(job.inputs)
            if job.parameters:
                inputs.update(job.parameters)
            dataset = pd.DataFrame.from_records(inputs)

        contents = format_strings(
            dataset,
            convert_json_columns=self.convert_json_columns,
        )

        save_params = {}
        if self.save:
            save_params = self.save.model_dump()

        sheet = self._gsheet.save_gsheet(df=contents, **save_params)
        save_params["sheet_id"] = sheet.id
        job.outputs = dict(sheet_url=sheet.url, **save_params)

        logger.info(f"Saved to sheet {job.outputs}")

        return job
