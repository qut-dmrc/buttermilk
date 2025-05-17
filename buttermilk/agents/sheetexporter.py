import pandas as pd
from pydantic import ConfigDict, Field, PrivateAttr

from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.contract import AgentInput, AgentTrace  # Import AgentInput and AgentTrace
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
    model_config = ConfigDict(extra="allow")

    async def process_job(  # Consider renaming this method to 'process' to align with Agent base class
        self,
        *,
        message: AgentInput,  # Changed parameter name and type
        **kwargs,
    ) -> AgentTrace:  # Changed return type
        # save the input data from this step to a spreadsheet so that we can compare later.
        from buttermilk.utils.gsheet import format_strings

        # should probably deal with parameters here somewhere

        if isinstance(message.inputs, dict):  # Access inputs from message
            inputs = [message.inputs]
        else:
            inputs = message.inputs

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

        # Create an AgentTrace object to return the results
        trace = AgentTrace(
            agent_id=self.agent_id,
            session_id=self.session_id,  # session_id is required for AgentTrace
            agent_info=self._cfg,  # agent_info is required for AgentTrace
            inputs=message,  # Include the original input message
            outputs=dict(sheet_url=sheet.url, **save_params),  # Store results in outputs
            # Add other relevant metadata if needed
        )

        logger.info(f"Saved to sheet {trace.outputs}")  # Log outputs from trace

        return trace  # Return the AgentTrace object
