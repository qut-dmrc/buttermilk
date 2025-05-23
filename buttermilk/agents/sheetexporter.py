"""Provides the GSheetExporter agent for exporting data to Google Sheets.

This module defines the `GSheetExporter` agent, which takes input data,
formats it into a tabular structure (Pandas DataFrame), and then uploads it
to a specified Google Sheet using the `GSheet` utility.
"""

from typing import Any  # For type hinting

import pandas as pd
from pydantic import ConfigDict, Field, PrivateAttr  # Pydantic components

from buttermilk import logger  # Centralized logger
from buttermilk._core.agent import Agent  # Buttermilk base Agent class
from buttermilk._core.contract import AgentInput, AgentTrace  # Buttermilk message contracts
from buttermilk.utils.gsheet import GSheet  # Utility for Google Sheets interaction


class GSheetExporter(Agent):
    """An agent that exports data to a Google Sheet.

    The `GSheetExporter` agent is designed to take input data, typically from
    an `AgentInput` message's `inputs` field, convert it into a Pandas DataFrame,
    perform some basic formatting (like converting specified columns to JSON strings),
    and then upload this DataFrame to a Google Sheet.

    The destination Google Sheet (ID, sheet name, etc.) is configured via the
    `save` attribute (an instance of `SaveInfo`, typically populated from
    `AgentConfig.save` in the Hydra configuration).

    Key Configuration:
        - `save` (SaveInfo): **Required**. In `AgentConfig`, this should specify
          the `type: "gsheets"` and details like `spreadsheet_id`, `sheet_name`, etc.
          See `buttermilk._core.config.SaveInfo` and `buttermilk.utils.gsheet.GSheet`.
        - `convert_json_columns` (list[str]): Optional. A list of column names
          in the input data that should be explicitly converted to JSON strings
          before uploading to the sheet. This is useful for columns containing
          complex objects or lists.

    Input:
        Expects an `AgentInput` message. The data to be exported is taken from
        `message.inputs`. If `message.inputs` is a dictionary, it's treated as a
        single row. If it's a list of dictionaries, each dictionary becomes a row.

    Output:
        Returns an `AgentTrace` where the `outputs` field contains a dictionary
        with details of the saved sheet, including `sheet_url` and `sheet_id`.
    """
    # Note: 'name' and 'flow' are Pydantic fields with init=False, meaning they are
    # class variables or intended to be set post-initialization, not via constructor.
    # Their usage pattern might need clarification if they are meant for dynamic config.
    name: str = Field(
        default="gsheetexporter",
        init=False, # Not initialized via __init__ args, treated as class var or set later
        description="Default name of the GSheetExporter agent."
    )
    flow: str | None = Field(
        default=None,
        init=False,
        description="Optional name of the flow or process step this exporter is part of.",
    )

    convert_json_columns: list[str] = Field(
        default_factory=list, # Use factory for mutable default
        description="List of column names from the input data that should be converted to JSON strings before saving to the sheet.",
    )

    _gsheet: GSheet = PrivateAttr(default_factory=GSheet)
    """Private attribute for the `GSheet` utility instance used for interacting
    with the Google Sheets API. Initialized on first use.
    """

    model_config = ConfigDict(extra="allow") # Allow extra fields if passed in config

    async def _process( # Renamed from process_job to align with Agent base class
        self,
        *,
        message: AgentInput,
        **kwargs: Any, # Allow for additional keyword arguments from base class or callers
    ) -> AgentTrace:
        """Processes the input data and exports it to a Google Sheet.

        This method takes data from `message.inputs`, converts it to a Pandas
        DataFrame, formats specified columns as JSON strings, and then uses
        the `_gsheet` utility to save the DataFrame to the configured Google Sheet.
        Details of the target sheet are taken from `self.save` (a `SaveInfo` object).

        Args:
            message: The `AgentInput` message containing the data to export in
                its `inputs` field. `message.inputs` can be a single dictionary
                (for one row) or a list of dictionaries (for multiple rows).
            **kwargs: Additional keyword arguments (not directly used in this method
                      but available for future extensions or if called by a wrapper).

        Returns:
            AgentTrace: An `AgentTrace` object. The `outputs` attribute of the
            trace will contain a dictionary with `sheet_url` and `sheet_id` of
            the Google Sheet where data was saved, along with other save parameters.
            If `self.save` is not configured, it might log an error or behave
            unexpectedly depending on `_gsheet.save_gsheet` implementation.
        """
        from buttermilk.utils.gsheet import format_strings  # Local import for utility

        if not message.inputs:
            logger.warning(f"GSheetExporter '{self.agent_id}': Received message with no inputs to export.")
            # Return an empty or error trace
            return AgentTrace(
                agent_id=self.agent_id,
                session_id=self.session_id,
                agent_info=self._cfg,
                inputs=message,
                outputs={"status": "No data provided in inputs."},
                error=["No data in message.inputs to export."]
            )

        # Ensure inputs is a list of records for DataFrame conversion
        input_data_list: list[dict[str, Any]]
        if isinstance(message.inputs, dict):
            input_data_list = [message.inputs]
        elif isinstance(message.inputs, list):
            input_data_list = message.inputs # type: ignore # Assuming list of dicts
        else:
            logger.error(f"GSheetExporter '{self.agent_id}': message.inputs is not a dict or list of dicts. Type: {type(message.inputs)}")
            return AgentTrace(
                agent_id=self.agent_id,
                session_id=self.session_id,
                agent_info=self._cfg,
                inputs=message,
                outputs={"status": "Invalid input data format."},
                error=[f"message.inputs type {type(message.inputs)} not supported."]
            )

        try:
            dataset_df = pd.DataFrame.from_records(input_data_list)
        except Exception as e:
            logger.error(f"GSheetExporter '{self.agent_id}': Failed to create DataFrame from inputs: {e!s}", exc_info=True)
            return AgentTrace(
                agent_id=self.agent_id,
                session_id=self.session_id,
                agent_info=self._cfg,
                inputs=message,
                outputs={"status": "DataFrame creation failed."},
                error=[f"Failed to create DataFrame: {e!s}"]
            )

        if dataset_df.empty:
            logger.info(f"GSheetExporter '{self.agent_id}': Input data resulted in an empty DataFrame. Nothing to export.")
            return AgentTrace(
                agent_id=self.agent_id,
                session_id=self.session_id,
                agent_info=self._cfg,
                inputs=message,
                outputs={"status": "Empty dataset, nothing exported."},
            )

        formatted_contents_df = format_strings(
            dataset_df,
            convert_json_columns=self.convert_json_columns,
        )

        save_config_params = {}
        if self.save: # self.save is an instance of SaveInfo from AgentConfig
            save_config_params = self.save.model_dump(exclude_none=True)
        else:
            logger.warning(f"GSheetExporter '{self.agent_id}': No 'save' configuration found. Attempting to save to GSheet with default parameters if GSheet utility supports it.")
            # Depending on GSheet.save_gsheet behavior, this might fail or use defaults.

        try:
            sheet_info = self._gsheet.save_gsheet(df=formatted_contents_df, **save_config_params)
            # Assuming sheet_info has 'id' and 'url' attributes as per original code
            output_payload = {"sheet_url": sheet_info.url, "sheet_id": sheet_info.id, **save_config_params}
            logger.info(f"GSheetExporter '{self.agent_id}': Successfully saved data to Google Sheet. URL: {sheet_info.url}, ID: {sheet_info.id}")
        except Exception as e:
            logger.error(f"GSheetExporter '{self.agent_id}': Failed to save data to Google Sheet: {e!s}", exc_info=True)
            return AgentTrace(
                agent_id=self.agent_id,
                session_id=self.session_id,
                agent_info=self._cfg,
                inputs=message,
                outputs={"status": "Failed to save to Google Sheet."},
                error=[f"GSheet save error: {e!s}"]
            )

        trace = AgentTrace(
            agent_id=self.agent_id,
            session_id=self.session_id,
            agent_info=self._cfg,
            inputs=message,
            outputs=output_payload,
        )
        return trace
