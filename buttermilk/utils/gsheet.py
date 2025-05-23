"""Utilities for interacting with Google Sheets using the gspread library.

This module provides a `GSheet` class that encapsulates common operations
such as connecting to Google Sheets, opening or creating spreadsheets and
worksheets, and saving Pandas DataFrames to sheets. It also includes a
helper function `format_strings` for preparing DataFrames before uploading,
such as converting complex data types to JSON/YAML strings and truncating
long text cells.
"""

from functools import cached_property  # For lazy-loading properties
from typing import Any, Optional  # For type hinting

import google.auth  # For Google Cloud authentication
import googleapiclient.discovery  # For Google API client discovery (though less used with gspread directly)
import googleapiclient.errors  # For Google API errors
import gspread  # The primary library for Google Sheets interaction
import pandas as pd
import yaml  # For converting complex Python objects to YAML strings
from pydantic import BaseModel, ConfigDict  # Pydantic components

from buttermilk._core.log import logger  # Centralized logger
from buttermilk.utils.utils import make_serialisable, reset_index_and_dedup_columns  # Utility functions


class GSheet(BaseModel):
    """A utility class for interacting with Google Sheets.

    This class provides methods to connect to Google Sheets using default
    application credentials, open existing spreadsheets or create new ones,
    and save Pandas DataFrames to worksheets. It leverages the `gspread` library.

    Authentication Note:
        This class relies on Google Application Default Credentials (ADC).
        Ensure the environment is authenticated, typically by running:
        `gcloud auth application-default login --scopes=https://www.googleapis.com/auth/spreadsheets,https://www.googleapis.com/auth/drive`
        The `--enable-gdrive-access` flag mentioned in original comments for `gcloud auth login`
        might also be relevant depending on the exact ADC setup and permissions required.

    Attributes:
        sheets (googleapiclient.discovery.Resource): A Google API client resource
            for the Sheets API (v4). This is a lower-level client.
            (Note: `gspread_client` is typically preferred for most operations).
        gspread_client (gspread.Client): An authorized `gspread` client instance,
            used for most sheet operations like opening, creating, and writing.
        model_config (ConfigDict): Pydantic model configuration.
            - `arbitrary_types_allowed`: True.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @cached_property
    def sheets(self) -> googleapiclient.discovery.Resource:
        """Provides a Google API client resource for the Google Sheets API (v4).

        This client is initialized using default application credentials with scopes
        for spreadsheets, drive, and feeds. It's a lower-level client compared to
        the `gspread_client`.

        Note:
            For most common operations like reading or writing sheet data,
            `self.gspread_client` is generally more convenient.
            Ensure ADC is configured with drive and sheets scopes:
            `gcloud auth application-default login --scopes=https://www.googleapis.com/auth/spreadsheets,https://www.googleapis.com/auth/drive`

        Returns:
            googleapiclient.discovery.Resource: An initialized Google Sheets API client resource.
        
        Raises:
            google.auth.exceptions.DefaultCredentialsError: If ADC are not found or invalid.
        """
        credentials, _ = google.auth.default(
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive", # Drive scope often needed for creating/finding sheets
                "https://spreadsheets.google.com/feeds", # Older scope, but sometimes included
            ]
        )
        return googleapiclient.discovery.build("sheets", "v4", credentials=credentials)

    @cached_property
    def gspread_client(self) -> gspread.Client:
        """Provides an authorized `gspread.Client` instance.

        This client is the primary interface for interacting with Google Sheets
        using the `gspread` library. It's initialized using default application
        credentials with the necessary scopes for spreadsheets and drive access.

        Note:
            Ensure ADC is configured with drive and sheets scopes:
            `gcloud auth application-default login --scopes=https://www.googleapis.com/auth/spreadsheets,https://www.googleapis.com/auth/drive`

        Returns:
            gspread.Client: An authorized `gspread` client.
        
        Raises:
            google.auth.exceptions.DefaultCredentialsError: If ADC are not found or invalid.
        """
        credentials, _ = google.auth.default( # project_id is not strictly needed for gspread authorize
            scopes=[
                "https://spreadsheets.google.com/feeds", # Legacy, but often included
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive", # Needed for creating new sheets or searching Drive
            ]
        )
        return gspread.authorize(credentials)

    def save_gsheet(
        self,
        df: pd.DataFrame,
        *, # Force subsequent arguments to be keyword-only
        sheet_name: Optional[str] = None,
        title: Optional[str] = None, # Title for a new spreadsheet
        sheet_id: Optional[str] = None, # Google Spreadsheet ID (key)
        uri: Optional[str] = None, # Full URL to the Google Spreadsheet
        header_format: Optional[dict[str, Any]] = None, # gspread cell format for header
        **kwargs: Any, # Catch-all for future gspread options
    ) -> gspread.spreadsheet.Spreadsheet:
        """Saves a Pandas DataFrame to a specified Google Sheet.

        This method handles opening an existing spreadsheet (by ID or URI) or
        creating a new one (if `title` is provided and `sheet_id`/`uri` are not).
        It then finds or creates a worksheet within that spreadsheet by `sheet_name`.
        The DataFrame's content (headers and rows) is appended to this worksheet.

        Args:
            df (pd.DataFrame): The Pandas DataFrame to save.
            sheet_name (Optional[str]): The name of the worksheet (tab) to save
                data to. If None, the first worksheet (index 0) is used or a new
                one is created with a default name if the spreadsheet is new.
            title (Optional[str]): If creating a new spreadsheet (i.e., `sheet_id`
                and `uri` are None), this title will be used for the new spreadsheet.
            sheet_id (Optional[str]): The ID (key) of an existing Google Spreadsheet
                to open.
            uri (Optional[str]): The full URL of an existing Google Spreadsheet to open.
                Takes precedence over `sheet_id` if both are provided.
            header_format (Optional[dict[str, Any]]): An optional dictionary defining
                gspread cell formatting for the header row (e.g., `{"textFormat": {"bold": True}}`).
            **kwargs: Additional keyword arguments (currently not used directly but
                      available for future extensions).

        Returns:
            gspread.spreadsheet.Spreadsheet: The `gspread.Spreadsheet` object
            that was written to.

        Raises:
            gspread.exceptions.APIError: For errors during interaction with the
                Google Sheets API.
            ValueError: If attempting to create a new sheet but `title` is not provided.
        """
        if df.empty:
            logger.warning("Input DataFrame is empty. No data will be saved to Google Sheet.")
            # Depending on desired behavior, could return None or raise error,
            # or create an empty sheet. For now, let's try to get/create the sheet.

        # Ensure DataFrame index is suitable for saving (e.g., no complex multi-index)
        df_to_save = reset_index_and_dedup_columns(df.copy())

        # Infer dtypes again after potential dict conversion from to_dict()
        # This step might be redundant if df_to_save is already well-typed.
        # df_to_save = pd.DataFrame(df_to_save.to_dict()) # This line might be problematic or unnecessary

        spreadsheet: gspread.Spreadsheet
        if uri:
            spreadsheet = self.gspread_client.open_by_url(uri)
        elif sheet_id:
            spreadsheet = self.gspread_client.open_by_key(sheet_id)
        elif title:
            spreadsheet = self.gspread_client.create(title)
        else:
            raise ValueError("Must provide either 'sheet_id', 'uri', or 'title' (for new sheet) to save_gsheet.")

        worksheet: gspread.Worksheet
        if not sheet_name: # Use the first sheet or create one if needed
            try:
                worksheet = spreadsheet.get_worksheet(0) # Get the first sheet
                logger.info(f"Using existing first worksheet '{worksheet.title}' in spreadsheet '{spreadsheet.title}'.")
                 # Clear existing content and append new headers + data
                worksheet.clear()
                worksheet.append_rows(values=[df_to_save.columns.values.tolist()], table_range="A1")
                if header_format: worksheet.format("A1:Z1", header_format) # Format header
            except gspread.exceptions.WorksheetNotFound: # Should not happen for index 0 unless sheet is truly empty
                 worksheet = spreadsheet.add_worksheet(title="Sheet1", rows=1, cols=max(1, len(df_to_save.columns)))
                 logger.info(f"Created new worksheet 'Sheet1' in spreadsheet '{spreadsheet.title}'.")
                 worksheet.append_rows(values=[df_to_save.columns.values.tolist()], table_range="A1")
                 if header_format: worksheet.format("A1:Z1", header_format)
        else: # Specific sheet_name provided
            try:
                worksheet = spreadsheet.worksheet(sheet_name)
                logger.info(f"Using existing worksheet '{sheet_name}' in spreadsheet '{spreadsheet.title}'. Appending data.")
                # For existing named sheet, we append. Consider if clearing is needed based on use case.
            except gspread.exceptions.WorksheetNotFound:
                worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=1, cols=max(1, len(df_to_save.columns)))
                logger.info(f"Created new worksheet '{sheet_name}' in spreadsheet '{spreadsheet.title}'.")
                # Add header row to the new sheet
                worksheet.append_rows(values=[df_to_save.columns.values.tolist()], table_range="A1")
                if header_format: worksheet.format("A1:Z1", header_format) # Format header

                # Original logic to delete default "Sheet1" if a new named sheet is created
                # and the spreadsheet was just created (implying it only had "Sheet1").
                # This is a bit heuristic.
                if title and len(spreadsheet.worksheets()) > 1: # If we just created this spreadsheet via title
                    default_sheet = spreadsheet.get_worksheet(0) # Index 0 might now be our new sheet if "Sheet1" was deleted
                    if default_sheet and default_sheet.title == "Sheet1" and default_sheet.id != worksheet.id:
                        try:
                            spreadsheet.del_worksheet(default_sheet)
                            logger.info("Deleted default 'Sheet1' after creating new named worksheet.")
                        except Exception as e_del:
                            logger.warning(f"Unable to delete default worksheet 'Sheet1': {e_del!s}")

        # Prepare data for appending: convert objects to strings, handle complex types
        # This loop attempts to convert all object columns to string, which might be too aggressive.
        # The `format_strings` function (called later if used) is more targeted.
        # Consider removing this generic loop if `format_strings` is always sufficient.
        for col in df_to_save.columns:
            if df_to_save[col].dtype == "object": # Check for object dtype
                try:
                    # Attempt conversion to string, handling potential errors for mixed types
                    df_to_save[col] = df_to_save[col].astype(str)
                except Exception as e_astype:
                    logger.warning(f"Could not convert column '{col}' to string type directly: {e_astype!s}. Values might be mixed.")
                    # Fallback: apply str() element-wise for complex objects within the column
                    df_to_save[col] = df_to_save[col].apply(lambda x: str(x) if pd.notnull(x) else "")


        # Use make_serialisable to handle complex types (like dicts/lists in cells) for gspread
        # gspread expects a list of lists for append_rows.
        serialisable_rows = make_serialisable(rows=df_to_save.to_dict(orient="records"))
        if isinstance(serialisable_rows, list) and all(isinstance(r, dict) for r in serialisable_rows):
            rows_to_append = [list(r.values()) for r in serialisable_rows]
            if rows_to_append: # Only append if there's data
                 worksheet.append_rows(rows_to_append, value_input_option="USER_ENTERED")
                 logger.info(f"Appended {len(rows_to_append)} rows to worksheet '{worksheet.title}'.")
            else:
                 logger.info(f"No rows to append to worksheet '{worksheet.title}' after serialization.")
        else:
            logger.error(f"Data for worksheet '{worksheet.title}' could not be serialized into list of lists. Type: {type(serialisable_rows)}")


        logger.info(f"Data saved to Google Sheet: {spreadsheet.url} (Worksheet: '{worksheet.title}')")
        return spreadsheet


def format_strings(df: pd.DataFrame, convert_json_columns: list[str] | None = None) -> pd.DataFrame:
    """Formats a Pandas DataFrame for better compatibility with Google Sheets.

    This function performs two main operations:
    1.  Converts specified columns (in `convert_json_columns`) containing complex
        Python objects (like dicts or lists) into YAML string representations.
        This makes them more readable in the spreadsheet.
    2.  Truncates all string cells (object dtype columns) to a maximum length
        of 50,000 characters, which is Google Sheets' cell character limit.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame to format.
        convert_json_columns (list[str] | None): An optional list of column names
            that should be converted to YAML/JSON strings. If None, no columns
            are converted this way. Defaults to None.

    Returns:
        pd.DataFrame: The modified DataFrame with formatted strings.
                      The operation is done in-place on a copy if modifications occur.
    """
    df_formatted = df.copy() # Work on a copy to avoid modifying original DataFrame

    if convert_json_columns:
        for col_name in convert_json_columns:
            if col_name in df_formatted.columns:
                try:
                    # Convert column to YAML string representation for readability in sheets
                    df_formatted[col_name] = df_formatted[col_name].apply(
                        lambda x: yaml.dump(x, default_flow_style=False, sort_keys=False, allow_unicode=True)
                        if pd.notnull(x) else "" # Handle NaNs gracefully
                    )
                except Exception as e_yaml:
                    logger.warning(f"Could not convert column '{col_name}' to YAML/JSON string: {e_yaml!s}. Column skipped for this conversion.")

    # Truncate all object (likely string) columns to Google Sheets cell character limit
    for col_name in df_formatted.select_dtypes(include=["object"]).columns:
        try:
            # Ensure column is string type before attempting string operations
            df_formatted[col_name] = df_formatted[col_name].astype(str).str.slice(0, 49999) # Leave a little buffer
        except Exception as e_slice: # Catch errors if astype(str) or slice fails
            logger.warning(f"Could not truncate string column '{col_name}': {e_slice!s}. Column may contain non-stringifiable data.")
            # Optionally, try a more robust conversion for problematic cells in this column
            # For now, just logs warning and continues.

    return df_formatted
