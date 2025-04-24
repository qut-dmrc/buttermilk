from functools import cached_property
from typing import Optional

import google.auth
import googleapiclient.discovery
import googleapiclient.errors
import gspread
import pandas as pd
import yaml
from pydantic import BaseModel

from buttermilk._core.log import logger
from buttermilk.utils.utils import make_serialisable, reset_index_and_dedup_columns


class GSheet(BaseModel):
    @cached_property
    def sheets(self):
        ## Make sure user logs in with drive credentials:
        ## gcloud auth login --enable-gdrive-access --update-adc --force [user gmail account]
        ##
        credentials = None
        credentials, _ = google.auth.default(
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
                "https://spreadsheets.google.com/feeds",
            ]
        )
        return googleapiclient.discovery.build("sheets", "v4", credentials=credentials)

    @cached_property
    def gspread_client(self):
        """Produce a gspread client."""
        import google.auth
        import gspread

        credentials, project_id = google.auth.default(
            scopes=[
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]
        )
        # google sheets client
        return gspread.authorize(credentials)

    def save_gsheet(
        self,
        df: pd.DataFrame,
        *,
        sheet_name: Optional[str] = None,
        title: Optional[str] = None,
        sheet_id: Optional[str] = None,
        uri: Optional[str] = None,
        **kwargs,
    ) -> gspread.spreadsheet.Spreadsheet:
        # drop the index so we can save it.
        df = reset_index_and_dedup_columns(df)

        # get pandas to infer dtypes by dumping out and back in
        df = pd.DataFrame(df.to_dict())

        if sheet_id:
            spreadsheet = self.gspread_client.open_by_key(key=sheet_id)
        elif uri:
            spreadsheet = self.gspread_client.open_by_url(url=uri)
        else:
            spreadsheet = self.gspread_client.create(title=title)

        if not sheet_name:
            worksheet = spreadsheet.get_worksheet(0)
            # add header row
            worksheet.append_rows(values=[df.columns.values.tolist()])
        else:
            try:
                worksheet = spreadsheet.worksheet(title=sheet_name)
                # cols = worksheet.
                #
            except gspread.exceptions.WorksheetNotFound:
                worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=1, cols=len(df.columns))

                # add header row
                worksheet.append_rows(values=[df.columns.values.tolist()])

                oldsheet = spreadsheet.get_worksheet(0)
                if oldsheet.title == "Sheet1":
                    try:
                        spreadsheet.del_worksheet(oldsheet)
                    except Exception as e:
                        logger.warning(f"Unable to delete default worksheet: {e} {e.args=}")

        # convert to string where we have to
        for col in df.columns:
            try:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str)
            except:
                df[col] = df[col].astype(str)

        # Add the dataframe rows to the end of the worksheet
        rows = make_serialisable(rows=df)  # .fillna(''))
        rows = [list(r.values()) for r in rows]
        worksheet.append_rows(rows)

        logger.info(f"Saved to {spreadsheet.url}")
        return spreadsheet


def format_strings(df, convert_json_columns: list = []):
    for col in convert_json_columns:
        if col in df.columns:
            # convert json columns
            df[col] = df[col].apply(lambda x: yaml.dump(x, default_flow_style=False, sort_keys=False))

    # for all the columns in the dataframe that are text columns, ensure that no cells are longer than 50,000 characters
    for col in df.select_dtypes(include=[object]).columns:  # type: ignore
        try:
            df[col] = df[col].str.slice(0, 50000)
        except:
            continue

    return df
