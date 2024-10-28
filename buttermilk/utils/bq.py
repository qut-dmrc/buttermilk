import datetime
from typing import Optional, Sequence
from pydantic import BaseModel, ConfigDict, Field
import pydantic
from google.cloud import bigquery_storage_v1beta2
from google.cloud.bigquery_storage_v1beta2.services.big_query_write.async_client import (
    BigQueryWriteAsyncClient,
)
from google.cloud import bigquery
import pandas as pd
from pydantic import BaseModel, Field
from .log import logger
from google.cloud.bigquery_storage_v1beta2.types import (
    AppendRowsRequest,
    CreateWriteStreamRequest,
    ProtoRows,
    ProtoSchema,
    WriteStream,
)

from .utils import make_serialisable, remove_punctuation
import json

def construct_dict_from_schema(schema: list, d: dict, remove_extra=True):
    """Recursively construct a new dictionary, changing data types to match BigQuery.
    Only uses fields from d that are in schema.

    INPUT: schema - list of dictionaries, each with keys 'name', 'type', and optionally 'fields'
    """

    new_dict = {}
    keys_deleted = []

    for row in schema:
        if isinstance(row, bigquery.SchemaField):
            row = row.to_api_repr()
        key_name = row["name"]
        if key_name not in d:
            keys_deleted.append(key_name)
            continue

        # Handle nested fields
        if isinstance(d[key_name], dict) and "fields" in row:
            new_dict[key_name] = construct_dict_from_schema(row["fields"], d[key_name])

        # Handle repeated fields - use the same schema as we were passed
        elif isinstance(d[key_name], list) and "fields" in row:
            new_dict[key_name] = [
                construct_dict_from_schema(row["fields"], item) for item in d[key_name]
            ]

        elif isinstance(d[key_name], str) and (
            str.upper(remove_punctuation(d[key_name])) == "NULL"
            or remove_punctuation(d[key_name]) == ""
        ):
            # don't add null values
            keys_deleted.append(key_name)
            continue

        elif str.upper(row["type"]) in ["TIMESTAMP", "DATETIME", "DATE"]:
            # convert string dates to datetimes
            if not isinstance(d[key_name], datetime.datetime):
                _ts = None
                if type(d[key_name]) is str:
                    if d[key_name].isnumeric():
                        _ts = float(d[key_name])
                    else:
                        new_dict[key_name] = pd.to_datetime(d[key_name])

                if type(d[key_name]) is int or type(d[key_name]) is float or _ts:
                    if not _ts:
                        _ts = d[key_name]

                    try:
                        new_dict[key_name] = datetime.datetime.utcfromtimestamp(_ts)
                    except (ValueError, OSError):
                        # time is likely in milliseconds
                        new_dict[key_name] = datetime.datetime.utcfromtimestamp(
                            _ts / 1000
                        )

                elif not isinstance(d[key_name], datetime.datetime):
                    new_dict[key_name] = pd.to_datetime(d[key_name])
            else:
                # Already a datetime, move it over
                new_dict[key_name] = d[key_name]

            # if it's a date only field, remove time
            if str.upper(row["type"]) == "DATE":
                new_dict[key_name] = new_dict[key_name].date()

        elif str.upper(row["type"]) in ["INTEGER", "FLOAT"]:
            # convert string numbers to integers
            if isinstance(d[key_name], str):
                new_dict[key_name] = pd.to_numeric(d[key_name])
            else:
                new_dict[key_name] = d[key_name]

        elif str.upper(row["type"]) == "BOOLEAN":
            if isinstance(d[key_name], str):
                try:
                    new_dict[key_name] = pydantic.TypeAdapter(bool).validate_python(d[key_name])
                except ValueError as e:
                    if new_dict[key_name] == "":
                        pass  # no value
                    raise e
            else:
                new_dict[key_name] = pydantic.TypeAdapter(bool).validate_python(d[key_name])

        else:
            new_dict[key_name] = d[key_name]

    return new_dict


class TableWriter(BaseModel):
    """
        Args:
            destination: fully qualified table ID in the form `dataset.project.table`
            project_id: The ID of the Google Cloud project.
            dataset_id: The ID of the dataset containing the table.
            table_id: The ID of the table to write to.
            schema: Optional schema to validate against.
    """

    write_client:  BigQueryWriteAsyncClient = Field(default_factory=BigQueryWriteAsyncClient)
    destination: Optional[str] = None
    schema: Optional[str] = None
    stream: str = '_default'  # fully qualified table ID in the form `dataset.project.table`
    project: str
    dataset_id: Optional[str] = None
    table_id: Optional[str] = None
    table_path: str = Field(init=False)

    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, populate_by_name=True
    )
    
    @pydantic.field_validator('schema', mode='before')
    def load_schema(cls, v) -> dict:
        if isinstance(v, str):
            return bigquery.Client().schema_from_json(v)
        
    @pydantic.field_validator('table_path', mode='after')
    def make_table_path(cls, v, values) -> str:
        if values.get('project_id') and values.get('dataset_id') and values.get('table_id'):
            pass
        elif values.get('table'):
            values['project_id'], values['dataset_id'], values['table_id'] = v.split('.')
        else:
            raise ValueError("Table path not provided")
        return f"projects/{values['project_id']}/datasets/{values['dataset_id']}/tables/{values['table_id']}"
    

    async def append_rows(self, rows: list) -> Sequence[bool]:
        """Appends rows to a BigQuery table asynchronously using the BigQuery Storage Write API.

        Args:
            rows: A list of dictionaries representing the rows to append. Each dictionary
                  should have keys corresponding to the table's column names.
        """
        write_stream = f'{self.table_path}/streams/{self.stream}'


        if isinstance(rows, pd.DataFrame):
            # deduplicate columns
            rows.columns = [x[1] if x[1] not in rows.columns[:x[0]] else f"{x[1]}_{list(rows.columns[:x[0]]).count(x[1])}" for x in enumerate(rows.columns)]

            batch = rows.to_dict(orient="records")
        else:
            batch = rows.copy()

        # Handle other conversions required for bigquery
        if self.schema:
            batch = [construct_dict_from_schema(self.schema, row) for row in batch]

        batch = make_serialisable(batch)

        if not batch:
            logger.warning("No rows found in save function.")
            return None

        # format the rows
        batch = ProtoRows(
                    serialized_rows=batch
                )
        
        # Construct the request
        request = bigquery_storage_v1beta2.types.storage.AppendRowsRequest(write_stream=write_stream, proto_rows=batch)

        # Send the request asynchronously and get a stream of responses
        stream = await self.write_client.append_rows([request])
        
        # Handle the responses asynchronously
        results = []
        async for response in stream:
            results.append(True)  # TODO: check whether insert was successful
            # Process the response here if needed
            logger.debug(f"Response: {response}")

        return results
    

    