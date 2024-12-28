import asyncio
import json
from collections.abc import AsyncGenerator, Sequence
from urllib.parse import parse_qs

import shortuuid
from fastapi import HTTPException
from pydantic import (AliasChoices, AnyUrl, BaseModel, ConfigDict, Field,
                      PrivateAttr, field_validator, model_validator)
from rich import print as rprint

from buttermilk._core.agent import Agent
from buttermilk._core.log import logger
from buttermilk._core.runner_types import Job, MediaObj, RecordInfo
from buttermilk.bm import BM
from buttermilk.llms import CHATMODELS
from buttermilk.runner.flow import Flow
from buttermilk.utils.media import download_and_convert
from buttermilk.utils.validators import (make_list_validator,
                                         make_uri_validator, sanitize_html)

bm = None


class FlowRequest(BaseModel):
    flow_id: str = Field(default_factory=shortuuid.uuid, init=False)

    model: str | Sequence[str] | None = None
    template: str | Sequence[str] | None = None
    template_vars: dict | Sequence[dict] | None = Field(default_factory=list)
    q: str | None = Field(
        default=None,
        validation_alias=AliasChoices("q", "query", "question", "prompt"),
    )
    record_id: str | None = None
    text: str | None = Field(
        default=None,
        validation_alias=AliasChoices("text", "body"),
    )
    uri: str | None = Field(
        default=None,
        validation_alias=AliasChoices("uri", "url", "link"),
    )
    content: bytes | None = Field(default=None, validation_alias=AliasChoices("content", "video","image"))
    source: str | Sequence[str] = []

    model_config = ConfigDict(
        arbitrary_types_allowed=False,
        extra="allow",
        populate_by_name=True,
        exclude_none=True,
        exclude_unset=True,
        use_enum_values=True,
    )
    _client: Agent = PrivateAttr()
    _job: Job = PrivateAttr()

    _ensure_list = field_validator(
        "model",
        "template",
        "template_vars",
        "source",
        mode="before",
    )(
        make_list_validator(),
    )

    _ensure_uri = field_validator("uri", mode="before")(make_uri_validator())

    @model_validator(mode="before")
    def preprocess_data(cls, values):
        # Check first if the values are coming as utf-8 encoded bytes
        try:
            values = values.decode("utf-8")
        except:
            pass
        try:
            # Might be HTML form data
            values = parse_qs(values)

            # Convert the values from lists to single values
            values = {key: value[0] for key, value in values.items()}
        except:
            pass

        try:
            # might be JSON
            values = json.loads(values)
        except:
            pass

        if not any(
            [
                values.get("content"),
                values.get("uri"),
                values.get("text"),
                values.get("q"),
            ],
        ):
            raise ValueError(
                "At least one of query, content, text, or uri must be provided.",
            )

        return values

    @field_validator("q", "uri", "text", mode="before")
    def sanitize_strings(cls, v):
        if v:
            v = v.strip()
            v = sanitize_html(v) 
        return v

    @model_validator(mode="after")
    def check_values(self) -> "FlowRequest":
        if self.model:
            for v in self.model:
                if v not in CHATMODELS:
                    raise ValueError(f"Valid model must be provided. {v} is unknown.")

        if isinstance(self.text, AnyUrl):
            if self.uri:
                raise ValueError("You should only pass one URL in at a time.")
            else:
                # Move the URL to the correct field
                self.uri = self.text
                self.text = None

        return self


async def flow_stream(
    flow: Flow,
    flow_request: FlowRequest,
    return_json=True,
) -> AsyncGenerator[str, None]:
    bm = BM()
    if not flow_request.source:
        flow_request.source = [bm.cfg.job]
    logger.info(
        f"Received request for flow {flow} and flow_request {flow_request}. Resolving media URIs.",
    )
    
    objects = await asyncio.gather(
        download_and_convert(flow_request.content),
        download_and_convert(flow_request.uri),
    )

    media = [x for x in objects if x and isinstance(x, MediaObj)]
    content = "\n".join([x for x in objects if isinstance(x, str)])
    record = None
    if media or content:
        if flow_request.record_id:
            record = RecordInfo(
                text=content, media=media, record_id=flow_request.record_id
            )
        else:
            record = RecordInfo(text=content, media=media)

    async for data in flow.run_flows(
        record=record,
        run_info=bm._run_metadata,
        source=flow_request.source,
        flow_id=flow_request.flow_id,
        q=flow_request.q,
    ):
        if data:
            if data.error:
                raise HTTPException(status_code=500, detail=str(data.error))
            if data.outputs:
                if return_json:
                    yield data.outputs.model_dump_json()
                else:
                    yield data.outputs
                rprint(data.outputs)
            else:
                logger.info(f"No data to return from {flow} (completed successfully).")
