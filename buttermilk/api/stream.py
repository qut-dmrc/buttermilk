import asyncio
import json
from collections.abc import AsyncGenerator, Sequence
from typing import (
    Self,
)
from urllib.parse import parse_qs

import shortuuid
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from rich import print as rprint

from buttermilk._core.agent import Agent
from buttermilk._core.log import logger
from buttermilk._core.runner_types import (
    Job,
    RecordInfo,
)
from buttermilk.bm import BM
from buttermilk.llms import CHATMODELS
from buttermilk.runner.creek import Creek
from buttermilk.utils.media import (
    validate_uri_extract_text,
    validate_uri_or_b64,
)
from buttermilk.utils.validators import make_list_validator

bm = None


class FlowRequest(BaseModel):
    flow_id: str = Field(default_factory=shortuuid.uuid, init=False)

    model: str | Sequence[str] | None = None
    template: str | Sequence[str] | None = None
    template_vars: dict | Sequence[dict] | None = Field(default_factory=list)
    record_id: str | None = None
    content: str | None = Field(
        default=None,
        validation_alias=AliasChoices("content", "text", "body"),
    )
    uri: str | bytes | None = Field(
        default=None,
        validation_alias=AliasChoices("uri", "url", "link"),
    )
    source: str | Sequence[str] | None = None
    video: str | bytes | None = None
    image: str | bytes | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=False,
        extra="allow",
        populate_by_name=True,
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
                values.get("image"),
                values.get("video"),
            ],
        ):
            raise ValueError(
                "At least one of content, text, uri, video or image must be provided.",
            )

        return values

    @field_validator("content", "image", "video", mode="before")
    def sanitize_strings(cls, v):
        if v:
            return v.strip()
        return v

    @model_validator(mode="after")
    def check_values(self) -> Self:
        if self.model:
            for v in self.model:
                if v not in CHATMODELS:
                    raise ValueError(f"Valid model must be provided. {v} is unknown.")

        # Add any additional vars into the template_vars dict.
        extras = []
        if self.template_vars and self.model_extra:
            # Template variables is a list of dicts that are run in combinations
            # When we add other variables, make sure to add them to each existing combination
            for existing_vars in self.template_vars:
                existing_vars.update(self.model_extra)
                extras.append(existing_vars)
        elif self.model_extra:
            self.template_vars = [self.model_extra]

        return self


async def flow_stream(
    flow: Creek,
    flow_request: FlowRequest,
    return_json=True,
) -> AsyncGenerator[str, None]:
    bm = BM()
    logger.info(
        f"Received request for flow {flow} and flow_request {flow_request}",
    )
    content, image, video = await asyncio.gather(
        validate_uri_extract_text(flow_request.content),
        validate_uri_or_b64(flow_request.image),
        validate_uri_or_b64(flow_request.video),
    )
    media = [x for x in [image, video] if x]
    if flow_request.record_id:
        record = RecordInfo(text=content, media=media, record_id=flow_request.record_id)
    else:
        record = RecordInfo(text=content, media=media)
    async for data in flow.run_flows(
        record=record,
        run_info=bm._run_metadata,
        source=flow_request.source,
    ):
        if data:
            rprint(data)
            if return_json:
                data = data.model_dump_json()
            yield data
