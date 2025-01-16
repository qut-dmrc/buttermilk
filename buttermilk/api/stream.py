import json
from collections.abc import AsyncGenerator, Sequence
from urllib.parse import parse_qs

import shortuuid
from pydantic import (
    AliasChoices,
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from buttermilk._core.agent import Agent
from buttermilk._core.log import logger
from buttermilk._core.runner_types import Job, RecordInfo
from buttermilk.bm import BM
from buttermilk.llms import CHATMODELS
from buttermilk.runner.flow import Flow
from buttermilk.utils.validators import (
    make_list_validator,
    make_uri_validator,
    sanitize_html,
)

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
    uri: str | None = Field(
        default=None,
        validation_alias=AliasChoices("uri", "url", "link"),
    )
    content: bytes | None = Field(
        default=None,
        validation_alias=AliasChoices("text", "body", "content", "video", "image"),
    )
    mime_type: str | None = None
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

    @field_validator("q", mode="before")
    def sanitize_strings(cls, v):
        if v and isinstance(v, str):
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
    record = None
    if flow_request.uri:
        record = await RecordInfo.from_uri(
            flow_request.uri, mimetype=flow_request.mime_type
        )
    elif flow_request.content:
        record = await RecordInfo.from_object(
            flow_request.content, mimetype=flow_request.mime_type
        )

    if record and flow_request.record_id:
        record.record_id = flow_request.record_id
    elif record:
        raise NotImplementedError("Loading by record ID is not yet supported.")

    async for job in flow.run_flows(
        record=record,
        run_info=bm.run_info,
        source=flow_request.source,
        flow_id=flow_request.flow_id,
        q=flow_request.q,
    ):
        if job:
            # if data.error:
            #     raise HTTPException(status_code=500, detail=str(data.error))
            if job.outputs:
                if return_json:
                    yield job.model_dump_json()
                else:
                    yield job
            else:
                logger.info(f"No data to return from {flow} (completed successfully).")
            # update record in case it has been changed
            if job.record:
                record = job.record
