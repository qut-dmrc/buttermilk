
import datetime
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import shortuuid
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    TypeAdapter,
    field_validator,
    model_validator,
    validator,
)

from buttermilk.utils import read_yaml

import datetime
from typing import Any, AsyncGenerator, Generator, Optional, Self, Type, Union

import numpy as np
import shortuuid
from cloudpathlib import CloudPath, GSPath
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    TypeAdapter,
    field_validator,
    model_validator,
)

class Score(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    measure: str
    score: Optional[float] = Field(default=None, validation_alias="scores")
    confidence: Optional[float|str] = None
    result: Optional[bool] = None
    labels: Optional[list[str]] = Field(default=[], validation_alias="label")
    reasons: list[str] = Field(default=[], validation_alias="reason")

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, labels: list[str]) -> list[str]:
        if labels:
            return [str(label).lower() for label in labels]
        return []



class Reasons(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    heading: str
    reasoning: str = Field(..., validation_alias=AliasChoices("reason", "reasons"))
    confidence: Optional[str] = None


class EvalRecord(BaseModel):
    eval_id: str = Field(default_factory=lambda: shortuuid.uuid())
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

    record_id: Optional[str] = None

    metadata: Optional[dict] = {}

    model: Optional[str] = None     # The model used for the evaluation
    process: Optional[str] = None   # The process followed for the evaluation
    standard: Optional[str] = None  # The standard used for the evaluation
    source: Optional[str] = None

    reasons: list[str] = Field(default=[], validation_alias="reason")
    predicted: Optional[Union[bool, None]] = Field(default=None, validation_alias="result")
    scores: list[Score] = []
    labels: list[str] = Field(default=[], validation_alias="label")

    error: Optional[str] = None
    response: Optional[str] = (
        None  # when we receive an invalid response, log it in this field
    )

    metadata: Optional[dict] = {}
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        use_enum_values=True,
        json_encoders={np.bool_: lambda v: bool(v)},
    )

    @field_validator("predicted")
    @classmethod
    def validate_result(cls, v) -> Union[bool, None]:
        if v is None:
            return None
        return TypeAdapter(bool).validate_python(v)

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, scores: list[Score]) -> list[Score]:
        if isinstance(scores, dict):
            scores = [Score(measure=k, score=v) for k, v in scores.items()]
        return scores

    @field_validator("reasons")
    @classmethod
    def validate_reasons(cls, reasons: list[Reasons]) -> list[Reasons]:
        converted_reasons = []
        for r in reasons:
            if isinstance(r, Reasons):
                converted_reasons.append(r)
            else:
                converted_reasons.append(Reasons(**r))
        return converted_reasons

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, labels: list[str]) -> list[str]:
        if labels:
            return [str(label).lower() for label in labels]
        return []

    @model_validator(mode="before")
    @classmethod
    def check_types(cls, values):
        if isinstance(values.get("scores"), dict):
            values["scores"] = [
                Score(measure=k, score=v) for k, v in values["scores"].items()
            ]

        if reasons := values.pop("reasons", None):
            converted_reasons = []
            for r in reasons:
                if isinstance(r, Reasons):
                    converted_reasons.append(r)
                else:
                    converted_reasons.append(Reasons(**r))
            values["reasons"] = converted_reasons

        return values

    def to_record(self) -> dict:
        record = self.model_dump()

        # add booleans
        if self.predicted is not None:
            record["predicted"] = self.predicted

        if self.scores and isinstance(self.scores, list):
            record["scores"] = [
                {"measure": s.measure, "score": s.score} for s in self.scores
            ]

        # ensure timestamp is in pandas format.
        record["timestamp"] = pd.to_datetime(record["timestamp"])

        return record

