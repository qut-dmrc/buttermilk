
import datetime

import numpy as np
import pandas as pd
import shortuuid
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from buttermilk.defaults import COL_PREDICTION


class Score(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    measure: str
    score: float | None = Field(default=None, validation_alias="scores")
    confidence: float | str | None = None
    severity: float | None = None
    result: bool | None = None
    labels: list[str] | None = Field(default=[], validation_alias="label")
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
    confidence: str | None = None


class EvalRecord(BaseModel):
    eval_id: str = Field(default_factory=lambda: shortuuid.uuid())
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

    record_id: str | None = None

    metadata: dict | None = {}

    model: str | None = None  # The model used for the evaluation
    process: str | None = None  # The process followed for the evaluation
    standard: str | None = None  # The standard used for the evaluation
    source: str | None = None

    reasons: list[Reasons] = Field(default=[], validation_alias="reason")
    prediction: bool | None = Field(default=None, validation_alias="result")
    scores: list[Score] = []
    labels: list[str] = Field(default=[], validation_alias="label")

    error: str | None = None
    response: str | None = (
        None  # when we receive an invalid response, log it in this field
    )

    metadata: dict | None = {}
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        use_enum_values=True,
        json_encoders={np.bool_: lambda v: bool(v)},
    )

    @field_validator("prediction")
    @classmethod
    def validate_result(cls, v) -> bool | None:
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
            elif isinstance(r, str):
                converted_reasons.append(Reasons(heading="unknown", reasoning=r))
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
        if self.prediction is not None:
            record[COL_PREDICTION] = self.prediction

        if self.scores and isinstance(self.scores, list):
            record["scores"] = [
                {"measure": s.measure, "score": s.score} for s in self.scores
            ]

        # ensure timestamp is in pandas format.
        record["timestamp"] = pd.to_datetime(record["timestamp"])

        return record
