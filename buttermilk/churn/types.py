from typing import Optional
from pydantic import BaseModel, Field
import shortuuid
import datetime

class AgentInfo(BaseModel):
    agent_id: str
    agent_version: str
    parameters: dict

class RunInfo(BaseModel):
    run_id: str
    experiment_name: str
    parameters: dict

class PredictionResult(BaseModel):
    predicted_class: Optional[str|int] = None
    predicted_result: Optional[float] = None
    labels: Optional[list[str]] = None
    confidence: Optional[float] = None

class PredictionInputs(BaseModel):
    record_id: str
    parameters: dict

## An individual result
class Prediction(BaseModel):
    prediction_id: str = Field(default_factory=lambda: str(shortuuid.ShortUUID().uuid()))
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now(tz=datetime.timezone.utc))
    agent_info: AgentInfo
    run_info: RunInfo
    outputs: PredictionResult
    inputs: PredictionInputs
