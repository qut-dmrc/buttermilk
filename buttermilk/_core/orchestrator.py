import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Optional,
    Sequence,
)

import shortuuid
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
)

from buttermilk._core.config import DataSource, SaveInfo
from buttermilk._core.contract import AgentOutput
from buttermilk._core.flow import FlowVariableRouter
from buttermilk._core.job import Job
from buttermilk._core.ui import IOInterface
from buttermilk._core.variants import AgentVariants

BASE_DIR = Path(__file__).absolute().parent



class Orchestrator(BaseModel, ABC):
    """ Runs a single instance of a flow, given an interface."""
    session_id: str = Field(default_factory=shortuuid.uuid, description="A unique session id for this set of flow runs.")
    save: Optional[SaveInfo] = Field(default=None)
    data: Sequence[DataSource] = Field(default_factory=list)
    steps: Sequence[AgentVariants]= Field(default_factory=list, description="A sequence of agent factories to run in order")
    interface: IOInterface

    _flow_data: FlowVariableRouter = PrivateAttr(default_factory=FlowVariableRouter)
    _records: list = PrivateAttr(default_factory=list)

    model_config = ConfigDict(
        extra = "forbid",
        arbitrary_types_allowed =False,
        populate_by_name = True,
        exclude_none = True,
        exclude_unset = True,
    )

    @field_validator("steps", mode="before")
    @classmethod
    def validate_steps(cls, value):
        if isinstance(value, Sequence) and not isinstance(value, str):
            return [AgentVariants(**step) if not isinstance(step, AgentVariants) else step
                   for step in value]
        return value

    @abstractmethod
    async def run(self, request=None) -> None:
        """ Starts a flow, given an incoming request"""
        self._flow_data = copy.deepcopy(self.data)  # process if needed
        # add request data
        # ...
        for step in self.flow:
            self._flow_data[step.name] = await step(self._flow_data)

        # save the results
        # flow_data ...

        return None

    async def __call__(self, request=None) -> Job:
        return await self.run(request=request)

    async def store_results(self, step: str, result: AgentOutput):
        if not result.error:
            self._flow_data.add(key=step, value=result)

            # Harvest records
            self._records.extend(result.records)
