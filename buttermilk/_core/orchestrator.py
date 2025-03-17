import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Optional,
    Sequence,
)

import shortuuid
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
)

from buttermilk._core.agent import Agent
from buttermilk._core.config import DataSource, SaveInfo
from buttermilk._core.job import Job
from buttermilk._core.ui import IOInterface
from buttermilk.bm import BM

BASE_DIR = Path(__file__).absolute().parent



class Orchestrator(BaseModel, ABC):
    """ Runs a single instance of a flow, given an interface."""
    session_id: str = Field(default_factory=shortuuid.uuid, description="A unique session id for this set of flow runs.")
    bm: BM
    save: Optional[SaveInfo] = Field(default=None)
    data: Sequence[DataSource] = Field(default_factory=list)
    flow: Sequence[Agent] = Field(default_factory=list)
    interface: IOInterface

    _flow_data: dict[str, Any] = PrivateAttr(default_factory=dict, description="Memory for this specific flow run.")

    model_config = ConfigDict(
        extra = "forbid",
        arbitrary_types_allowed =False,
        populate_by_name = True,
        exclude_none = True,
        exclude_unset = True,
    )

    @abstractmethod
    async def run(self, request) -> None:
        """ Starts a flow, given an incoming request"""
        self._flow_data = copy.deepcopy(self.data)  # process if needed
        # add request data
        # ...
        for step in self.flow:
            self._flow_data[step.name] = await step(self._flow_data)

        # save the results
        # flow_data ...

        return None

    async def __call__(self, request) -> Job:
        return await self.run(request)
