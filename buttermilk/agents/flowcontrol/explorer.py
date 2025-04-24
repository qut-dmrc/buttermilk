from typing import Optional

from pydantic import BaseModel

from buttermilk._core.contract import (
    ConductorRequest,
    StepRequest,
)
from buttermilk.agents.flowcontrol.host import LLMHostAgent

TRUNCATE_LEN = 1000  # characters per history message


class ExplorerHost(LLMHostAgent):
    _output_model: Optional[type[BaseModel]] = StepRequest

    async def _choose(self, message: ConductorRequest) -> StepRequest:
        step = await self._process(message=message)
        return step
